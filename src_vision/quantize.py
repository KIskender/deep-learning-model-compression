import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision import models
# Özel Quantization kütüphanesini dahil ediyoruz
from torchvision.models.quantization import resnet18 as quantized_resnet18
import os
import time
import copy
import warnings

# Gereksiz uyarıları gizle
warnings.filterwarnings("ignore")

# --- AYARLAR ---
device = torch.device("cpu") 
BASELINE_PATH = "models/teacher_resnet18.pth"
BACKEND = "fbgemm" # Windows/Intel CPU için en uygun motor

print(f"--- Quantization İşlemi Başlıyor ---")
print(f"İşlem Cihazı: {device}")
print(f"Backend Motoru: {BACKEND}")

# --- VERİ SETİ ---
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=0)

# --- YARDIMCI FONKSİYONLAR ---
def evaluate_model(model, loader, description="Model", limit=1000):
    correct = 0
    total = 0
    start = time.time()
    
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(loader):
            if i >= limit: break 
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    duration = time.time() - start
    accuracy = 100 * correct / total
    avg_speed = (duration / total) * 1000 
    
    print(f"[{description}]")
    print(f"   -> Doğruluk (İlk {limit} örnek): %{accuracy:.2f}")
    print(f"   -> Örnek Başına Hız: {avg_speed:.4f} ms")
    return accuracy, avg_speed

def get_size(model, path="temp.pth"):
    # Quantized modelleri kaydederken script kullanmak daha güvenlidir
    torch.save(model.state_dict(), path)
    size_mb = os.path.getsize(path) / (1024 * 1024)
    if os.path.exists(path): os.remove(path)
    return size_mb

# --- 1. BASELINE MODELİ HAZIRLAMA ---
print("\n1. Baseline Model (FP32) Hazırlanıyor...")

if not os.path.exists(BASELINE_PATH):
    raise FileNotFoundError("Model dosyası bulunamadı! Önce main.py'yi çalıştırın.")

# A. Standart modeli oluşturup ağırlıkları yüklüyoruz
state_dict = torch.load(BASELINE_PATH, map_location=device)

# B. Şimdi "Quantization-Ready" bir ResNet oluşturuyoruz
# Bu model, 'add' işlemleri için FloatFunctional kullanır, hata vermez.
model_to_quantize = quantized_resnet18(pretrained=False, quantize=False)
model_to_quantize.fc = nn.Linear(model_to_quantize.fc.in_features, 10)

# Ağırlıkları aktar
model_to_quantize.load_state_dict(state_dict)
model_to_quantize.to(device)
model_to_quantize.eval()

# Baseline Performansını Ölç
# (Modeli henüz quantize etmedik, sadece yapısını değiştirdik, performansı aynı olmalı)
acc_fp32, speed_fp32 = evaluate_model(model_to_quantize, testloader, "BASELINE (FP32)", limit=1000)
size_fp32 = get_size(model_to_quantize)
print(f"   -> Boyut: {size_fp32:.2f} MB")

# --- 2. FUSION (Birleştirme) ---
# Conv+BN+ReLU katmanlarını tek bir işlemde birleştirir (Hız artırır)
print("\n2. Fusion (Katman Birleştirme) Uygulanıyor...")
model_to_quantize.fuse_model()

# --- 3. QUANTIZATION HAZIRLIĞI ---
print("3. Quantization Hazırlanıyor (INT8)...")

# Backend Ayarı
torch.backends.quantized.engine = BACKEND
model_to_quantize.qconfig = torch.quantization.get_default_qconfig(BACKEND)

# Modeli Hazırla (Insert Observers)
torch.quantization.prepare(model_to_quantize, inplace=True)

# --- 4. CALIBRATION (Kalibrasyon) ---
print("   -> Kalibrasyon yapılıyor (İlk 200 örnek)...")
evaluate_model(model_to_quantize, testloader, "Kalibrasyon Süreci", limit=200)

# --- 5. CONVERT (Dönüştürme) ---
print("   -> INT8 formatına dönüştürülüyor...")
torch.quantization.convert(model_to_quantize, inplace=True)

# --- 6. QUANTIZED MODEL ÖLÇÜMÜ ---
print("\n4. Quantized Model (INT8) Test Ediliyor...")
acc_int8, speed_int8 = evaluate_model(model_to_quantize, testloader, "QUANTIZED (INT8)", limit=1000)
size_int8 = get_size(model_to_quantize)
print(f"   -> Boyut: {size_int8:.2f} MB")

# --- SONUÇ TABLOSU ---
print("\n" + "="*60)
print(f"{'METRİK':<15} | {'BASELINE':<15} | {'QUANTIZED':<15} | {'DEĞİŞİM':<10}")
print("-" * 65)
print(f"{'Boyut (MB)':<15} | {size_fp32:<15.2f} | {size_int8:<15.2f} | x{size_fp32/size_int8:.1f} Küçüldü")
print(f"{'Hız (ms)':<15} | {speed_fp32:<15.4f} | {speed_int8:<15.4f} | x{speed_fp32/speed_int8:.1f} Hızlandı")
print(f"{'Doğruluk (%)':<15} | {acc_fp32:<15.2f} | {acc_int8:<15.2f} | Fark: {acc_int8-acc_fp32:.2f}")
print("="*60)