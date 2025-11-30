import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import os
import time
import copy
import warnings

warnings.filterwarnings("ignore")

# --- AYARLAR ---
device = torch.device("cpu") # Quantization CPU işidir
STUDENT_PATH = "models/student_distilled.pth"
BACKEND = "fbgemm"

print(f"--- Öğrenci Modeli Sıkıştırılıyor (Combo: Distillation + Quantization) ---")
print(f"Backend: {BACKEND}")

# --- 1. MODEL TANIMI (DÜZELTİLDİ) ---
class LightCNN(nn.Module):
    def __init__(self):
        super(LightCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.features(x)
        # .view() yerine .reshape() kullanıyoruz. 
        # Quantization sonrası bellek yapısı değiştiği için reshape güvenlidir.
        x = x.reshape(-1, 128 * 4 * 4) 
        # -----------------------
        x = self.classifier(x)
        return x

# --- 2. QUANTIZATION WRAPPER ---
class QuantizedLightCNN(nn.Module):
    def __init__(self, model_fp32):
        super(QuantizedLightCNN, self).__init__()
        self.quant = torch.quantization.QuantStub()
        self.model_fp32 = model_fp32
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.model_fp32(x)
        x = self.dequant(x)
        return x

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
    print(f"[{description}] -> Acc: %{accuracy:.2f} | Hız: {avg_speed:.4f} ms")
    return accuracy

def get_size(model, path="models/temp_student.pth"):
    torch.save(model.state_dict(), path)
    size_mb = os.path.getsize(path) / (1024 * 1024)
    if os.path.exists(path): os.remove(path)
    return size_mb

# --- İŞLEM AKIŞI ---

# 1. Float32 Öğrenciyi Yükle
if not os.path.exists(STUDENT_PATH):
    raise FileNotFoundError("Öğrenci modeli yok! distillation.py çalıştırın.")

student_fp32 = LightCNN()
student_fp32.load_state_dict(torch.load(STUDENT_PATH, map_location=device))
student_fp32.to(device)
student_fp32.eval()

print("\n1. Orijinal Öğrenci (Float32) Ölçülüyor...")
acc_fp32 = evaluate_model(student_fp32, testloader, "STUDENT FP32")
size_fp32 = get_size(student_fp32)

# 2. Quantization Hazırlığı
print("\n2. Quantization (INT8) Uygulanıyor...")
# Kapsüle koy
model_q = QuantizedLightCNN(student_fp32)
model_q.to(device)
model_q.eval()

# Backend ayarı
torch.backends.quantized.engine = BACKEND
model_q.qconfig = torch.quantization.get_default_qconfig(BACKEND)
torch.quantization.prepare(model_q, inplace=True)

# 3. Kalibrasyon
print("   -> Kalibrasyon (200 örnek)...")
evaluate_model(model_q, testloader, "Kalibrasyon", limit=200)

# 4. Dönüştürme
print("   -> Dönüştürülüyor...")
torch.quantization.convert(model_q, inplace=True)

# 5. Sonuç
print("\n3. Sıkıştırılmış Öğrenci (INT8) Ölçülüyor...")
acc_int8 = evaluate_model(model_q, testloader, "STUDENT INT8")
size_int8 = get_size(model_q)

print("\n" + "="*50)
print(f"FİNAL SONUÇ (DISTILLATION + QUANTIZATION)")
print("="*50)
print(f"Orijinal Teacher Boyutu : ~40.00 MB")
print(f"Öğrenci (Distilled)     :  {size_fp32:.2f} MB")
print(f"Öğrenci (Quantized)     :  {size_int8:.2f} MB")
print("-" * 50)
print(f"TOPLAM KÜÇÜLME          : {40.0/size_int8:.1f} KAT!")
print(f"SON DOĞRULUK            : %{acc_int8:.2f}")
print("="*50)