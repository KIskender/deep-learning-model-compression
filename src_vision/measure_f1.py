import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision import models
from sklearn.metrics import f1_score, accuracy_score, classification_report
import os
import copy
import warnings

warnings.filterwarnings("ignore")

# --- AYARLAR ---
device = torch.device("cpu")
TEACHER_PATH = "models/teacher_resnet18.pth"
STUDENT_PATH = "models/student_distilled.pth"

print("--- F1 Skor ve Detaylı Performans Ölçümü ---")

# --- VERİ SETİ ---
# F1 ölçümü için tüm test setini (10.000 resim) kullanacağız
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=0)

# --- MODEL TANIMLARI ---
# 1. Student Model Mimarisi (LightCNN)
class LightCNN(nn.Module):
    def __init__(self):
        super(LightCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, 256), nn.ReLU(), nn.Dropout(0.1), nn.Linear(256, 10)
        )
    def forward(self, x):
        x = self.features(x)
        x = x.reshape(-1, 128 * 4 * 4) # .view yerine .reshape kullandık
        x = self.classifier(x)
        return x

# 2. Quantization Wrapper
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

# --- ÖLÇÜM FONKSİYONU ---
def get_metrics(model, name):
    print(f"\n[{name}] Test Ediliyor...")
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    # Metrikleri Hesapla
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted') # 'weighted': Sınıf ağırlıklı ortalama
    
    print(f"   -> Accuracy (Doğruluk): %{acc*100:.2f}")
    print(f"   -> F1 Score           : {f1:.4f}")
    
    return acc, f1

# --- 1. TEACHER (ResNet18) ÖLÇÜMÜ ---
teacher = models.resnet18(weights=None)
teacher.fc = nn.Linear(teacher.fc.in_features, 10)
if os.path.exists(TEACHER_PATH):
    teacher.load_state_dict(torch.load(TEACHER_PATH, map_location=device))
    teacher.to(device)
    get_metrics(teacher, "Teacher (ResNet18)")
else:
    print("Teacher model dosyası bulunamadı!")

# --- 2. STUDENT (Quantized) ÖLÇÜMÜ ---
if os.path.exists(STUDENT_PATH):
    # Önce Float32 yükle
    student_fp32 = LightCNN()
    student_fp32.load_state_dict(torch.load(STUDENT_PATH, map_location=device))
    
    # Sonra Quantize et (Canlı)
    backend = "fbgemm"
    torch.backends.quantized.engine = backend
    student_q = QuantizedLightCNN(copy.deepcopy(student_fp32))
    student_q.qconfig = torch.quantization.get_default_qconfig(backend)
    torch.quantization.prepare(student_q, inplace=True)
    # Kalibrasyon (Hızlıca 1 batch)
    first_batch = next(iter(testloader))[0].to(device)
    student_q(first_batch)
    torch.quantization.convert(student_q, inplace=True)
    
    student_q.to(device)
    get_metrics(student_q, "Student (Combo: Distilled + Quantized)")
else:
    print("Student model dosyası bulunamadı!")

print("\nİşlem Tamamlandı.")