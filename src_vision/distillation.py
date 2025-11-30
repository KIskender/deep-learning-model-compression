import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import models
import torch.nn.functional as F
import os
import time
import zipfile

# --- AYARLAR ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TEACHER_PATH = "models/teacher_resnet18.pth"
STUDENT_PATH = "models/student_distilled.pth"

# Hiperparametreler
EPOCH_SAYISI = 50     # Öğrencinin öğrenme süresi
TEMPERATURE = 4       # Öğretmenin bilgisini yumuşatma katsayısı (Genelde 2-5 arası)
ALPHA = 0.7           # Öğretmene ne kadar güveneceği (0.7 = %70 Öğretmen, %30 Etiketler)
BATCH_SIZE = 64

print(f"--- Knowledge Distillation Başlıyor ---")
print(f"Cihaz: {device}")

# --- 1. MODELLER ---

# A. TEACHER (Öğretmen) - ResNet18
def load_teacher():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 10)
    # Teacher daha önce eğitilmiş olmalı
    if not os.path.exists(TEACHER_PATH):
        raise FileNotFoundError("Öğretmen model dosyası bulunamadı! Önce main.py çalıştırın.")
    model.load_state_dict(torch.load(TEACHER_PATH, map_location=device))
    model.to(device)
    model.eval() # Öğretmen öğrenmez, sadece öğretir
    return model

# B. STUDENT (Öğrenci) - Basit ve Küçük Bir CNN
# ResNet-18'den çok daha az katmanlı, kendi tasarımım
class LightCNN(nn.Module):
    def __init__(self):
        super(LightCNN, self).__init__()
        # Basit 3 katmanlı Konvolüsyon
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 32x32 -> 16x16
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 16x16 -> 8x8
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # 8x8 -> 4x4
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 128 * 4 * 4)
        x = self.classifier(x)
        return x

# --- 2. VERİ SETİ ---
print("[1/4] Veri Seti Yükleniyor...")
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=0)

# --- 3. DAMITMA FONKSİYONU (Knowledge Distillation Loss) ---
def distillation_loss(student_logits, teacher_logits, labels, T, alpha):
    # 1. Soft Target Loss (Öğretmeni taklit etme kaybı)
    # KLDivLoss: İki olasılık dağılımı arasındaki farkı ölçer
    soft_loss = nn.KLDivLoss(reduction='batchmean')(
        F.log_softmax(student_logits / T, dim=1),
        F.softmax(teacher_logits / T, dim=1)
    ) * (T * T)
    
    # 2. Hard Target Loss (Gerçek etiketleri bilme kaybı)
    hard_loss = F.cross_entropy(student_logits, labels)
    
    # Toplam Kayıp: İkisinin ağırlıklı toplamı
    return alpha * soft_loss + (1. - alpha) * hard_loss

# --- 4. EĞİTİM DÖNGÜSÜ ---
teacher_model = load_teacher()
student_model = LightCNN().to(device)
optimizer = optim.Adam(student_model.parameters(), lr=0.001)

print(f"\n[2/4] Öğretmen-Öğrenci Eğitimi Başlıyor ({EPOCH_SAYISI} Epoch)...")
start_train = time.time()

for epoch in range(EPOCH_SAYISI):
    student_model.train()
    running_loss = 0.0
    
    for inputs, labels in trainloader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # 1. Öğretmene sor (Gradient hesaplanmaz, o sadece referans)
        with torch.no_grad():
            teacher_logits = teacher_model(inputs)
            
        # 2. Öğrenci tahmin yapsın
        optimizer.zero_grad()
        student_logits = student_model(inputs)
        
        # 3. Kaybı hesapla (Öğretmen + Etiket karışık)
        loss = distillation_loss(student_logits, teacher_logits, labels, TEMPERATURE, ALPHA)
        
        # 4. Öğrenciyi güncelle
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
    print(f"Epoch {epoch+1}/{EPOCH_SAYISI} | Loss: {running_loss/len(trainloader):.4f}")

print(f"Eğitim Tamamlandı! Süre: {(time.time() - start_train)/60:.1f} dk")
torch.save(student_model.state_dict(), STUDENT_PATH)

# --- 5. SONUÇ ÖLÇÜMÜ ---
def evaluate_and_measure(model, name):
    model.eval()
    correct = 0
    total = 0
    start = time.time()
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    duration = time.time() - start
    
    # Boyut ölçümü (Ziplenmiş)
    torch.save(model.state_dict(), "temp.pth")
    with zipfile.ZipFile("temp.zip", 'w', compression=zipfile.ZIP_DEFLATED) as f:
        f.write("temp.pth")
    size_mb = os.path.getsize("temp.zip") / (1024 * 1024)
    os.remove("temp.pth")
    os.remove("temp.zip")
    
    return 100 * correct / total, size_mb

print("\n[3/4] Karşılaştırma Yapılıyor...")
acc_teacher, size_teacher = evaluate_and_measure(teacher_model, "Teacher (ResNet18)")
acc_student, size_student = evaluate_and_measure(student_model, "Student (LightCNN)")

print("\n" + "="*60)
print(f"{'METRİK':<15} | {'TEACHER (ResNet18)':<20} | {'STUDENT (LightCNN)':<20}")
print("-" * 65)
print(f"{'Boyut (MB)':<15} | {size_teacher:<20.2f} | {size_student:<20.2f}")
print(f"{'Doğruluk (%)':<15} | {acc_teacher:<20.2f} | {acc_student:<20.2f}")
print(f"{'Küçülme Oranı':<15} | {'-':<20} | {size_teacher/size_student:.1f} KAT DAHA KÜÇÜK!")
print("="*60)