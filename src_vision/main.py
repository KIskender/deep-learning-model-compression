import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import models
import torch.nn.utils.prune as prune
import os
import zipfile
import time

# --- AYARLAR ---
NUM_WORKERS = 0 
EPOCH_SAYISI = 20
BATCH_SIZE = 64
# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# FILENAME = os.path.join(BASE_DIR, "models", "teacher_resnet18.pth")
FILENAME = "models/teacher_resnet18.pth"

# Cihaz ayarı
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("="*40)
print(f" İŞLEM CİHAZI: {device}")
if torch.cuda.is_available():
    print(f" EKRAN KARTI: {torch.cuda.get_device_name(0)}")
print("="*40)

# --- VERİ SETİ ---
print("\n[1/4] Veri Seti Hazırlanıyor...")
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
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=NUM_WORKERS)

# --- MODEL VE FONKSİYONLAR ---
def create_model():
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, 10)
    model = model.to(device)
    return model

def evaluate_model(model, loader, description="Model"):
    model.eval()
    correct = 0
    total = 0
    start = time.time()
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    duration = time.time() - start
    avg_inference_time_ms = (duration / total) * 1000
    accuracy = 100 * correct / total
    return accuracy, duration

def get_zipped_size(model, filename="models/temp_model.pth"):
    torch.save(model.state_dict(), filename)
    zip_name = filename + ".zip"
    with zipfile.ZipFile(zip_name, 'w', compression=zipfile.ZIP_DEFLATED) as f:
        f.write(filename)
    size_mb = os.path.getsize(zip_name) / (1024 * 1024)
    if os.path.exists(filename): os.remove(filename)
    if os.path.exists(zip_name): os.remove(zip_name)
    return size_mb

# --- ANA İŞLEM ---

# 1. Eğitim
model = create_model()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

print(f"\n[2/4] Eğitim Başlıyor ({EPOCH_SAYISI} Epoch)...")
print("-" * 60)
start_train = time.time()

model.train()
for epoch in range(EPOCH_SAYISI):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        loss = criterion(model(inputs), labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}/{EPOCH_SAYISI} | Loss: {running_loss/len(trainloader):.4f}")

print("-" * 60)
print(f"Eğitim Tamamlandı! Süre: {(time.time() - start_train)/60:.1f} dk")

torch.save(model.state_dict(), FILENAME)
print(f"Model dosyaya kaydedildi: {FILENAME}")

# 2. Baseline Ölçümü
print(f"\n[3/4] Baseline Performansı Ölçülüyor...")
acc_base, time_base = evaluate_model(model, testloader, "BASELINE")
size_base = get_zipped_size(model, "baseline_temp.pth") 

# 3. Pruning Uygulama
print(f"\n[4/4] Pruning ve Raporlama...")
parameters_to_prune = []
for module in model.modules():
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
        parameters_to_prune.append((module, 'weight'))

prune.global_unstructured(
    parameters_to_prune,
    pruning_method=prune.L1Unstructured,
    amount=0.3,
)
for module, _ in parameters_to_prune:
    prune.remove(module, 'weight')

acc_pruned, time_pruned = evaluate_model(model, testloader, "PRUNED")
size_pruned_zip = get_zipped_size(model, "pruned_temp.pth")

print("\n" + "="*40)
print(f"DOĞRULUK : %{acc_base:.2f} --> %{acc_pruned:.2f}")
print(f"BOYUT    : {size_base:.2f} MB --> {size_pruned_zip:.2f} MB")
print("="*40)