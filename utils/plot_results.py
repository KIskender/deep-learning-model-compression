import matplotlib.pyplot as plt
import numpy as np

# --- PROJE SONUÇ VERİLERİ) ---
models = ['Teacher\n(ResNet18)', 'Pruned\n(ResNet18)', 'Quantized\n(ResNet18)', 'Student\n(Distilled)', 'Combo\n(Dist.+Quant.)']

# Boyutlar (MB)
sizes = [39.70, 30.85, 10.78, 2.38, 0.62]

# Doğruluklar (%)
# (Teacher Baseline: 84.10, Pruned: 83.63, Quantized Teacher: 84.30, Student FP32: 83.88, Student INT8: 83.50)
accuracies = [81.30, 81.36, 79.00, 80.50, 80.40]

# Hızlar (ms) - CPU Latency
latencies = [5.50, 11.37, 3.27, 0.90, 0.99]

# --- GRAFİK AYARLARI ---
plt.style.use('ggplot')
colors = ['#3498db', '#9b59b6', '#e67e22', '#2ecc71', '#e74c3c']


# 1. BOYUT KARŞILAŞTIRMASI (BAR CHART)
plt.figure(figsize=(10, 6))
bars = plt.bar(models, sizes, color=colors)
plt.title('Model Boyutu Karşılaştırması (Daha Düşük Daha İyi)', fontsize=14)
plt.ylabel('Boyut (MB)', fontsize=12)
plt.xlabel('Yöntemler', fontsize=12)

# Değerleri yaz
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.5, f'{yval} MB', ha='center', va='bottom', fontweight='bold')

plt.savefig('results/grafik_1_boyut.png', dpi=300)
print("Grafik 1 (Boyut) oluşturuldu.")

# 2. DOĞRULUK KARŞILAŞTIRMASI (BAR CHART)
plt.figure(figsize=(10, 6))
plt.ylim(70, 90) # Farkları göstermek için eksen daraltıldı
bars = plt.bar(models, accuracies, color=colors)
plt.title('Doğruluk (Accuracy) Karşılaştırması (Daha Yüksek Daha İyi)', fontsize=14)
plt.ylabel('Doğruluk (%)', fontsize=12)

# Değerleri yaz
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.5, f'%{yval}', ha='center', va='bottom', fontweight='bold')

plt.savefig('results/grafik_2_dogruluk.png', dpi=300)
print("Grafik 2 (Doğruluk) oluşturuldu.")

# 3. HIZ (LATENCY) KARŞILAŞTIRMASI (BAR CHART)
plt.figure(figsize=(10, 6))
bars = plt.bar(models, latencies, color=colors)
plt.title('Çıkarım Hızı (Latency) Karşılaştırması (Daha Düşük Daha İyi)', fontsize=14)
plt.ylabel('Süre (ms)', fontsize=12)

# Değerleri yaz
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.2, f'{yval} ms', ha='center', va='bottom', fontweight='bold')

plt.savefig('results/grafik_3_hiz.png', dpi=300)
print("Grafik 3 (Hız) oluşturuldu.")

# 4. HIZ - DOĞRULUK ÖDÜNLEŞİMİ (SCATTER PLOT - DÜZELTİLMİŞ)
plt.figure(figsize=(12, 8))

# Eksen sınırlarını dinamik ayarlar
y_min, y_max = min(accuracies), max(accuracies)
plt.ylim(y_min - 0.5, y_max + 0.5) 
plt.xlim(min(latencies) - 2, max(latencies) + 4)

# Metin kaydırma ayarları (x, y) - Üst üste binmeyi önlemek için
text_offsets = [
    (0, 0.15),    # Teacher: Yukarı
    (0, -0.30),   # Pruned: Aşağı (Teacher ile karışmasın)
    (0, 0.15),    # Quantized: Yukarı
    (0, 0.15),    # Student: Yukarı
    (0, -0.30)    # Combo: Aşağı (Student ile karışmasın)
]

for i, model in enumerate(models):
    # Balon boyutu: En küçük model bile görünsün diye +100 ekledik
    plt.scatter(latencies[i], accuracies[i], s=(sizes[i]*40)+100, 
                color=colors[i], label=model, alpha=0.7, edgecolors='black')
    
    # Yazıyı kaydırarak yaz
    plt.text(latencies[i] + text_offsets[i][0], 
             accuracies[i] + text_offsets[i][1], 
             model, 
             fontsize=10, 
             ha='center', 
             fontweight='bold',
             bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=1))

plt.title('Hız vs Doğruluk Ödünleşimi (Trade-off)\n(Balon büyüklüğü = Model Boyutu)', fontsize=16)
plt.xlabel('Gecikme Süresi (ms) <-- Daha Hızlı', fontsize=12)
plt.ylabel('Doğruluk (%) <-- Daha Zeki', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)

plt.savefig('results/grafik_4_tradeoff.png', dpi=300)
print("Grafik 4 (Trade-off) oluşturuldu.")

# 5. BOYUT - DOĞRULUK ÖDÜNLEŞİMİ (SCATTER PLOT)
plt.figure(figsize=(12, 8))

# Eksen sınırlarını dinamik ayarlar
y_min, y_max = min(accuracies), max(accuracies)
plt.ylim(y_min - 0.5, y_max + 0.5) 
plt.xlim(min(sizes) - 2, max(sizes) + 5)

# Metin kaydırma ayarları (x, y) - Üst üste binmeyi önlemek için
text_offsets_size = [
    (0, 0.15),    # Teacher: Yukarı
    (0, -0.30),   # Pruned: Aşağı (Teacher ile karışmasın)
    (0, 0.15),    # Quantized: Yukarı
    (0, -0.30),   # Student: Aşağı
    (0, 0.15)     # Combo: Yukarı
]

for i, model in enumerate(models):
    # Balon boyutu: Latency değerine göre (hızı temsil eder)
    plt.scatter(sizes[i], accuracies[i], s=(latencies[i]*80)+100, 
                color=colors[i], label=model, alpha=0.7, edgecolors='black')
    
    # Yazıyı kaydırarak yaz
    plt.text(sizes[i] + text_offsets_size[i][0], 
             accuracies[i] + text_offsets_size[i][1], 
             model, 
             fontsize=10, 
             ha='center', 
             fontweight='bold',
             bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=1))

plt.title('Boyut vs Doğruluk Ödünleşimi (Trade-off)\n(Balon büyüklüğü = Çıkarım Süresi)', fontsize=16)
plt.xlabel('Model Boyutu (MB) <-- Daha Küçük', fontsize=12)
plt.ylabel('Doğruluk (%) <-- Daha Zeki', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)

plt.savefig('results/grafik_5_boyut_dogruluk.png', dpi=300)
print("Grafik 5 (Boyut vs Doğruluk Trade-off) oluşturuldu.")