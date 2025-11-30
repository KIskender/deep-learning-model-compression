# Deep Learning Model Compression Project

Bu proje, derin öğrenme modellerini sıkıştırmak (compression) ve hızlandırmak amacıyla **Knowledge Distillation (Bilgi Damıtma)** ve **Quantization (Nicemleme)** tekniklerini uygulamaktadır. Proje kapsamında hem Görüntü İşleme (Vision) hem de Doğal Dil İşleme (NLP) alanlarında çalışmalar yapılmıştır.

## 🚀 Özellikler

- **Knowledge Distillation**: Büyük ve karmaşık bir "Teacher" modelden (örn. ResNet18), daha küçük bir "Student" modele (örn. LightCNN) bilgi aktarımı.
- **Quantization**: Model ağırlıklarını FP32'den INT8 formatına dönüştürerek boyut küçültme ve çıkarım (inference) hızını artırma.
- **Pruning (Budama)**: Gereksiz ağırlıkların modelden atılması (src_vision/main.py içerisinde baseline olarak gösterilmektedir).
- **Demo Uygulaması**: Sıkıştırma sonuçlarını görselleştirmek ve karşılaştırmak için interaktif Streamlit arayüzü.

## 📂 Proje Yapısı

```
.
├── data/               # Veri setlerinin indirildiği klasör (CIFAR-10 vb.)
├── demo/               # Streamlit demo uygulaması
│   └── app.py          # Demo ana dosyası
├── models/             # Eğitilmiş model dosyaları (.pth)
├── src_nlp/            # NLP modelleri için sıkıştırma kodları
│   ├── nlp_f1_score.py # NLP model değerlendirme
│   └── nlp_quantize.py # NLP model quantization
├── src_vision/         # Görüntü işleme modelleri için kodlar
│   ├── main.py         # Teacher model eğitimi ve pruning
│   ├── distillation.py # Knowledge Distillation işlemi
│   ├── quantize.py     # Quantization işlemleri
│   └── ...
├── utils/              # Yardımcı fonksiyonlar
└── requirements.txt    # Gerekli kütüphaneler
```

## ⚙️ Kurulum (Installation)

Projeyi kendi bilgisayarınızda çalıştırmak için aşağıdaki adımları izleyin.

1. **Gereksinimlerin Yüklenmesi**: Python 3.10+ ve pip yüklü olduğundan emin olun.

```bash
pip install -r requirements.txt
```

2. **Donanım**: Proje CPU üzerinde çalışabilir ancak eğitim (training) aşamaları için NVIDIA GPU (CUDA) önerilir. Test ve Demo aşamaları standart bir laptop işlemcisiyle sorunsuz çalışır.

## � Kullanım (How to Run)

### A. Görüntü İşleme Modülü (Vision)

Sırasıyla eğitim ve sıkıştırma adımlarını gerçekleştirmek için:

1. **Öğretmen Modeli Eğit (ResNet-18)**:
```bash
python src_vision/main.py
```

2. **Sıkıştırılmış Öğretmeni Test Et**:
```bash
python src_vision/quantize.py
```

3. **Öğrenci Modeli Eğit (Knowledge Distillation)**:
```bash
python src_vision/distillation.py
```

4. **Final Hibrit Sıkıştırma (Combo)**:
```bash
python src_vision/quantize_student.py
```

### B. Doğal Dil İşleme Modülü (NLP)

Metin verileri üzerindeki sıkıştırma başarısını görmek için:

```bash
python src_nlp/nlp_quantize.py
```

### C. Canlı Demo (Arayüz)

Tüm modelleri görsel bir arayüzde test etmek ve karşılaştırmak için:

```bash
streamlit run demo/app.py
```

## 🛠️ Kullanılan Teknolojiler

- **Dil**: Python 3.11
- **Framework**: PyTorch, Torchvision
- **NLP**: Hugging Face Transformers
- **Arayüz**: Streamlit
- **Görselleştirme**: Matplotlib
- **Donanım**: NVIDIA GeForce RTX 3060 Laptop GPU

## 📊 Sonuçlar (Örnek)

Demo uygulamasında gözlemlenen tipik sonuçlar:

| Model | Boyut | Hız (Inference) | Doğruluk |
|-------|-------|-----------------|----------|
| **Teacher (ResNet18)** | ~40 MB | ~Yavaş | Yüksek |
| **Student (Distilled + Quantized)** | ~0.6 MB | ~Hızlı | Kabul edilebilir kayıp |

*Student model, Teacher modele göre yaklaşık **60-70 kat** daha küçük boyutludur.*

## 👥 Katkıda Bulunanlar

- İskender KAHRAMAN (Senior Design Project)
