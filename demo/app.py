import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from PIL import Image
import time
import os
import copy
import io

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="DeepCompress Demo", layout="wide", page_icon="🚀")

st.title("🚀 Model Sıkıştırma ve Hızlandırma Projesi")
st.markdown("""
Bu uygulama, **Görüntü İşleme** ve **Doğal Dil İşleme (NLP)** modellerinin sıkıştırma öncesi ve sonrası 
performanslarını (Hız, Boyut, Doğruluk) karşılaştırmaktadır.
""")

# --- CİHAZ AYARI ---
device = torch.device("cpu")

# ==========================================
# 1. GÖRÜNTÜ İŞLEME (VISION) MODÜLLERİ
# ==========================================

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
        x = x.reshape(-1, 128 * 4 * 4)
        x = self.classifier(x)
        return x

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

@st.cache_resource
def load_vision_models():
    # Model yolları (Klasör yapısına uygun olarak)
    TEACHER_PATH = "models/resnet18_cifar10_10epoch.pth"
    STUDENT_PATH = "models/student_distilled.pth"

    # 1. TEACHER YÜKLE
    teacher = models.resnet18(weights=None)
    teacher.fc = nn.Linear(teacher.fc.in_features, 10)
    if os.path.exists(TEACHER_PATH):
        teacher.load_state_dict(torch.load(TEACHER_PATH, map_location=device))
    teacher.eval()

    # 2. STUDENT YÜKLE & QUANTIZE ET
    student_fp32 = LightCNN()
    if os.path.exists(STUDENT_PATH):
        student_fp32.load_state_dict(torch.load(STUDENT_PATH, map_location=device))
    student_fp32.eval()

    # Canlı Quantization
    backend = "fbgemm"
    torch.backends.quantized.engine = backend
    student_q = QuantizedLightCNN(copy.deepcopy(student_fp32))
    student_q.qconfig = torch.quantization.get_default_qconfig(backend)
    torch.quantization.prepare(student_q, inplace=True)
    student_q(torch.randn(1, 3, 32, 32)) # Dummy calibration
    torch.quantization.convert(student_q, inplace=True)
    student_q.eval()

    return teacher, student_q

def predict_image(model, img_tensor):
    start = time.time()
    with torch.no_grad():
        output = model(img_tensor)
        prob = torch.nn.functional.softmax(output[0], dim=0)
        conf, pred = torch.max(prob, 0)
    end = time.time()
    return pred.item(), conf.item(), (end - start) * 1000

# ==========================================
# 2. DOĞAL DİL İŞLEME (NLP) MODÜLLERİ
# ==========================================

@st.cache_resource
def load_nlp_models():
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    
    # Tokenizer ve FP32 Model
    tokenizer = DistilBertTokenizer.from_pretrained(model_name)
    model_fp32 = DistilBertForSequenceClassification.from_pretrained(model_name).to(device)
    model_fp32.eval()
    
    # INT8 Model (Dynamic Quantization)
    model_int8 = torch.quantization.quantize_dynamic(
        model_fp32, {torch.nn.Linear}, dtype=torch.qint8
    )
    
    return tokenizer, model_fp32, model_int8

def predict_text(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)
    start = time.time()
    with torch.no_grad():
        outputs = model(**inputs)
        prob = torch.nn.functional.softmax(outputs.logits[0], dim=0)
        conf, pred = torch.max(prob, 0)
    end = time.time()
    return pred.item(), conf.item(), (end - start) * 1000

# ==========================================
# ARAYÜZ (UI)
# ==========================================

# Sekmeler oluştur
tab1, tab2 = st.tabs(["📷 Görüntü İşleme (Vision)", "📝 Doğal Dil İşleme (NLP)"])

# --- TAB 1: GÖRÜNTÜ ---
with tab1:
    st.header("Nesne Sınıflandırma (CIFAR-10)")
    teacher_model, student_model = load_vision_models()
    CLASSES = ['Uçak', 'Otomobil', 'Kuş', 'Kedi', 'Geyik', 'Köpek', 'Kurbağa', 'At', 'Gemi', 'Kamyon']
    
    uploaded_file = st.file_uploader("Bir resim yükleyin...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        image = Image.open(uploaded_file).convert('RGB')
        with col1:
            st.image(image, caption='Yüklenen Resim', use_container_width=True)
            
        # Hazırlık
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        img_tensor = transform(image).unsqueeze(0)

        if st.button('Görüntüyü Analiz Et', type="primary"):
            t_pred, t_conf, t_time = predict_image(teacher_model, img_tensor)
            s_pred, s_conf, s_time = predict_image(student_model, img_tensor)
            
            c1, c2 = st.columns(2)
            with c1:
                st.info("👨‍🏫 TEACHER (ResNet-18)")
                st.metric("Tahmin", f"{CLASSES[t_pred]}", f"%{t_conf*100:.1f} Güven")
                st.write(f"⏱️ Süre: **{t_time:.2f} ms**")
                st.write("💾 Boyut: **40.0 MB**")
            with c2:
                st.success("👶 STUDENT (Combo)")
                st.metric("Tahmin", f"{CLASSES[s_pred]}", f"%{s_conf*100:.1f} Güven")
                st.write(f"⏱️ Süre: **{s_time:.2f} ms**")
                st.write("💾 Boyut: **0.62 MB**")
                st.caption(f"🚀 {t_time/s_time:.1f}x Hızlı | 64x Küçük")

# --- TAB 2: NLP ---
with tab2:
    st.header("Duygu Analizi (IMDB Sentiment)")
    st.write("İngilizce bir film yorumu yazın, model duygu durumunu (Pozitif/Negatif) tahmin etsin.")
    
    tokenizer, nlp_fp32, nlp_int8 = load_nlp_models()
    LABELS = ["NEGATİF 😞", "POZİTİF 😊"]
    
    text_input = st.text_area("Yorumunuzu girin:", "This movie was absolutely fantastic! The acting was great.")
    
    if st.button('Metni Analiz Et'):
        if text_input:
            # Baseline Tahmin
            b_pred, b_conf, b_time = predict_text(nlp_fp32, tokenizer, text_input)
            # Quantized Tahmin
            q_pred, q_conf, q_time = predict_text(nlp_int8, tokenizer, text_input)
            
            c1, c2 = st.columns(2)
            with c1:
                st.info("📦 BASELINE (DistilBERT FP32)")
                st.metric("Sonuç", LABELS[b_pred], f"%{b_conf*100:.1f}")
                st.write(f"⏱️ Süre: **{b_time:.2f} ms**")
                st.write("💾 Boyut: **255 MB**")
            with c2:
                st.success("⚡ QUANTIZED (DistilBERT INT8)")
                st.metric("Sonuç", LABELS[q_pred], f"%{q_conf*100:.1f}")
                st.write(f"⏱️ Süre: **{q_time:.2f} ms**")
                st.write("💾 Boyut: **132 MB**")
                st.caption(f"🚀 {b_time/q_time:.1f}x Hızlı | 2x Küçük")