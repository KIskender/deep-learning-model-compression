import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import os
import time

# --- AYARLAR ---
device = torch.device("cpu") 

print("--- NLP Model Sıkıştırma (DistilBERT) Başlıyor ---")

# 1. MODELİ VE TOKENIZER'I İNDİR (Hugging Face'den Otomatik)
print("[1/3] Pre-trained DistilBERT indiriliyor (IMDB Sentiment için eğitilmiş)...")
# 'distilbert-base-uncased-finetuned-sst-2-english' modeli film yorumlarını (Pozitif/Negatif) anlar.
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = DistilBertTokenizer.from_pretrained(model_name)
model_fp32 = DistilBertForSequenceClassification.from_pretrained(model_name).to(device)
model_fp32.eval()

# 2. ÖLÇÜM FONKSİYONLARI
def print_size_of_model(model, label=""):
    filename = "models/temp_quant.p"
    torch.save(model.state_dict(), filename)
    size = os.path.getsize(filename) / (1024 * 1024)
    
    try:
        os.remove(filename)
    except OSError:
        pass
        
    print(f"{label} Model Boyutu: {size:.2f} MB")
    return size

def measure_latency(model, text, label=""):
    inputs = tokenizer(text, return_tensors="pt").to(device)
    start = time.time()
    with torch.no_grad():
        model(**inputs)
    end = time.time()
    elapsed = (end - start) * 1000
    print(f"{label} Latency (Tepki Süresi): {elapsed:.2f} ms")
    return elapsed

# 3. BASELINE TESTİ
print("\n[2/3] Baseline (FP32) Test Ediliyor...")
text = "This movie was absolutely fantastic! The acting was great."
base_size = print_size_of_model(model_fp32, "Baseline (FP32)")
base_time = measure_latency(model_fp32, text, "Baseline (FP32)")

# 4. QUANTIZATION (DYNAMIC)
# NLP modelleri için PyTorch'un "quantize_dynamic" fonksiyonu kullanılır.
print("\n[3/3] Dynamic Quantization Uygulanıyor...")
model_int8 = torch.quantization.quantize_dynamic(
    model_fp32, 
    {torch.nn.Linear},  # Sadece Linear katmanları sıkıştır (Transformer'ın çoğu budur)
    dtype=torch.qint8
)

int8_size = print_size_of_model(model_int8, "Quantized (INT8)")
int8_time = measure_latency(model_int8, text, "Quantized (INT8)")

# SONUÇ
print("\n" + "="*50)
print(f"NLP MODEL SIKIŞTIRMA SONUCU")
print("="*50)
print(f"Boyut Küçülmesi : {base_size:.2f} MB -> {int8_size:.2f} MB (x{base_size/int8_size:.1f} Kat)")
print(f"Hızlanma        : {base_time:.2f} ms -> {int8_time:.2f} ms")
print("="*50)