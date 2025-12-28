import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from datasets import load_dataset
from sklearn.metrics import f1_score, accuracy_score
import time
import os
import warnings

# Gereksiz uyarıları kapatır
warnings.filterwarnings("ignore")

# --- AYARLAR ---
device = torch.device("cpu") # NLP Quantization CPU'da yapılır
MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english" 

print("--- NLP (IMDB) F1 Skor ve Performans Testi ---")

# 1. VERİ SETİ HAZIRLIĞI
print("[1/4] IMDB Veri Setinden Örnekler İndiriliyor...")
try:
    dataset = load_dataset("imdb", split="test[:100]") 
    texts = dataset["text"]
    labels = dataset["label"] 
except Exception as e:
    print(f"İnternet/Dataset hatası: {e}")

    texts = ["This movie is great", "I hate this film"] * 50
    labels = [1, 0] * 50

# 2. MODEL VE TOKENIZER
print("[2/4] DistilBERT Modeli Yükleniyor...")
tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)
model_fp32 = DistilBertForSequenceClassification.from_pretrained(MODEL_NAME).to(device)
model_fp32.eval()

# 3. ÖLÇÜM FONKSİYONU
def evaluate_performance(model, texts, true_labels, name="Model"):
    print(f"\n--- {name} Test Ediliyor ---")
    preds = []
    start_time = time.time()
    
    with torch.no_grad():
        for text in texts:
            # Metni token'lara ayır
            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)
            outputs = model(**inputs)
            prediction = torch.argmax(outputs.logits, dim=1).item()
            preds.append(prediction)
            
    end_time = time.time()
    duration = end_time - start_time
    
    # Metrikler
    acc = accuracy_score(true_labels, preds)
    # average='weighted' eklendi ki 0 çıkmasın
    f1 = f1_score(true_labels, preds, average='weighted') 
    avg_latency = (duration / len(texts)) * 1000 
    
    # Model Boyutu
    temp_filename = "models/temp_nlp.p"
    torch.save(model.state_dict(), temp_filename)
    size_mb = os.path.getsize(temp_filename) / (1024 * 1024)
    
    # Dosya silme işlemini güvenli hale getirildi
    try:
        os.remove(temp_filename)
    except PermissionError:
        pass # Windows dosyayı tutuyorsa geçsim
    
    print(f"   -> Doğruluk (Accuracy): %{acc*100:.2f}")
    print(f"   -> F1 Skoru           : {f1:.4f}")
    print(f"   -> Ortalama Hız       : {avg_latency:.2f} ms")
    print(f"   -> Model Boyutu       : {size_mb:.2f} MB")
    
    return acc, f1, avg_latency, size_mb

# 4. TESTLER

# A. Baseline (FP32) Testi
acc_base, f1_base, lat_base, size_base = evaluate_performance(model_fp32, texts, labels, "BASELINE (FP32)")

# B. Quantization (INT8) Uygulama
print("\n[3/4] Dynamic Quantization Uygulanıyor...")
model_int8 = torch.quantization.quantize_dynamic(
    model_fp32, {torch.nn.Linear}, dtype=torch.qint8
)

# C. Quantized (INT8) Testi
acc_int8, f1_int8, lat_int8, size_int8 = evaluate_performance(model_int8, texts, labels, "QUANTIZED (INT8)")

# 5. FİNAL RAPORU
print("\n" + "="*60)
print(f"{'METRİK (NLP)':<20} | {'BASELINE (FP32)':<15} | {'QUANTIZED (INT8)':<15}")
print("-" * 60)
print(f"{'Boyut (MB)':<20} | {size_base:<15.2f} | {size_int8:<15.2f} (x{size_base/size_int8:.1f} Küçük)")
print(f"{'Hız (ms)':<20} | {lat_base:<15.2f} | {lat_int8:<15.2f} (x{lat_base/lat_int8:.1f} Hızlı)")
print(f"{'F1 Skoru':<20} | {f1_base:<15.4f} | {f1_int8:<15.4f}")
print("="*60)