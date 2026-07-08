import requests
import pandas as pd
import numpy as np
import sys

print("1. Toplu Doğruluk ve Metrik Testi (Batch Evaluation) başlatılıyor...")
URL = "http://127.0.0.1:8000/predict"
BATCH_SIZE = 500

print(f"2. Test için eğitim setinden bağımsız {BATCH_SIZE} adet ardışık işlem okunuyor...")

skip_indices = range(1, 50001) 

try:
    df_transaction = pd.read_csv('train_transaction.csv', skiprows=skip_indices, nrows=BATCH_SIZE)
    df_identity = pd.read_csv('train_identity.csv', skiprows=skip_indices, nrows=BATCH_SIZE)
except FileNotFoundError:
    print("HATA: CSV dosyaları bulunamadı. Terminalin proje kök dizininde olduğundan emin ol.")
    sys.exit(1)

df = df_transaction.merge(df_identity, on='TransactionID', how='left')

actual_labels = df['isFraud'].values
df_payloads = df.drop(['isFraud', 'TransactionID'], axis=1)

df_payloads = df_payloads.replace({np.nan: None, np.inf: None, -np.inf: None})
payloads = df_payloads.astype(object).where(pd.notnull(df_payloads), None).to_dict(orient='records')

tp, fp, tn, fn = 0, 0, 0, 0
failed_requests = 0

print("3. İşlemler API'ye iletiliyor ve sonuçlar matrise işleniyor...")
for payload, actual in zip(payloads, actual_labels):
    try:
        response = requests.post(URL, json=payload, timeout=2)
        if response.status_code == 200:
            res = response.json()
            status = res['status']  # 'blocked' veya 'approved'
            
            predicted_fraud = 1 if status == 'blocked' else 0
            
            if predicted_fraud == 1 and actual == 1:
                tp += 1
            elif predicted_fraud == 1 and actual == 0:
                fp += 1
            elif predicted_fraud == 0 and actual == 0:
                tn += 1
            elif predicted_fraud == 0 and actual == 1:
                fn += 1
        else:
            failed_requests += 1
    except requests.exceptions.RequestException:
        failed_requests += 1

print("\n--- Toplu Doğruluk ve Karmaşıklık Matrisi (Confusion Matrix) ---")
print(f"Toplam Test Edilen İşlem: {BATCH_SIZE}")
print(f"Başarısız/Erişilemeyen İstek: {failed_requests}")
print(f"True Positive (Doğru Engellenen Fraud): {tp}")
print(f"False Positive (Yanlış Engellenen Masum - FP): {fp}")
print(f"True Negative (Doğru Onaylanan Masum): {tn}")
print(f"False Negative (Kaçırılan Sinsi Fraud - FN): {fn}")

total_evaluated = tp + fp + tn + fn
if total_evaluated > 0:
    accuracy = (tp + tn) / total_evaluated * 100
    precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
    print(f"\nGenel Doğruluk (Accuracy): {accuracy:.2f}%")
    print(f"Kesinlik (Precision): {precision:.2f}%")
    print(f"Duyarlılık / Yakalama (Recall): {recall:.2f}%")
