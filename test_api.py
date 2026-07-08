import requests
import time
import pandas as pd
import numpy as np

print("1. Yük testi (Load Testing) için güvenli veri hazırlığı...")
URL = "http://127.0.0.1:8000/predict"
NUM_REQUESTS = 1000

df_transaction = pd.read_csv('train_transaction.csv', nrows=NUM_REQUESTS)
df_identity = pd.read_csv('train_identity.csv', nrows=NUM_REQUESTS)
df = df_transaction.merge(df_identity, on='TransactionID', how='left')

df = df.drop(['isFraud', 'TransactionID'], axis=1)

# Numpy ve Pandas tiplerini saf Python tiplerine ve JSON null (None) değerine çevirme
df = df.replace({np.nan: None, np.inf: None, -np.inf: None})
payloads = df.astype(object).where(pd.notnull(df), None).to_dict(orient='records')

print(f"2. {NUM_REQUESTS} adet istek FastAPI ucuna gönderiliyor...")
success_count = 0
latencies = []
start_total_time = time.time()

for payload in payloads:
    req_start = time.time()
    try:
        response = requests.post(URL, json=payload, timeout=2)
        req_end = time.time()
        
        if response.status_code == 200:
            success_count += 1
            latencies.append((req_end - req_start) * 1000)
        else:
            # Hata detayının ilk hatalı istekte konsola yazdırılarak kontrolün sağlanması
            print(f"Hata Kodu: {response.status_code}, Detay: {response.text}")
            break
    except requests.exceptions.RequestException as e:
        print(f"Bağlantı Hatası: {e}")
        break

end_total_time = time.time()

print("\n--- Test Raporu (QA Sonuçları) ---")
if latencies:
    avg_latency = sum(latencies) / len(latencies)
    print(f"Başarılı İstek Sayısı: {success_count} / {NUM_REQUESTS}")
    print(f"Ortalama Yanıt Süresi (Latency): {avg_latency:.2f} ms")
    if avg_latency < 100:
        print("\nSONUÇ: BAŞARILI. Sistem 100ms altı yanıt süresi hedefini karşılamaktadır.")
else:
    print("Test başarısız oldu veya tüm istekler reddedildi.")
