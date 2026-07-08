import pandas as pd
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.metrics import classification_report
import joblib
import gc

print("1. Veri setleri diske okunuyor...")
df_transaction = pd.read_csv('train_transaction.csv')
df_identity = pd.read_csv('train_identity.csv')

print("2. Tablolar TransactionID anahtarı üzerinden birleştiriliyor...")
df = df_transaction.merge(df_identity, on='TransactionID', how='left')

del df_transaction, df_identity
gc.collect()

print("3. Veri Ön İşleme (Preprocessing) adımları uygulanıyor...")
y = df['isFraud']
X = df.drop(['isFraud', 'TransactionID'], axis=1)

# Kategorik sütunların tespiti ve dönüştürülmesi
cat_cols = X.select_dtypes(include=['object', 'string']).columns.tolist()
for col in cat_cols:
    X[col] = X[col].astype('category')

feature_names = X.columns.tolist()

print("4. Veri seti %80 Eğitim ve %20 Test olarak ayrıştırılıyor...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("5. LightGBM sınıflandırıcısı eğitiliyor...")
model = lgb.LGBMClassifier(
    random_state=42, 
    n_estimators=150,
    is_unbalance=True, 
    n_jobs=-1
)
model.fit(X_train, y_train)

print("6. Model performans değerlendirmesi:")
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

print("7. Model, kategorik liste ve sütun şeması dışa aktarılıyor...")
joblib.dump(model, 'ieee_fraud_model.pkl')
joblib.dump(cat_cols, 'categorical_columns.pkl')
joblib.dump(feature_names, 'feature_names.pkl')

print("Eğitim ve şema kaydı başarıyla tamamlanmıştır.")
