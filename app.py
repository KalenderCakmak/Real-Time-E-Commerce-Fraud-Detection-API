from fastapi import FastAPI, HTTPException
from typing import Dict, Any
import joblib
import pandas as pd
import numpy as np

# Model ve kayıtlı şema dosyalarının yüklenmesi
model = joblib.load('ieee_fraud_model.pkl')
cat_cols = joblib.load('categorical_columns.pkl')
feature_names = joblib.load('feature_names.pkl')

app = FastAPI(title="Gerçek Zamanlı E-Ticaret Sahtekarlık Tespit API")

@app.post("/predict")
async def predict_fraud(transaction: Dict[str, Any]):
    try:
        input_data = pd.DataFrame([transaction])
        
        for col in feature_names:
            if col not in input_data.columns:
                input_data[col] = None
                
        input_data = input_data[feature_names]
        
        for col in feature_names:
            if col in cat_cols:
                input_data[col] = input_data[col].astype('category')
            else:
                input_data[col] = pd.to_numeric(input_data[col], errors='coerce')
        
        # Olasılık tahmini
        fraud_probability = model.predict_proba(input_data)[0][1]
        
        if fraud_probability > 0.50:
            status = "blocked"
        else:
            status = "approved"
            
        return {
            "transaction_id": "req-ieee-realtime",
            "status": status,
            "fraud_probability": round(float(fraud_probability), 4)
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
