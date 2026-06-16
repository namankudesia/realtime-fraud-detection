"""Real-Time Fraud Detection API."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from datetime import datetime
from app.features.feature_engineer import Transaction, FeatureEngineer
from app.models.fraud_detector import FraudDetector

app = FastAPI(title="Real-Time Fraud Detection", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

engineer = FeatureEngineer()
detector = FraudDetector()

class TxnRequest(BaseModel):
    transaction_id: str
    user_id: str
    amount: float
    merchant_category: str
    merchant_country: str = "IN"
    card_present: bool = True
    timestamp: Optional[str] = None

@app.post("/predict")
async def predict(req: TxnRequest):
    txn = Transaction(
        transaction_id=req.transaction_id, user_id=req.user_id, amount=req.amount,
        merchant_category=req.merchant_category, merchant_country=req.merchant_country,
        timestamp=req.timestamp or datetime.now().isoformat(), card_present=req.card_present
    )
    features = engineer.engineer(txn)
    prediction = detector.predict(features)
    return {"prediction": prediction.__dict__, "features": {"amount_zscore": features.amount_zscore,
        "velocity_1h": features.velocity_1h, "risk_features": features.risk_features}}

@app.get("/health")
async def health():
    return {"status": "ok", "service": "Real-Time Fraud Detection v1.0"}

@app.get("/")
async def root():
    return {"service": "Real-Time Fraud Detection", "docs": "/docs"}
