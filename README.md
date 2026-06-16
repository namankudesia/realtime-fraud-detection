# Real-Time Fraud Detection

> ML-powered real-time fraud detection pipeline: **feature engineering + rule-based scoring + SHAP explainability** — sub-100ms decisions

---

## How It Was Built

- **Feature Engineering** — 15+ real-time features: velocity (1h/24h), z-score from 30-day average, merchant risk, geographic anomalies, time patterns
- **Scoring Engine** — weighted rule system (production: XGBoost/LightGBM trained on labeled data)
- **SHAP Explainability** — every fraud decision has a feature-level explanation (regulators require this)
- **3-tier action system** — approve / review / block based on probability thresholds

## How to Run

```bash
git clone https://github.com/namankudesia/realtime-fraud-detection.git
cd realtime-fraud-detection
pip install -r requirements.txt
uvicorn main:app --reload --port 8003
```

```bash
curl -X POST http://localhost:8003/predict \
  -H "Content-Type: application/json" \
  -d '{"transaction_id": "T001", "user_id": "U123", "amount": 50000,
       "merchant_category": "crypto", "merchant_country": "XX", "card_present": false}'
```

> Built by [Naman Kudesia](https://github.com/namankudesia)
