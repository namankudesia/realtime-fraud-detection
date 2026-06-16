"""
Fraud detection model: rule-based + ML scoring + SHAP explainability.
In production: trained XGBoost/LightGBM on labeled transaction data.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List
from app.features.feature_engineer import EnrichedFeatures

@dataclass
class FraudPrediction:
    transaction_id: str
    fraud_probability: float
    is_fraud: bool
    risk_level: str         # low | medium | high | critical
    risk_factors: List[str]
    action: str             # approve | review | block
    shap_explanation: Dict[str, float]

RULE_WEIGHTS = {
    "high_risk_merchant": 0.25,
    "foreign_transaction": 0.15,
    "high_amount_zscore": 0.20,
    "high_velocity": 0.20,
    "card_not_present": 0.10,
    "night_transaction": 0.05,
    "high_deviation": 0.05,
}

class FraudDetector:
    THRESHOLDS = {"low": 0.3, "medium": 0.5, "high": 0.7, "critical": 0.85}

    def predict(self, features: EnrichedFeatures) -> FraudPrediction:
        # Rule-based score (production: replace with trained ML model)
        score = sum(features.risk_features.get(k, 0.0) * w for k, w in RULE_WEIGHTS.items())
        score = min(1.0, score)

        risk_factors = [k.replace("_", " ").title() for k, v in features.risk_features.items() if v > 0.5]
        risk_level = self._risk_level(score)
        action = "block" if score >= 0.85 else "review" if score >= 0.5 else "approve"

        shap = {k: round(features.risk_features.get(k, 0) * w, 4) for k, w in RULE_WEIGHTS.items()}

        return FraudPrediction(
            transaction_id=features.transaction_id,
            fraud_probability=round(score, 4),
            is_fraud=score >= 0.5,
            risk_level=risk_level,
            risk_factors=risk_factors,
            action=action,
            shap_explanation=shap
        )

    def _risk_level(self, score: float) -> str:
        if score >= 0.85: return "critical"
        if score >= 0.70: return "high"
        if score >= 0.50: return "medium"
        return "low"
