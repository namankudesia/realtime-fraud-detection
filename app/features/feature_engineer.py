"""Real-time feature engineering for fraud detection."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional
import statistics
from datetime import datetime

@dataclass
class Transaction:
    transaction_id: str
    user_id: str
    amount: float
    merchant_category: str
    merchant_country: str
    timestamp: str
    card_present: bool
    ip_address: Optional[str] = None
    device_fingerprint: Optional[str] = None

@dataclass
class EnrichedFeatures:
    transaction_id: str
    amount: float
    amount_zscore: float
    hour_of_day: int
    is_weekend: bool
    is_foreign_transaction: bool
    merchant_risk_score: float
    velocity_1h: int        # num transactions last 1 hour
    velocity_24h: int       # num transactions last 24 hours
    avg_amount_30d: float
    amount_deviation: float # % deviation from 30d avg
    card_not_present: bool
    risk_features: Dict[str, float]

HIGH_RISK_MERCHANTS = {"gambling", "crypto", "adult", "money_transfer"}
HIGH_RISK_COUNTRIES = {"XX", "ZZ"}  # placeholder for known high-risk codes

class FeatureEngineer:
    def __init__(self, user_history_store: dict = None):
        self.history = user_history_store or {}

    def engineer(self, txn: Transaction) -> EnrichedFeatures:
        dt = datetime.fromisoformat(txn.timestamp)
        history = self.history.get(txn.user_id, [])
        amounts = [t["amount"] for t in history] or [txn.amount]
        avg_30d = statistics.mean(amounts)
        std_30d = statistics.stdev(amounts) if len(amounts) > 1 else 1.0
        z_score = (txn.amount - avg_30d) / std_30d if std_30d else 0.0
        now = dt
        v1h = sum(1 for t in history if (now - datetime.fromisoformat(t["timestamp"])).seconds < 3600)
        v24h = sum(1 for t in history if (now - datetime.fromisoformat(t["timestamp"])).days < 1)
        deviation_pct = abs(txn.amount - avg_30d) / (avg_30d + 1e-9) * 100

        risk_features = {
            "high_risk_merchant": float(txn.merchant_category.lower() in HIGH_RISK_MERCHANTS),
            "foreign_transaction": float(txn.merchant_country not in ("IN", "US", "GB")),
            "high_amount_zscore": float(abs(z_score) > 3),
            "high_velocity": float(v1h > 5),
            "card_not_present": float(not txn.card_present),
            "night_transaction": float(dt.hour < 6 or dt.hour > 22),
            "high_deviation": float(deviation_pct > 200),
        }

        return EnrichedFeatures(
            transaction_id=txn.transaction_id,
            amount=txn.amount,
            amount_zscore=round(z_score, 3),
            hour_of_day=dt.hour,
            is_weekend=dt.weekday() >= 5,
            is_foreign_transaction=txn.merchant_country not in ("IN", "US", "GB"),
            merchant_risk_score=float(txn.merchant_category.lower() in HIGH_RISK_MERCHANTS),
            velocity_1h=v1h,
            velocity_24h=v24h,
            avg_amount_30d=round(avg_30d, 2),
            amount_deviation=round(deviation_pct, 2),
            card_not_present=not txn.card_present,
            risk_features=risk_features
        )
