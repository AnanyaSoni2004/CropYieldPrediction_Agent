"""
Crop Prediction Agent
---------------------
Loads the trained Random Forest model and predicts the top-N most suitable
crops given soil and environmental parameters.
"""
import os
import pickle
from typing import Any

import numpy as np

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]


class CropPredictionAgent:
    """Wraps the trained RF classifier for inference."""

    def __init__(self):
        self.model = self._load(os.path.join(MODEL_DIR, "crop_model.pkl"))
        self.scaler = self._load(os.path.join(MODEL_DIR, "scaler.pkl"))
        self.encoder = self._load(os.path.join(MODEL_DIR, "label_encoder.pkl"))

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def predict(self, soil_data: dict[str, float], top_n: int = 3) -> dict[str, Any]:
        """
        Return the top_n crop predictions with confidence scores.

        Parameters
        ----------
        soil_data : dict with keys N, P, K, temperature, humidity, ph, rainfall
        top_n     : number of top predictions to return

        Returns
        -------
        {
            "top_prediction": str,
            "confidence":     float,
            "top_crops": [{"crop": str, "confidence": float}, ...],
            "soil_summary": dict,
        }
        """
        X = self._build_feature_vector(soil_data)
        X_scaled = self.scaler.transform(X)

        proba = self.model.predict_proba(X_scaled)[0]
        top_indices = np.argsort(proba)[::-1][:top_n]

        top_crops = [
            {
                "crop": self.encoder.classes_[i],
                "confidence": round(float(proba[i]), 4),
            }
            for i in top_indices
        ]

        return {
            "top_prediction": top_crops[0]["crop"],
            "confidence": top_crops[0]["confidence"],
            "top_crops": top_crops,
            "soil_summary": self._summarise_soil(soil_data),
        }

    # ------------------------------------------------------------------
    # private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load(path: str) -> Any:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Model artifact not found: {path}\n"
                "Run `python models/train_model.py` first."
            )
        with open(path, "rb") as f:
            return pickle.load(f)

    def _build_feature_vector(self, soil_data: dict) -> np.ndarray:
        return np.array([[soil_data[f] for f in FEATURES]], dtype=float)

    @staticmethod
    def _summarise_soil(soil_data: dict) -> dict:
        """Produce a human-readable soil quality summary."""
        n, p, k = soil_data["N"], soil_data["P"], soil_data["K"]
        ph = soil_data["ph"]

        nitrogen_status = (
            "high" if n > 80 else "medium" if n > 40 else "low"
        )
        phosphorus_status = (
            "high" if p > 60 else "medium" if p > 30 else "low"
        )
        potassium_status = (
            "high" if k > 60 else "medium" if k > 30 else "low"
        )

        if ph < 5.5:
            ph_desc = "strongly acidic"
        elif ph < 6.5:
            ph_desc = "slightly acidic"
        elif ph < 7.5:
            ph_desc = "neutral"
        elif ph < 8.5:
            ph_desc = "slightly alkaline"
        else:
            ph_desc = "strongly alkaline"

        return {
            "nitrogen":   nitrogen_status,
            "phosphorus": phosphorus_status,
            "potassium":  potassium_status,
            "ph_status":  ph_desc,
            "rainfall_mm": soil_data["rainfall"],
        }
