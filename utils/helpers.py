"""Shared helper utilities for AgroAgent."""
from typing import Any


REQUIRED_SOIL_KEYS = {"N", "P", "K", "temperature", "humidity", "ph", "rainfall"}


def validate_soil_input(data: dict) -> tuple[bool, str]:
    """Return (True, '') if soil input is valid, else (False, error message)."""
    missing = REQUIRED_SOIL_KEYS - data.keys()
    if missing:
        return False, f"Missing fields: {', '.join(sorted(missing))}"

    ranges = {
        "N":           (0, 300),
        "P":           (0, 300),
        "K":           (0, 300),
        "temperature": (-10, 55),
        "humidity":    (0, 100),
        "ph":          (0, 14),
        "rainfall":    (0, 5000),
    }
    for field, (lo, hi) in ranges.items():
        val = data[field]
        if not (lo <= float(val) <= hi):
            return False, f"{field} value {val} is outside valid range [{lo}, {hi}]"

    if float(data["ph"]) == 0:
        return False, "Soil pH cannot be 0 — it is physically impossible. Typical agricultural soil pH is between 3.5 and 9.5."

    if all(float(data[k]) == 0 for k in ["N", "P", "K", "humidity", "ph"]):
        return False, "All nutrient, humidity, and pH values are 0. Please enter your actual field measurements."

    return True, ""


def format_recommendation(
    crop: str,
    reasoning: str,
    additional_advice: str,
    confidence: float | None = None,
) -> dict[str, Any]:
    """Package the final recommendation into a consistent response dict."""
    result: dict[str, Any] = {
        "recommended_crop": crop,
        "reasoning": reasoning,
        "additional_advice": additional_advice,
    }
    if confidence is not None:
        result["confidence"] = round(confidence, 4)
    return result
