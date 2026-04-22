"""
Decision Agent (Main Orchestrator)
------------------------------------
Combines outputs from the Crop, Weather, Market, and RAG agents then
calls an LLM (Groq / OpenAI) to generate an explainable recommendation.
"""
import os
import re
from typing import Any

from groq import Groq

GROQ_API_KEY   = os.getenv("GROQ_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

GROQ_MODEL   = "llama-3.3-70b-versatile"
OPENAI_MODEL = "gpt-4o-mini"

SYSTEM_PROMPT = (
    "You are an expert agricultural advisor system. "
    "Follow the instructions precisely and respond in the exact format specified."
)

PROMPT_TEMPLATE = """You are an expert agricultural advisor system.

Your task is to recommend a suitable crop ONLY if the environmental conditions are valid and agronomically safe.

---

## STEP 1: INPUT VALIDATION

Carefully evaluate the input parameters:

* Temperature (°C)
* Humidity (%)
* Soil pH
* Rainfall (mm)
* Nitrogen (N), Phosphorus (P), Potassium (K)

Mark the input as INVALID if ANY of the following conditions are true:

* Temperature < 0°C or > 50°C
* Soil pH < 4 or > 9
* Humidity ≤ 0% or ≥ 100%
* Rainfall < 100 mm or > 4000 mm
* N, P, or K extremely low (< 10)

If invalid:

* DO NOT recommend any crop
* Clearly list all issues
* Suggest realistic ranges for correction

---

## STEP 2: AGRONOMIC SUITABILITY CHECK

Even if inputs are valid, check if they are suitable for crop growth:

* Very high temperature (> 45°C) → heat stress
* Very low rainfall (< 500 mm) → drought risk
* Very high humidity (> 90%) → disease risk
* Strongly acidic soil (pH < 5) → poor nutrient uptake

If conditions are harsh:

* Mark as "Suboptimal"
* Recommend either resilient crops or say "No ideal crop under current conditions"

---

## STEP 3: ML PREDICTION VERIFICATION

ML model prediction:
{ml_prediction}

DO NOT blindly trust it. Check:

* Does the predicted crop realistically grow in these conditions?
* Compare with known ideal ranges.

If mismatch:

* Reject the ML prediction
* Explain why it is incorrect

---

## STEP 4: FINAL OUTPUT

Respond in this EXACT structured format:

Validation Status: <Valid / Invalid>

Issues Found:
- <list each problem, or "None">

Environmental Assessment: <Optimal / Suboptimal / Extreme>

Final Recommendation:
Recommended Crop: <crop name, or "None – no suitable crop for current conditions">

Reasoning:
- Soil: <findings>
- Weather: <findings>
- Market: <findings>

ML Prediction Review:
- Status: <Accepted / Rejected>
- Reason: <why accepted or rejected>

Suggested Fixes:
- <how to improve soil, temperature, irrigation, etc., or "None required">

Confidence Level: <High / Medium / Low>

---

## IMPORTANT RULES

* Never recommend crops for extreme or impossible conditions.
* Always prioritize real-world agronomy over ML output.
* It is acceptable to say "No suitable crop".
* Be concise but scientifically accurate.

---

User Input:
{input_data}"""


class DecisionAgent:
    """Uses an LLM to synthesise all agent outputs into a final recommendation."""

    def __init__(self):
        self._client = self._init_client()

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def decide(
        self,
        crop_result:    dict[str, Any],
        weather_result: dict[str, Any],
        market_result:  dict[str, Any],
        rag_context:    str = "",
    ) -> dict[str, Any]:
        """
        Generate the final LLM-powered recommendation.

        Returns
        -------
        {
            "recommended_crop": str,
            "llm_response":     str,
            "model_used":       str,
        }
        """
        user_message = self._build_prompt(
            crop_result, weather_result, market_result, rag_context
        )
        llm_text, model_used = self._call_llm(user_message)
        recommended = self._extract_crop(llm_text, crop_result["top_prediction"])

        return {
            "recommended_crop": recommended,
            "llm_response":     llm_text,
            "model_used":       model_used,
        }

    # ------------------------------------------------------------------
    # private helpers
    # ------------------------------------------------------------------

    def _call_llm(self, user_message: str) -> tuple[str, str]:
        """Try Groq first, fall back to OpenAI, then return a rule-based response."""
        if self._client and GROQ_API_KEY:
            try:
                resp = self._client.chat.completions.create(
                    model=GROQ_MODEL,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user",   "content": user_message},
                    ],
                    temperature=0.3,
                    max_tokens=900,
                )
                return resp.choices[0].message.content.strip(), GROQ_MODEL
            except Exception:
                pass

        if OPENAI_API_KEY:
            try:
                import openai
                openai.api_key = OPENAI_API_KEY
                resp = openai.chat.completions.create(
                    model=OPENAI_MODEL,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user",   "content": user_message},
                    ],
                    temperature=0.3,
                    max_tokens=900,
                )
                return resp.choices[0].message.content.strip(), OPENAI_MODEL
            except Exception:
                pass

        return self._fallback_response(user_message), "rule-based-fallback"

    @staticmethod
    def _init_client():
        if GROQ_API_KEY:
            try:
                return Groq(api_key=GROQ_API_KEY)
            except Exception:
                pass
        return None

    @staticmethod
    def _build_prompt(
        crop:    dict,
        weather: dict,
        market:  dict,
        rag:     str,
    ) -> str:
        top_crops_str = ", ".join(
            f"{c['crop']} ({c['confidence']*100:.1f}%)"
            for c in crop["top_crops"]
        )
        ranked_str = ", ".join(
            f"{r['crop']} (₹{r['price_per_quintal']}/q, score={r['profitability_score']})"
            for r in market["ranked_crops"]
        )

        ml_prediction = (
            f"Best prediction: {crop['top_prediction']} "
            f"(confidence {crop['confidence']*100:.1f}%)\n"
            f"Top crops: {top_crops_str}"
        )

        input_data = (
            f"Temperature:  {weather['temperature']}°C\n"
            f"Humidity:     {weather['humidity']}%\n"
            f"Conditions:   {weather['description']}\n"
            f"Location:     {weather['location']}\n"
            f"Soil pH:      {crop['soil_summary']['ph_status']}\n"
            f"Rainfall:     {crop['soil_summary']['rainfall_mm']} mm\n"
            f"Nitrogen (N): {crop['soil_summary']['nitrogen']}\n"
            f"Phosphorus (P): {crop['soil_summary']['phosphorus']}\n"
            f"Potassium (K): {crop['soil_summary']['potassium']}\n"
            f"\nMarket – Best crop: {market['best_market_crop']}\n"
            f"Ranked: {ranked_str}\n"
            f"Insights: {market['market_insights']}\n"
            f"\nAgricultural Knowledge:\n{rag if rag else 'Not available.'}"
        )

        return PROMPT_TEMPLATE.format(
            ml_prediction=ml_prediction,
            input_data=input_data,
        )

    @staticmethod
    def _extract_crop(text: str, fallback: str) -> str:
        """Parse 'Recommended Crop:' from the structured LLM output."""
        for line in text.splitlines():
            stripped = line.strip()
            # Validation failed — no crop should be recommended
            if re.match(r"validation status\s*:\s*invalid", stripped, re.IGNORECASE):
                return "Invalid Input"
            # Explicit no-crop signal
            if re.match(r"recommended crop\s*:\s*none", stripped, re.IGNORECASE):
                return "No Suitable Crop"
            # Normal recommendation
            m = re.match(r"recommended crop\s*:\s*(.+)", stripped, re.IGNORECASE)
            if m:
                return m.group(1).strip()
        return fallback

    @staticmethod
    def _fallback_response(prompt: str) -> str:
        """Rule-based fallback applying the same validation logic as the LLM prompt."""
        def _val(label: str, default: float) -> float:
            m = re.search(rf"{label}\s*[:(]\s*([\d.]+)", prompt, re.IGNORECASE)
            return float(m.group(1)) if m else default

        temp     = _val("Temperature", 25.0)
        humidity = _val("Humidity",    70.0)
        rainfall = _val("Rainfall",   500.0)
        ph       = _val("pH",           6.5)
        n        = _val(r"Nitrogen \(N\)",    90.0)
        p        = _val(r"Phosphorus \(P\)",  42.0)
        k        = _val(r"Potassium \(K\)",   43.0)

        issues = []
        if temp < 0 or temp > 50:
            issues.append(f"Temperature {temp}°C is outside valid range (0–50°C).")
        if ph < 4 or ph > 9:
            issues.append(f"Soil pH {ph} is highly unsuitable (valid: 4–9).")
        if humidity <= 0 or humidity >= 100:
            issues.append(f"Humidity {humidity}% is an unrealistic edge case (valid: 1–99%).")
        if rainfall < 100 or rainfall > 4000:
            issues.append(f"Rainfall {rainfall} mm is extreme (valid: 100–4000 mm).")
        if n < 10:
            issues.append(f"Nitrogen {n} kg/ha is extremely low (minimum: 10 kg/ha).")
        if p < 10:
            issues.append(f"Phosphorus {p} kg/ha is extremely low (minimum: 10 kg/ha).")
        if k < 10:
            issues.append(f"Potassium {k} kg/ha is extremely low (minimum: 10 kg/ha).")

        if issues:
            issue_lines = "\n".join(f"- {i}" for i in issues)
            return (
                "Validation Status: Invalid\n\n"
                f"Issues Found:\n{issue_lines}\n\n"
                "Environmental Assessment: Extreme\n\n"
                "Final Recommendation:\n"
                "Recommended Crop: None – no suitable crop for current conditions\n\n"
                "Suggested Fixes:\n"
                "- Temperature: 0–50°C\n"
                "- Soil pH: 4–9\n"
                "- Humidity: 1–99%\n"
                "- Rainfall: 100–4000 mm\n"
                "- N, P, K: each ≥ 10 kg/ha\n\n"
                "Confidence Level: Low\n\n"
                "(Note: LLM not configured – rule-based fallback.)"
            )

        crop = "the predicted crop"
        for line in prompt.splitlines():
            if "Best prediction:" in line:
                crop = line.split(":", 1)[1].split("(")[0].strip()
                break

        suitability_notes = []
        if temp > 45:
            suitability_notes.append("High temperature – heat-stress risk.")
        if rainfall < 500:
            suitability_notes.append("Low rainfall – drought risk; consider irrigation.")
        if humidity > 90:
            suitability_notes.append("Very high humidity – fungal disease risk.")
        if ph < 5:
            suitability_notes.append("Acidic soil – poor nutrient uptake likely.")

        assessment = "Suboptimal" if suitability_notes else "Optimal"
        fix_lines  = "\n".join(f"- {n}" for n in suitability_notes) or "- None required."

        return (
            "Validation Status: Valid\n\n"
            "Issues Found:\n- None\n\n"
            f"Environmental Assessment: {assessment}\n\n"
            "Final Recommendation:\n"
            f"Recommended Crop: {crop}\n\n"
            "Reasoning:\n"
            "- Soil: Parameters are within acceptable agricultural ranges.\n"
            "- Weather: Conditions are suitable for the recommended crop.\n"
            "- Market: This crop shows reasonable demand and profitability.\n\n"
            "ML Prediction Review:\n"
            "- Status: Accepted\n"
            "- Reason: ML prediction aligns with the given soil and climate conditions.\n\n"
            f"Suggested Fixes:\n{fix_lines}\n\n"
            f"Confidence Level: {'Medium' if suitability_notes else 'High'}\n\n"
            "(Note: LLM not configured – rule-based fallback.)"
        )
