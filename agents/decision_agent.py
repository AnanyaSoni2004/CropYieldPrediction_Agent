"""
Decision Agent (Main Orchestrator)
------------------------------------
Combines outputs from the Crop, Weather, Market, and RAG agents then
calls an LLM (Groq / OpenAI) to generate an explainable recommendation.
"""
import os
from typing import Any

from groq import Groq

GROQ_API_KEY  = os.getenv("GROQ_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Model selection – prefer Groq (free-tier) over OpenAI
GROQ_MODEL   = "llama3-70b-8192"
OPENAI_MODEL = "gpt-4o-mini"

SYSTEM_PROMPT = """You are AgroAgent, an expert agricultural advisory AI.
You receive structured data from multiple specialised agents and produce a
clear, evidence-based crop recommendation for a farmer.

Always respond in this EXACT format:

Recommended Crop: <crop_name>

Reasoning:
- Soil analysis: <findings>
- Weather conditions: <findings>
- Market trends: <findings>

Additional Advice:
- Fertilizer usage: <advice>
- Best practices: <advice>
- Risk mitigation: <advice>

Keep each bullet concise (1-2 sentences). Be practical and farmer-friendly."""


class DecisionAgent:
    """Uses an LLM to synthesise all agent outputs into a final recommendation."""

    def __init__(self):
        self._client = self._init_client()

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def decide(
        self,
        crop_result:   dict[str, Any],
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
            "llm_response":     str,   # full formatted output
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
        """Try Groq first, fall back to OpenAI, then return a template response."""
        if self._client and GROQ_API_KEY:
            try:
                resp = self._client.chat.completions.create(
                    model=GROQ_MODEL,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user",   "content": user_message},
                    ],
                    temperature=0.3,
                    max_tokens=800,
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
                    max_tokens=800,
                )
                return resp.choices[0].message.content.strip(), OPENAI_MODEL
            except Exception:
                pass

        # Deterministic fallback when no API key is configured
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
        return f"""
=== CROP PREDICTION AGENT ===
Top ML predictions: {top_crops_str}
Best ML prediction: {crop['top_prediction']} (confidence {crop['confidence']*100:.1f}%)
Soil – Nitrogen: {crop['soil_summary']['nitrogen']}, Phosphorus: {crop['soil_summary']['phosphorus']},
       Potassium: {crop['soil_summary']['potassium']}, pH: {crop['soil_summary']['ph_status']},
       Rainfall: {crop['soil_summary']['rainfall_mm']} mm

=== WEATHER AGENT ===
Location:    {weather['location']}
Temperature: {weather['temperature']}°C
Humidity:    {weather['humidity']}%
Conditions:  {weather['description']}
Suitability: {weather['suitability']}

=== MARKET AGENT ===
Best market crop: {market['best_market_crop']}
Ranked crops: {ranked_str}
Market insights: {market['market_insights']}

=== RAG KNOWLEDGE BASE ===
{rag if rag else 'No specific knowledge retrieved.'}

Based on all of the above data, provide your expert agricultural recommendation.
""".strip()

    @staticmethod
    def _extract_crop(text: str, fallback: str) -> str:
        """Parse 'Recommended Crop: <name>' from the LLM output."""
        for line in text.splitlines():
            if line.lower().startswith("recommended crop:"):
                return line.split(":", 1)[1].strip()
        return fallback

    @staticmethod
    def _fallback_response(prompt: str) -> str:
        # Extract crop name from prompt for a minimal coherent response
        crop = "the predicted crop"
        for line in prompt.splitlines():
            if "Best ML prediction:" in line:
                crop = line.split(":", 1)[1].split("(")[0].strip()
                break
        return (
            f"Recommended Crop: {crop}\n\n"
            "Reasoning:\n"
            "- Soil analysis: Soil parameters align with the predicted crop's requirements.\n"
            "- Weather conditions: Current weather is within acceptable range for this crop.\n"
            "- Market trends: This crop shows reasonable market demand and pricing.\n\n"
            "Additional Advice:\n"
            "- Fertilizer usage: Apply balanced NPK fertilizer as per soil test recommendations.\n"
            "- Best practices: Ensure proper irrigation scheduling and pest monitoring.\n"
            "- Risk mitigation: Monitor weather forecasts and adjust irrigation accordingly.\n\n"
            "(Note: LLM API key not configured – response generated by rule-based fallback.)"
        )
