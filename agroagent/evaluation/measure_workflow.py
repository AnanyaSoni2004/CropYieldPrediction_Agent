"""
AgroAgent Workflow Measurement Harness
=======================================
Instruments the existing LangGraph multi-agent pipeline and measures:

  1. Agent decisions        – the key output each agent commits to
  2. Agent disagreements    – where agents pick different crops / verdicts
  3. Tool calls             – external/IO calls (weather API, LLM, RF model, vector DB, CSV)
  4. Recommendation quality – confidence, cross-agent agreement, validation correctness
  5. Time to final recommendation – per-node and end-to-end latency

The harness does NOT modify the agents. It mirrors the orchestrator's node
order so it can observe every intermediate result, and it monkey-patches the
five tool boundaries to count real calls.

Run:  python evaluation/measure_workflow.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from collections import Counter
from contextlib import contextmanager

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
load_dotenv()

from agents.crop_prediction_agent import CropPredictionAgent
from agents.weather_agent import WeatherAgent
from agents.market_agent import MarketAgent
from agents.decision_agent import DecisionAgent
from agents.rag_knowledge_agent import RAGKnowledgeAgent

# ---------------------------------------------------------------------------
# Tool-call instrumentation: patch the 5 real IO/compute boundaries
# ---------------------------------------------------------------------------

TOOL_CALLS: Counter = Counter()


@contextmanager
def instrument_tool_calls():
    """Patch the real tool boundaries so each invocation is counted by category."""
    import agents.weather_agent as wx
    import groq.resources.chat.completions as gc
    import sklearn.ensemble._forest as rf
    import rag.vector_store as vs
    import pandas as pd

    orig_get      = wx.requests.get
    orig_llm      = gc.Completions.create
    orig_proba    = rf.ForestClassifier.predict_proba
    orig_vquery   = vs.VectorStore.query
    orig_readcsv  = pd.read_csv

    def get_patched(*a, **k):
        TOOL_CALLS["weather_api (OpenWeatherMap HTTP GET)"] += 1
        return orig_get(*a, **k)

    def llm_patched(self, *a, **k):
        TOOL_CALLS["llm_completion (Groq/OpenAI chat)"] += 1
        return orig_llm(self, *a, **k)

    def proba_patched(self, *a, **k):
        TOOL_CALLS["ml_inference (RandomForest.predict_proba)"] += 1
        return orig_proba(self, *a, **k)

    def vquery_patched(self, *a, **k):
        TOOL_CALLS["vector_search (ChromaDB query)"] += 1
        return orig_vquery(self, *a, **k)

    def readcsv_patched(*a, **k):
        TOOL_CALLS["csv_read (market_prices.csv)"] += 1
        return orig_readcsv(*a, **k)

    wx.requests.get               = get_patched
    gc.Completions.create         = llm_patched
    rf.ForestClassifier.predict_proba = proba_patched
    vs.VectorStore.query          = vquery_patched
    pd.read_csv                   = readcsv_patched
    try:
        yield
    finally:
        wx.requests.get               = orig_get
        gc.Completions.create         = orig_llm
        rf.ForestClassifier.predict_proba = orig_proba
        vs.VectorStore.query          = orig_vquery
        pd.read_csv                   = orig_readcsv


# ---------------------------------------------------------------------------
# Instrumented run: mirrors orchestrator node order, captures everything
# ---------------------------------------------------------------------------

def parse_decision_text(text: str) -> dict:
    """Pull the structured fields the DecisionAgent LLM is asked to emit."""
    import re
    fields = {}
    patterns = {
        "validation_status":  r"validation status\s*:\s*(\w+)",
        "environmental":      r"environmental assessment\s*:\s*(\w+)",
        "ml_review_status":   r"-?\s*status\s*:\s*([A-Za-z—\- ]+)",
        "confidence_level":   r"confidence level\s*:\s*(\w+)",
    }
    for key, pat in patterns.items():
        m = re.search(pat, text, re.IGNORECASE)
        fields[key] = m.group(1).strip() if m else None
    return fields


def measure_case(name: str, soil: dict, location: str, user_query: str = "") -> dict:
    """Run one input through the full pipeline with timing + decision capture."""
    timings = {}
    decisions = {}

    t_total0 = time.perf_counter()

    # ---- weather node ----
    t0 = time.perf_counter()
    weather = WeatherAgent().get_weather(location)
    timings["weather"] = time.perf_counter() - t0
    decisions["weather"] = {
        "source":      weather.get("source"),
        "temperature": weather.get("temperature"),
        "humidity":    weather.get("humidity"),
        "suitability": weather.get("suitability"),
    }

    # ---- crop prediction node (live weather overrides temp/humidity) ----
    soil = dict(soil)
    if weather.get("source") == "openweathermap":
        soil["temperature"] = weather["temperature"]
        soil["humidity"]    = weather["humidity"]
    t0 = time.perf_counter()
    crop = CropPredictionAgent().predict(soil)
    timings["crop_prediction"] = time.perf_counter() - t0
    decisions["crop_prediction"] = {
        "top_prediction": crop["top_prediction"],
        "confidence":     crop["confidence"],
        "top_crops":      [(c["crop"], c["confidence"]) for c in crop["top_crops"]],
    }

    # ---- market node ----
    t0 = time.perf_counter()
    market = MarketAgent().analyse([c["crop"] for c in crop["top_crops"]])
    timings["market"] = time.perf_counter() - t0
    decisions["market"] = {
        "best_market_crop": market["best_market_crop"],
        "ranking":          [(r["crop"], r["profitability_score"]) for r in market["ranked_crops"]],
    }

    # ---- rag node ----
    t0 = time.perf_counter()
    rag_agent = RAGKnowledgeAgent()
    rag_context = rag_agent.retrieve_for_crop(crop["top_prediction"])
    if user_query:
        extra = rag_agent.query(user_query)
        rag_context = extra["answer"] + "\n\n" + rag_context
    timings["rag"] = time.perf_counter() - t0
    decisions["rag"] = {"context_chars": len(rag_context)}

    # ---- decision node ----
    t0 = time.perf_counter()
    decision = DecisionAgent().decide(
        crop_result=crop, weather_result=weather,
        market_result=market, rag_context=rag_context,
    )
    timings["decision"] = time.perf_counter() - t0
    parsed = parse_decision_text(decision["llm_response"])
    decisions["decision"] = {
        "recommended_crop": decision["recommended_crop"],
        "model_used":       decision["model_used"],
        **parsed,
    }

    timings["TOTAL"] = time.perf_counter() - t_total0

    # ---- disagreement analysis ----
    ml_crop     = crop["top_prediction"].lower()
    market_crop = market["best_market_crop"].lower()
    final_crop  = decision["recommended_crop"].lower()
    is_blocked  = decision["recommended_crop"] in ("Invalid Input", "No Suitable Crop")

    disagreements = {
        "ml_vs_market":  ml_crop != market_crop,
        "final_vs_ml":   (not is_blocked) and final_crop != ml_crop,
        "llm_blocked_ml": is_blocked,
        "weather_poor_but_recommended":
            weather.get("suitability", "").startswith("poor") and not is_blocked,
    }
    disagreement_count = sum(1 for v in disagreements.values() if v)

    # ---- recommendation quality scoring ----
    quality = score_quality(crop, weather, market, decision, parsed, disagreements)

    return {
        "case": name,
        "input": {"soil": soil, "location": location, "user_query": user_query},
        "timings_sec": {k: round(v, 3) for k, v in timings.items()},
        "decisions": decisions,
        "disagreements": disagreements,
        "disagreement_count": disagreement_count,
        "quality": quality,
        "final_recommendation": decision["recommended_crop"],
        "llm_response": decision["llm_response"],
    }


def score_quality(crop, weather, market, decision, parsed, disagreements) -> dict:
    """
    Heuristic recommendation-quality score (0-100), composed of:
      - ML confidence            (0-30)
      - Cross-agent agreement    (0-25)  ML==final, ML==market
      - Structured-output completeness (0-20) all 4 fields parsed
      - Decision decisiveness    (0-15)  produced a concrete crop OR correctly blocked
      - LLM (not fallback) used  (0-10)
    """
    score = 0
    notes = []

    conf = crop["confidence"]
    score += round(conf * 30)
    notes.append(f"ML confidence {conf*100:.1f}% -> {round(conf*30)}/30")

    agree = 0
    if not disagreements["final_vs_ml"] and not disagreements["llm_blocked_ml"]:
        agree += 15
        notes.append("final == ML (+15)")
    if not disagreements["ml_vs_market"]:
        agree += 10
        notes.append("ML == market best (+10)")
    score += agree

    completeness = sum(5 for f in ("validation_status", "environmental",
                                   "ml_review_status", "confidence_level")
                       if parsed.get(f))
    score += completeness
    notes.append(f"structured fields parsed -> {completeness}/20")

    final = decision["recommended_crop"]
    if final == "Invalid Input":
        score += 15
        notes.append("decisive block of invalid input (+15)")
    elif final == "No Suitable Crop":
        score += 10
        notes.append("decisive 'no suitable crop' (+10)")
    else:
        score += 15
        notes.append("concrete crop recommended (+15)")

    if decision["model_used"] != "rule-based-fallback":
        score += 10
        notes.append("real LLM used (+10)")
    else:
        notes.append("rule-based fallback (0/10)")

    return {"score": score, "breakdown": notes}


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

CASES = [
    ("valid_humid_coastal",
     dict(N=90, P=42, K=43, temperature=20.8, humidity=82, ph=6.5, rainfall=202),
     "Mumbai", ""),
    ("valid_high_npk_neutral",
     dict(N=120, P=80, K=70, temperature=25, humidity=60, ph=7.0, rainfall=1200),
     "Bengaluru", ""),
    ("suboptimal_drought",
     dict(N=40, P=30, K=30, temperature=42, humidity=25, ph=6.0, rainfall=300),
     "Jaipur", ""),
    ("invalid_extreme_ph_low_npk",
     dict(N=2, P=3, K=4, temperature=25, humidity=50, ph=12.0, rainfall=50),
     "Delhi", ""),
    ("valid_with_user_query",
     dict(N=85, P=55, K=45, temperature=28, humidity=70, ph=6.8, rainfall=900),
     "Hyderabad", "What fertilizer schedule should I follow?"),
]


def main():
    results = []
    with instrument_tool_calls():
        for name, soil, loc, q in CASES:
            print(f"\n{'='*70}\nRunning case: {name}\n{'='*70}")
            TOOL_CALLS.clear()
            res = measure_case(name, soil, loc, q)
            res["tool_calls"] = dict(TOOL_CALLS)
            res["tool_calls_total"] = sum(TOOL_CALLS.values())
            results.append(res)
            print(f"  final recommendation : {res['final_recommendation']}")
            print(f"  total latency        : {res['timings_sec']['TOTAL']}s")
            print(f"  tool calls           : {res['tool_calls_total']} {res['tool_calls']}")
            print(f"  disagreements        : {res['disagreement_count']} {[k for k,v in res['disagreements'].items() if v]}")
            print(f"  quality score        : {res['quality']['score']}/100")

    out_path = os.path.join(os.path.dirname(__file__), "measurement_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nFull results written to {out_path}")

    print_summary(results)


def print_summary(results):
    print(f"\n{'#'*70}\nSUMMARY ACROSS {len(results)} CASES\n{'#'*70}")
    print(f"{'case':<28}{'final':<16}{'latency':<10}{'tools':<7}{'disagr':<8}{'quality'}")
    for r in results:
        print(f"{r['case']:<28}{r['final_recommendation'][:15]:<16}"
              f"{r['timings_sec']['TOTAL']:<10}{r['tool_calls_total']:<7}"
              f"{r['disagreement_count']:<8}{r['quality']['score']}")

    avg_latency = sum(r["timings_sec"]["TOTAL"] for r in results) / len(results)
    avg_quality = sum(r["quality"]["score"] for r in results) / len(results)
    print(f"\nAvg latency: {avg_latency:.2f}s | Avg quality: {avg_quality:.1f}/100")

    # node latency breakdown (averaged)
    nodes = ["weather", "crop_prediction", "market", "rag", "decision"]
    print("\nAvg per-node latency (s):")
    for n in nodes:
        avg = sum(r["timings_sec"].get(n, 0) for r in results) / len(results)
        print(f"  {n:<18}{avg:.3f}")


if __name__ == "__main__":
    main()
