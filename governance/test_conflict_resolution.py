"""
Test 1 — Agent Conflict Resolution
==================================
Constructs a scenario where AgroAgent's agents favor DIFFERENT crops and shows:

    • What happens TODAY (ungoverned AgroAgent)
    • What happens WITH PrivateVault consensus (coordination layer)

Each agent's crop preference is derived from a REAL signal:
    weather_agent → best fit to the KB's documented ideal temp/humidity/rainfall
    market_agent  → highest real profitability score (MarketAgent)
    kb_agent      → highest real ChromaDB similarity to the conditions
    ml_agent      → RandomForest top-1 prediction (CropPredictionAgent)

Run:  python governance/test_conflict_resolution.py
"""
from __future__ import annotations

import glob
import os
import re
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.market_agent import MarketAgent                       # noqa: E402
from agents.crop_prediction_agent import CropPredictionAgent      # noqa: E402
from rag.vector_store import VectorStore                          # noqa: E402
from governance.pv_coordination import CoordinationGovernor, resolve_crop_conflict  # noqa: E402

DOCS = os.path.join(os.path.dirname(__file__), "..", "rag", "documents")

# Candidate crops that all have a KB guide (so every agent can express a grounded view)
CANDIDATES = ["rice", "cotton", "maize", "wheat", "potato"]

# The conflict scenario: warm, humid, high-rainfall conditions.
SCENARIO = {"temperature": 30, "humidity": 75, "rainfall": 1500}
# Soil nutrients for the ML model (kept fixed; weather/humidity/rainfall come from SCENARIO)
SOIL = {"N": 80, "P": 45, "K": 45, "ph": 6.5, **SCENARIO}


# ---- grounded per-agent crop preference signals ---------------------------

def _parse_ranges(path):
    t = open(path, encoding="utf-8").read()
    def rng(label):
        m = re.search(rf"{label}:\s*([\d.]+)\s*[–-]\s*([\d.]+)", t)
        return (float(m.group(1)), float(m.group(2))) if m else None
    return {"temp": rng("Temperature"), "rain": rng("Rainfall"), "hum": rng("Humidity")}

KB_RANGES = {os.path.basename(p).replace(".txt", ""): _parse_ranges(p)
             for p in sorted(glob.glob(os.path.join(DOCS, "*.txt")))}


def _fit(val, rng):
    if not rng:
        return 0.0
    lo, hi = rng
    if lo <= val <= hi:
        return 1.0
    d = (lo - val) if val < lo else (val - hi)
    return max(0.0, 1 - d / max(hi - lo, 1e-6))


def weather_scores():
    return {c: round((_fit(SCENARIO["temperature"], KB_RANGES[c]["temp"])
                      + _fit(SCENARIO["humidity"], KB_RANGES[c]["hum"])
                      + _fit(SCENARIO["rainfall"], KB_RANGES[c]["rain"])) / 3, 3)
            for c in CANDIDATES}


def market_scores():
    ranked = MarketAgent().analyse(CANDIDATES)["ranked_crops"]
    return {r["crop"]: r["profitability_score"] for r in ranked}


def kb_scores():
    vs = VectorStore()
    q = (f"ideal crop for temperature {SCENARIO['temperature']} C "
         f"humidity {SCENARIO['humidity']} percent rainfall {SCENARIO['rainfall']} mm cultivation")
    hits = vs.query(q, n_results=10)
    out = {c: 0.0 for c in CANDIDATES}
    for h in hits:
        crop = h["metadata"].get("crop", "").lower()
        if crop in out:
            out[crop] = max(out[crop], round(1 - h["distance"], 3))
    return out


def ml_pick():
    return CropPredictionAgent().predict(SOIL)["top_prediction"]


# ---------------------------------------------------------------------------

def main():
    ws, ms, ks = weather_scores(), market_scores(), kb_scores()
    ml = ml_pick()

    nominations = {
        "weather_agent": max(CANDIDATES, key=lambda c: ws[c]),
        "market_agent":  max(CANDIDATES, key=lambda c: ms.get(c, 0)),
        "kb_agent":      max(CANDIDATES, key=lambda c: ks[c]),
        "ml_agent":      ml,
    }

    print(f"\nScenario: {SCENARIO}  (candidates: {', '.join(CANDIDATES)})\n")
    print(f"{'crop':<9}{'weather_fit':<13}{'market':<9}{'kb_sim':<8}")
    for c in CANDIDATES:
        print(f"{c:<9}{ws[c]:<13}{ms.get(c,0):<9}{ks[c]:<8}")

    print("\n--- Each agent's preferred crop (from its own signal) ---")
    for a, c in nominations.items():
        print(f"  {a:<14} → {c}")
    distinct = len(set(nominations.values()))
    print(f"  ({distinct} distinct crops nominated → "
          f"{'CONFLICT' if distinct > 1 else 'agreement'})")

    # =====================================================================
    # WHAT HAPPENS TODAY (ungoverned)
    # =====================================================================
    print("\n" + "=" * 72)
    print("TODAY (ungoverned AgroAgent)")
    print("=" * 72)
    print(f"  Final recommendation = ML top-1 = '{ml}'.")
    print("  • The DecisionAgent always defers to the ML prediction on valid input.")
    print(f"  • market_agent preferred '{nominations['market_agent']}' — computed, shown, then IGNORED.")
    print("  • weather_agent and kb_agent never vote on crop choice at all.")
    print("  • The disagreement is neither resolved nor recorded — ML simply wins.")

    # =====================================================================
    # WHAT HAPPENS WITH PRIVATEVAULT CONSENSUS
    # =====================================================================
    print("\n" + "=" * 72)
    print("WITH PrivateVault consensus (coordination layer)")
    print("=" * 72)

    gov = CoordinationGovernor()
    res = gov.govern_conflict(nominations, quorum_frac=0.5)

    print("  Trust-weighted votes per nominated crop:")
    for c, info in sorted(res["per_crop"].items(),
                          key=lambda kv: -kv[1]["support_weight"]):
        print(f"     {c:<9} support={info['support_frac']*100:5.1f}%  "
              f"weight={info['support_weight']:<7} backers={info['backers']}")
    print(f"  Quorum required: {res['quorum_frac']*100:.0f}% of trust weight")
    if res["status"] == "RESOLVED":
        print(f"  → CONSENSUS RESOLVED to '{res['winner']}' "
              f"({res['consensus_frac']*100:.0f}% trust-weighted support)")
        if res["winner"] != ml:
            print(f"    (note: differs from today's ML pick '{ml}' — consensus overrode it)")
    else:
        print(f"  → NO QUORUM ({res['consensus_frac']*100:.0f}% < "
              f"{res['quorum_frac']*100:.0f}%): ESCALATE to human review")
        print("    (PrivateVault refuses to let one agent silently win a genuine deadlock)")
    print(f"  Audit entry: {res['audit']['entry_hash'][:16]}…  "
          f"(prev {res['audit']['prev_hash'][:8]}…)")

    # =====================================================================
    # Effect of TRUST history on the same conflict
    # =====================================================================
    print("\n" + "=" * 72)
    print("SAME conflict, but market_agent has earned higher trust over time")
    print("=" * 72)
    biased_trust = {"weather_agent": 1.0, "market_agent": 2.0,
                    "kb_agent": 1.0, "ml_agent": 1.0}
    res2 = resolve_crop_conflict(nominations, biased_trust, quorum_frac=0.4)
    print(f"  Trust weights: {biased_trust}")
    for c, info in sorted(res2["per_crop"].items(),
                          key=lambda kv: -kv[1]["support_weight"]):
        print(f"     {c:<9} support={info['support_frac']*100:5.1f}%  backers={info['backers']}")
    if res2["status"] == "RESOLVED":
        print(f"  → CONSENSUS RESOLVED to '{res2['winner']}' — the higher-trust "
              f"agent's nomination now carries the vote.")
    else:
        print(f"  → ESCALATE ({res2['consensus_frac']*100:.0f}% support)")

    print("\nKey contrast:")
    print("  TODAY        : ML wins unconditionally; other agents' views discarded; no audit.")
    print("  PRIVATEVAULT : every agent gets a trust-weighted vote; consensus either")
    print("                 resolves to a backed crop or escalates a true deadlock — all audited.\n")


if __name__ == "__main__":
    main()
