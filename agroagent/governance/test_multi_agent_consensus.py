"""
Test 4 — Multi-Agent Consensus
==============================
For each scenario, shows the full consensus picture:
    • Individual agent outputs (votes + reasons)
    • Consensus score
    • Trust scores (before → after)
    • Final decision (ALLOW / BLOCK)
    • Rejected decisions (REJECT votes, and the withheld crop on a BLOCK)

Then answers: does consensus IMPROVE recommendation quality? Quality is judged by
an INDEPENDENT oracle, NOT by the governance policy itself:

    • validity label  — agronomic hard-range check on the raw inputs
                        (should we recommend at all?)
    • soundness oracle — KB-documented ideal-range fit of the surfaced crop to the
                        actual conditions (is the surfaced crop agronomically sensible?)

Baseline (ungoverned) surfaces the pipeline's recommendation; governed surfaces it
only on ALLOW. We compare both against the oracle.

Run:  python governance/test_multi_agent_consensus.py
"""
from __future__ import annotations

import glob
import os
import re
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv
load_dotenv()

from governance.governed_orchestrator import run_governed       # noqa: E402
from governance.pv_coordination import VOTERS                   # noqa: E402

DOCS = os.path.join(os.path.dirname(__file__), "..", "rag", "documents")
BLOCK_TOKENS = {"invalid input", "no suitable crop"}

# ---- independent agronomic oracle (NOT the governance policy) --------------

def _parse_ranges(path):
    t = open(path, encoding="utf-8").read()
    def rng(label):
        m = re.search(rf"{label}:\s*([\d.]+)\s*[–-]\s*([\d.]+)", t)
        return (float(m.group(1)), float(m.group(2))) if m else None
    return {"temp": rng("Temperature"), "rain": rng("Rainfall"), "hum": rng("Humidity")}

KB_RANGES = {os.path.basename(p).replace(".txt", "").lower(): _parse_ranges(p)
             for p in sorted(glob.glob(os.path.join(DOCS, "*.txt")))}


def _fit1(val, rng):
    if not rng:
        return 0.0
    lo, hi = rng
    if lo <= val <= hi:
        return 1.0
    d = (lo - val) if val < lo else (val - hi)
    return max(0.0, 1 - d / max(hi - lo, 1e-6))


def agronomic_fit(crop, temp, hum, rain):
    """Independent soundness oracle for a surfaced crop (None if no KB doc)."""
    r = KB_RANGES.get(crop.lower())
    if not r:
        return None
    return round((_fit1(temp, r["temp"]) + _fit1(hum, r["hum"]) + _fit1(rain, r["rain"])) / 3, 3)


def valid_inputs(soil):
    """Independent validity label from documented hard agronomic ranges."""
    issues = []
    if not (0 <= soil["temperature"] <= 50): issues.append("temp")
    if not (4 <= soil["ph"] <= 9):           issues.append("ph")
    if not (1 <= soil["humidity"] <= 99):    issues.append("humidity")
    if not (100 <= soil["rainfall"] <= 4000):issues.append("rainfall")
    if soil["N"] < 10 or soil["P"] < 10 or soil["K"] < 10: issues.append("NPK")
    return (len(issues) == 0), issues


# ---- scenarios (label = should we recommend, from validity oracle) ---------

SCENARIOS = [
    ("valid_rice_humid",   dict(N=90,  P=42, K=43, temperature=24, humidity=80, ph=6.5, rainfall=1500), "Mumbai"),
    ("valid_maize_moderate",dict(N=100, P=50, K=50, temperature=24, humidity=60, ph=6.8, rainfall=700), "Bengaluru"),
    ("valid_cool_wheat",   dict(N=60,  P=40, K=40, temperature=18, humidity=55, ph=6.8, rainfall=550), "Pune"),
    ("valid_marginal_dry", dict(N=45,  P=30, K=30, temperature=33, humidity=45, ph=6.0, rainfall=180), "Nagpur"),
    ("invalid_extreme_ph", dict(N=2,   P=3,  K=4,  temperature=25, humidity=50, ph=13.0, rainfall=40),  "Delhi"),
    ("invalid_low_rain",   dict(N=20,  P=20, K=20, temperature=30, humidity=60, ph=6.0, rainfall=30),  "Jaipur"),
]


def main():
    rows = []
    print("Running scenarios through the governed pipeline (live)...\n")
    for name, soil, loc in SCENARIOS:
        should_rec, issues = valid_inputs(soil)
        out = run_governed(soil, location=loc)
        g = out["governance"]
        state = out["pipeline_state"]
        actual_temp = state["weather_result"].get("temperature", soil["temperature"])
        actual_hum  = state["weather_result"].get("humidity", soil["humidity"])

        before = out["ungoverned_recommendation"]
        base_surfaces = before.strip().lower() not in BLOCK_TOKENS
        gov_surfaces  = g["final_status"] == "ALLOW"

        fit = agronomic_fit(before, actual_temp, actual_hum, soil["rainfall"]) if base_surfaces else None

        print("=" * 74)
        print(f"CASE: {name}   [label: {'RECOMMEND' if should_rec else 'WITHHOLD'}"
              f"{' (issues: '+','.join(issues)+')' if issues else ''}]")
        print("=" * 74)
        print(f"  Individual agent outputs (vote — reason):")
        for v in g["votes"]:
            drift = "  [DRIFT-IGNORED]" if v["context"].get("drift") else ""
            print(f"     {v['agent_id']:<14}{v['decision']:<8}{v['reason']}{drift}")
        print(f"  Consensus score : {g['consensus_score']} → {g['consensus']} "
              f"(threshold {g['threshold']}, active trust {g['active_trust']})")
        print(f"  Trust scores    : "
              + ", ".join(f"{a} {g['trust_in'][a]:.3f}→{g['trust_out'][a]:.3f}" for a in VOTERS))
        print(f"  Policy gate     : {'PASS' if g['policy_pass'] else 'FAIL'} — {g['policy_reason']}")
        print(f"  FINAL DECISION  : {g['final_status']}  → {out['governed_recommendation']}")
        rejected = [v["agent_id"] for v in g["votes"] if v["decision"] == "REJECT"]
        print(f"  Rejected by     : {rejected or 'none'}"
              + (f"   |  withheld crop: '{before}'" if g['final_status'] == 'BLOCK' else ""))
        if fit is not None:
            print(f"  [oracle] surfaced '{before}' agronomic fit to "
                  f"{actual_temp:.0f}°C/{actual_hum:.0f}%/{soil['rainfall']}mm = {fit} "
                  f"({'sound' if fit>=0.5 else 'UNSOUND'})")
        print()

        rows.append({
            "name": name, "should_rec": should_rec,
            "before": before, "base_surfaces": base_surfaces,
            "gov_surfaces": gov_surfaces, "final": g["final_status"],
            "consensus_score": g["consensus_score"], "fit": fit,
        })

    quality_report(rows)


def quality_report(rows):
    print("#" * 74)
    print("DOES CONSENSUS IMPROVE RECOMMENDATION QUALITY?")
    print("#" * 74)

    # (A) Action correctness vs the independent validity label
    def action_correct(r, surfaces):
        return (r["should_rec"] and surfaces) or (not r["should_rec"] and not surfaces)
    base_acc = sum(action_correct(r, r["base_surfaces"]) for r in rows)
    gov_acc  = sum(action_correct(r, r["gov_surfaces"]) for r in rows)
    n = len(rows)

    # (B) Soundness of surfaced crops vs the independent KB-fit oracle
    def unsound_surfaced(surfaces_key):
        return sum(1 for r in rows
                   if r[surfaces_key] and r["fit"] is not None and r["fit"] < 0.5)
    base_unsound = unsound_surfaced("base_surfaces")
    gov_unsound  = unsound_surfaced("gov_surfaces")

    print(f"\n(A) Action correctness (surface valid / withhold invalid), vs validity label:")
    print(f"     BEFORE (ungoverned): {base_acc}/{n}")
    print(f"     AFTER  (governed)  : {gov_acc}/{n}")

    print(f"\n(B) Unsound crops surfaced (KB-fit < 0.5), vs agronomic oracle:")
    print(f"     BEFORE (ungoverned): {base_unsound}")
    print(f"     AFTER  (governed)  : {gov_unsound}")

    print("\nPer-case action comparison:")
    print(f"  {'case':<22}{'label':<11}{'before':<10}{'after':<8}{'agree w/ label?'}")
    for r in rows:
        b = "surface" if r["base_surfaces"] else "withhold"
        a = "surface" if r["gov_surfaces"] else "withhold"
        ok_b = "✓" if (r["should_rec"] == r["base_surfaces"]) else "✗"
        ok_a = "✓" if (r["should_rec"] == r["gov_surfaces"]) else "✗"
        print(f"  {r['name']:<22}{('RECOMMEND' if r['should_rec'] else 'WITHHOLD'):<11}"
              f"{b+' '+ok_b:<10}{a+' '+ok_a:<8}")

    print("\nConclusion:")
    if gov_acc > base_acc or gov_unsound < base_unsound:
        print(f"  Consensus IMPROVED quality: action correctness {base_acc}→{gov_acc}/{n}, "
              f"unsound surfaced {base_unsound}→{gov_unsound}.")
    elif gov_acc == base_acc and gov_unsound == base_unsound:
        print("  Consensus MATCHED the baseline's outcomes on these cases — its value here is")
        print("  robustness (independent confidence floor, drift handling, audit, no single-agent")
        print("  override), not flipping decisions. Quality gains appear when the LLM is")
        print("  over-permissive or thresholds are tightened (see note in the docs).")
    else:
        print(f"  Consensus was MORE CONSERVATIVE: action correctness {base_acc}→{gov_acc}/{n} "
              f"(it withheld some valid cases). Tune thresholds to trade safety vs coverage.")


if __name__ == "__main__":
    main()
