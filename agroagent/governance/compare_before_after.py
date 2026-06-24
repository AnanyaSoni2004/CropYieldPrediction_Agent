"""
Before / After PrivateVault comparison
=======================================
Runs the SAME inputs through:
  BEFORE — the ungoverned AgroAgent pipeline (whatever it recommends is surfaced)
  AFTER  — the pipeline wrapped with PrivateVault's coordination governance
           (consensus + trust + policy gate can withhold a recommendation)

Reports, per case: the recommended crop before, the governance verdict after,
the consensus score, the per-agent votes, trust deltas, and audit-chain status.

Run:  python governance/compare_before_after.py
"""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv
load_dotenv()

from governance.governed_orchestrator import run_governed          # noqa: E402
from governance.pv_coordination import verify_audit_chain, VOTERS  # noqa: E402

# Same representative inputs used by the Phase-1 measurement harness.
CASES = [
    ("valid_humid_coastal",
     dict(N=90, P=42, K=43, temperature=20.8, humidity=82, ph=6.5, rainfall=202), "Mumbai", ""),
    ("valid_high_npk_neutral",
     dict(N=120, P=80, K=70, temperature=25, humidity=60, ph=7.0, rainfall=1200), "Bengaluru", ""),
    ("suboptimal_drought",
     dict(N=40, P=30, K=30, temperature=42, humidity=25, ph=6.0, rainfall=300), "Jaipur", ""),
    ("invalid_extreme_ph_low_npk",
     dict(N=2, P=3, K=4, temperature=25, humidity=50, ph=12.0, rainfall=50), "Delhi", ""),
    ("valid_with_user_query",
     dict(N=85, P=55, K=45, temperature=28, humidity=70, ph=6.8, rainfall=900),
     "Hyderabad", "What fertilizer schedule should I follow?"),
]


def main():
    results = []
    for name, soil, loc, q in CASES:
        print(f"\n{'='*72}\nCASE: {name}\n{'='*72}")
        t0 = time.perf_counter()
        out = run_governed(soil, location=loc, user_query=q)
        elapsed = round(time.perf_counter() - t0, 3)
        g = out["governance"]

        before = out["ungoverned_recommendation"]
        after_status = g["final_status"]
        after_crop = out["governed_recommendation"]
        changed = (after_status == "BLOCK")

        print(f"  BEFORE (ungoverned) : {before}")
        print(f"  AFTER  (governed)   : {after_status}  → {after_crop}")
        print(f"  consensus           : {g['consensus']} (score={g['consensus_score']}, "
              f"threshold={g['threshold']}, active_trust={g['active_trust']})")
        print(f"  policy              : {'PASS' if g['policy_pass'] else 'FAIL'} — {g['policy_reason']}")
        print("  votes:")
        for v in g["votes"]:
            drift = "  [DRIFT-IGNORED]" if v["context"].get("drift") else ""
            print(f"     {v['agent_id']:<14}{v['decision']:<8}{v['reason']}{drift}")
        print(f"  trust Δ             : "
              + ", ".join(f"{a}:{g['trust_in'][a]:.3f}→{g['trust_out'][a]:.3f}" for a in VOTERS))
        print(f"  audit entry_hash    : {g['audit']['entry_hash'][:16]}…  (prev {g['audit']['prev_hash'][:8]}…)")
        print(f"  governance changed outcome: {'YES (withheld)' if changed else 'no'}   [{elapsed}s]")

        results.append({
            "case": name,
            "before_recommendation": before,
            "after_status": after_status,
            "after_recommendation": after_crop,
            "governance_changed_outcome": changed,
            "consensus": g["consensus"],
            "consensus_score": g["consensus_score"],
            "policy_pass": g["policy_pass"],
            "votes": [{"agent": v["agent_id"], "decision": v["decision"],
                       "drift": bool(v["context"].get("drift"))} for v in g["votes"]],
            "trust_in": g["trust_in"],
            "trust_out": g["trust_out"],
            "audit_hash": g["audit"]["entry_hash"],
            "elapsed_sec": elapsed,
        })

    chain_ok = verify_audit_chain()
    print_summary(results, chain_ok)

    out_path = os.path.join(os.path.dirname(__file__), "comparison_results.json")
    with open(out_path, "w") as f:
        json.dump({"audit_chain_verified": chain_ok, "cases": results}, f, indent=2, default=str)
    print(f"\nFull comparison written to {out_path}")


def print_summary(results, chain_ok):
    print(f"\n{'#'*72}\nBEFORE vs AFTER PrivateVault — SUMMARY\n{'#'*72}")
    print(f"{'case':<28}{'BEFORE':<16}{'AFTER':<10}{'crop':<22}{'changed'}")
    for r in results:
        print(f"{r['case']:<28}{r['before_recommendation'][:15]:<16}"
              f"{r['after_status']:<10}{r['after_recommendation'][:21]:<22}"
              f"{'YES' if r['governance_changed_outcome'] else 'no'}")
    blocked = sum(1 for r in results if r["after_status"] == "BLOCK")
    print(f"\nGoverned BLOCK verdicts: {blocked}/{len(results)} "
          f"(ungoverned pipeline would have surfaced all {len(results)})")
    print(f"Audit chain integrity (tamper-evidence): {'VERIFIED ✓' if chain_ok else 'BROKEN ✗'}")


if __name__ == "__main__":
    main()
