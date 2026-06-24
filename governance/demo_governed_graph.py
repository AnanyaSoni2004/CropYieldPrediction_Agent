"""
Demo — LangGraph-native governed workflow
=========================================
Exercises all three governance routes embedded in the graph:

    • ALLOW    (auto)  — strong consensus, no human needed
    • BLOCK    (auto)  — policy gate fails (invalid input)
    • ESCALATE (pause) — low-confidence pass → interrupt → human approves / rejects

Shows the graph topology, the pause/resume via checkpointer, and the final outcome.

Run:  python governance/demo_governed_graph.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv
load_dotenv()

from governance.governed_graph import (build_governed_graph,
                                        run_to_completion_or_pause,
                                        resume_with_human_decision)

CASES = {
    "mumbai_valid":  (dict(N=90, P=42, K=43, temperature=24, humidity=80, ph=6.5, rainfall=1500), "Mumbai"),
    "delhi_invalid": (dict(N=2, P=3, K=4, temperature=25, humidity=50, ph=13.0, rainfall=40), "Delhi"),
    "bengaluru_lowconf": (dict(N=120, P=80, K=70, temperature=25, humidity=60, ph=7.0, rainfall=1200), "Bengaluru"),
}


def show(snap_state):
    g = snap_state.get("governance", {})
    fo = snap_state.get("final_outcome")
    print(f"    class={snap_state.get('decision_class')}  "
          f"consensus={g.get('consensus')}({g.get('consensus_score')})  "
          f"ml_conf={g.get('request',{}).get('ml_confidence')}")
    if fo:
        print(f"    FINAL: {fo['status']} → {fo['crop']}  [{fo['path']}: {fo['reason']}]")


def main():
    graph = build_governed_graph()

    print("GRAPH TOPOLOGY (mermaid)\n" + "-" * 60)
    try:
        print(graph.get_graph().draw_mermaid())
    except Exception as e:
        print(f"(mermaid render unavailable: {e})")

    # ---- 1. ALLOW (auto) ----
    print("\n" + "=" * 70)
    print("CASE 1 — Mumbai (valid, strong): expect AUTO-ALLOW (no interrupt)")
    print("=" * 70)
    soil, loc = CASES["mumbai_valid"]
    r = run_to_completion_or_pause(graph, soil, loc, thread_id="mumbai")
    print(f"  paused? {r['is_paused']}")
    show(r["state"])

    # ---- 2. BLOCK (auto) ----
    print("\n" + "=" * 70)
    print("CASE 2 — Delhi (invalid pH/rainfall/NPK): expect AUTO-BLOCK (no interrupt)")
    print("=" * 70)
    soil, loc = CASES["delhi_invalid"]
    r = run_to_completion_or_pause(graph, soil, loc, thread_id="delhi")
    print(f"  paused? {r['is_paused']}")
    show(r["state"])

    # ---- 3. ESCALATE → human APPROVES ----
    print("\n" + "=" * 70)
    print("CASE 3 — Bengaluru (valid but low ML confidence): expect ESCALATE → PAUSE")
    print("=" * 70)
    soil, loc = CASES["bengaluru_lowconf"]
    r = run_to_completion_or_pause(graph, soil, loc, thread_id="beng_approve")
    print(f"  paused? {r['is_paused']}  (interrupt_before → {r['paused_before']})")
    show(r["state"])
    print("  …workflow is suspended; state + audit persisted by checkpointer.")
    print("  → human reviewer APPROVES; resuming…")
    final = resume_with_human_decision(graph, "beng_approve", "approve")
    show(final)

    # ---- 4. ESCALATE → human REJECTS (same input, different reviewer choice) ----
    print("\n" + "=" * 70)
    print("CASE 4 — same Bengaluru decision, reviewer REJECTS")
    print("=" * 70)
    r = run_to_completion_or_pause(graph, soil, loc, thread_id="beng_reject")
    print(f"  paused? {r['is_paused']}")
    print("  → human reviewer REJECTS; resuming…")
    final = resume_with_human_decision(graph, "beng_reject", "reject")
    show(final)

    print("\n" + "#" * 70)
    print("Governance is now a NODE in the graph with conditional routing + HITL:")
    print("  • allow  → finalize → END         (auto)")
    print("  • block  → finalize → END         (auto)")
    print("  • escalate → [interrupt] → human_approval → END  (resumes via checkpointer)")
    print("#" * 70)


if __name__ == "__main__":
    main()
