"""
LangGraph-native governed workflow
==================================
Embeds PrivateVault governance INSIDE the LangGraph execution graph (rather than
wrapping it post-hoc). The control is a real pre-execution gate with human-in-the-loop.

Graph topology:

    START → weather → crop_prediction → market → rag → decision → governance
                                                                      │
                                              ┌───────────────────────┼───────────────────────┐
                                        (allow)│                 (block)│                (escalate)│
                                              ▼                        ▼                          ▼
                                          finalize                 finalize              human_approval ⏸ (interrupt)
                                              │                        │                          │
                                              ▼                        ▼                          ▼
                                             END                      END                        END

- `governance` runs the CoordinationGovernor (consensus + trust + policy) and
  classifies the decision into allow / block / escalate.
- A CONDITIONAL EDGE routes on that class.
- `human_approval` is guarded by `interrupt_before`, so execution PAUSES there;
  a human resumes it via `graph.update_state(...); graph.invoke(None, config)`.
- A MemorySaver checkpointer persists state + audit across the interrupt.

Public API:
    build_governed_graph() -> compiled graph (with checkpointer + interrupt)
    run_to_completion_or_pause(graph, soil, location, thread_id) -> snapshot
    resume_with_human_decision(graph, thread_id, decision) -> final state
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from typing import Any, TypedDict

import numpy as np
from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.memory import MemorySaver

from agents.orchestrator import (weather_node, crop_prediction_node,
                                  market_node, rag_node, decision_node)
from governance.pv_coordination import CoordinationGovernor

# Escalation thresholds: a *passing* recommendation still goes to a human when it
# is only weakly supported.
ESCALATE_CONF_BELOW = 0.40   # ML confidence below this → human review even on ALLOW
ESCALATE_IF_ANY_REJECT = True  # any dissenting agent on an ALLOW → human review


class GovernedState(TypedDict, total=False):
    # ---- pipeline inputs/intermediates (mirror AgentState) ----
    soil_data:   dict[str, float]
    location:    str
    user_query:  str
    crop_result:    dict[str, Any]
    weather_result: dict[str, Any]
    market_result:  dict[str, Any]
    rag_context:    str
    final_recommendation: dict[str, Any]
    # ---- governance ----
    governance:     dict[str, Any]
    decision_class: str            # "allow" | "block" | "escalate"
    human_decision: str            # "approve" | "reject"  (supplied on resume)
    final_outcome:  dict[str, Any]


_governor: CoordinationGovernor | None = None


def _get_governor() -> CoordinationGovernor:
    global _governor
    if _governor is None:
        _governor = CoordinationGovernor()
    return _governor


# ---------------------------------------------------------------------------
# New nodes
# ---------------------------------------------------------------------------

def governance_node(state: GovernedState) -> dict:
    """Run coordination governance and classify the decision into 3 routes."""
    gov = _get_governor().govern(state)

    if gov["final_status"] == "BLOCK":
        cls = "block"
    else:  # ALLOW from the governor — decide whether it's clear or needs a human
        conf = gov["request"]["ml_confidence"]
        any_reject = any(v["decision"] == "REJECT" for v in gov["votes"])
        if conf < ESCALATE_CONF_BELOW or (ESCALATE_IF_ANY_REJECT and any_reject):
            cls = "escalate"
        else:
            cls = "allow"
    return {"governance": gov, "decision_class": cls}


def finalize_node(state: GovernedState) -> dict:
    """Terminal node for clear allow/block decisions (no human needed)."""
    gov = state["governance"]
    if state["decision_class"] == "allow":
        return {"final_outcome": {
            "status": "ALLOW", "crop": gov["ungoverned_crop"],
            "path": "auto", "reason": "strong consensus, no review required"}}
    return {"final_outcome": {
        "status": "BLOCK", "crop": "Withheld — needs review",
        "path": "auto", "reason": gov["policy_reason"]}}


def human_approval_node(state: GovernedState) -> dict:
    """
    Human-in-the-loop gate. Execution is PAUSED before this node (interrupt_before);
    by the time it runs, a human has supplied `human_decision` via update_state.
    """
    gov = state["governance"]
    decision = state.get("human_decision", "reject")
    if decision == "approve":
        return {"final_outcome": {
            "status": "ALLOW (human-approved)", "crop": gov["ungoverned_crop"],
            "path": "human", "reason": "escalated low-confidence decision approved by reviewer"}}
    return {"final_outcome": {
        "status": "BLOCK (human-rejected)", "crop": "Withheld",
        "path": "human", "reason": "escalated decision rejected by reviewer"}}


def _route(state: GovernedState) -> str:
    return state["decision_class"]


def _native(o):
    """Recursively coerce numpy scalars/arrays to JSON-native types so the
    checkpointer (msgpack) can serialise the state across the interrupt."""
    if isinstance(o, dict):
        return {k: _native(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_native(v) for v in o]
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o


def _wrap(fn):
    """Wrap a node so its returned dict is coerced to native Python types."""
    def inner(state):
        return _native(fn(state))
    inner.__name__ = getattr(fn, "__name__", "node")
    return inner


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_governed_graph(checkpointer: Any | None = None):
    b = StateGraph(GovernedState)
    b.add_node("weather",         _wrap(weather_node))
    b.add_node("crop_prediction", _wrap(crop_prediction_node))
    b.add_node("market",          _wrap(market_node))
    b.add_node("rag",             _wrap(rag_node))
    b.add_node("decision",        _wrap(decision_node))
    b.add_node("governance",      _wrap(governance_node))
    b.add_node("human_approval",  _wrap(human_approval_node))
    b.add_node("finalize",        _wrap(finalize_node))

    b.add_edge(START,             "weather")
    b.add_edge("weather",         "crop_prediction")
    b.add_edge("crop_prediction", "market")
    b.add_edge("market",          "rag")
    b.add_edge("rag",             "decision")
    b.add_edge("decision",        "governance")

    # CONDITIONAL EDGE: route on the governance classification
    b.add_conditional_edges("governance", _route, {
        "allow":    "finalize",
        "block":    "finalize",
        "escalate": "human_approval",
    })
    b.add_edge("finalize",       END)
    b.add_edge("human_approval", END)

    return b.compile(
        checkpointer=checkpointer or MemorySaver(),
        interrupt_before=["human_approval"],   # PAUSE for the human here
    )


# ---------------------------------------------------------------------------
# Driver helpers
# ---------------------------------------------------------------------------

def run_to_completion_or_pause(graph, soil_data, location="New Delhi",
                               user_query="", thread_id="t1"):
    """Invoke the graph; returns the state snapshot (paused if it escalated)."""
    config = {"configurable": {"thread_id": thread_id}}
    graph.invoke({"soil_data": soil_data, "location": location,
                  "user_query": user_query}, config)
    snap = graph.get_state(config)
    paused = bool(snap.next)  # non-empty `next` => interrupted before that node
    return {"state": snap.values, "paused_before": list(snap.next), "is_paused": paused,
            "config": config}


def resume_with_human_decision(graph, thread_id, decision: str):
    """Supply the human's approve/reject and resume from the interrupt."""
    config = {"configurable": {"thread_id": thread_id}}
    graph.update_state(config, {"human_decision": decision})
    graph.invoke(None, config)               # resume
    return graph.get_state(config).values
