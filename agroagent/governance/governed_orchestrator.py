"""
Governed orchestrator
======================
Runs AgroAgent's existing LangGraph pipeline UNCHANGED, then wraps its output
with PrivateVault's coordination governance (consensus / trust / validation /
approval / audit).

    from governance.governed_orchestrator import run_governed
    result = run_governed(soil, location="Mumbai")
    result["governance"]["final_status"]   # ALLOW | BLOCK
    result["governance"]["governed_crop"]  # surfaced crop, or "Withheld — needs review"
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agents.orchestrator import run as run_pipeline      # noqa: E402
from governance.pv_coordination import CoordinationGovernor  # noqa: E402

_governor: CoordinationGovernor | None = None


def _get_governor() -> CoordinationGovernor:
    global _governor
    if _governor is None:
        _governor = CoordinationGovernor()
    return _governor


def run_governed(soil_data: dict, location: str = "New Delhi",
                 user_query: str = "") -> dict:
    """
    Execute the full pipeline and apply coordination governance.

    Returns the raw pipeline state plus a `governance` block.
    """
    state = run_pipeline(soil_data=soil_data, location=location, user_query=user_query)
    governance = _get_governor().govern(state)
    return {
        "pipeline_state": state,
        "governance": governance,
        # convenience top-level fields
        "ungoverned_recommendation": governance["ungoverned_crop"],
        "governed_recommendation": governance["governed_crop"],
        "final_status": governance["final_status"],
    }
