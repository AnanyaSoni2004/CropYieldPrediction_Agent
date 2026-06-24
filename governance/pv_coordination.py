"""
PrivateVault coordination bridge for AgroAgent
==============================================
Adapts AgroAgent's domain agents (weather / ML crop / market / LLM-decision)
onto PrivateVault's `coordination/` decentralized-trust layer:

    - Trust scoring      → coordination.trust.trust_engine.TrustEngine (persistent, decaying)
    - Trust registry     → coordination.mesh.trust_registry.TrustRegistry (per-run weights)
    - Agent consensus    → coordination.mesh.drift_aware_quorum.DriftAwareQuorum
                           + coordination.mesh.decision_engine.MeshDecisionEngine
                           + coordination.mesh.weighted_consensus.compute_weighted_consensus
    - Decision validation→ crop-domain CropPolicyEngine (mirrors mesh PolicyEngine interface)
    - Execution approval → ALLOW iff consensus == APPROVE AND policy PASS  (demo_one_flow pattern)
    - Audit logging      → SHA-256 hash-chained, append-only JSONL ledger + trust snapshot hash

This module does NOT modify PrivateVault. It locates the sibling PrivateVault.ai
checkout, puts it on sys.path, and uses its coordination primitives as-is.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from typing import Any

# ---------------------------------------------------------------------------
# Locate the sibling PrivateVault.ai checkout and import its coordination layer
# ---------------------------------------------------------------------------

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_PV_ROOT = os.path.join(_REPO_ROOT, "PrivateVault.ai")
if _PV_ROOT not in sys.path:
    sys.path.insert(0, _PV_ROOT)

from coordination.mesh.drift_aware_quorum import DriftAwareQuorum     # noqa: E402
from coordination.mesh.trust_registry import TrustRegistry            # noqa: E402
from coordination.mesh.decision_engine import MeshDecisionEngine      # noqa: E402
from coordination.mesh.weighted_consensus import compute_weighted_consensus  # noqa: E402
from coordination.trust.trust_engine import TrustEngine               # noqa: E402

from governance.injection_detector import detect_injection            # noqa: E402
from governance.poison_detector import detect_poison                  # noqa: E402

# Local, writable state (trust ledger + audit chain) — kept inside agroagent.
_STATE_DIR = os.path.join(os.path.dirname(__file__), "state")
os.makedirs(_STATE_DIR, exist_ok=True)
TRUST_STATE_PATH = os.path.join(_STATE_DIR, "trust_state.json")
AUDIT_LEDGER_PATH = os.path.join(_STATE_DIR, "audit_ledger.jsonl")

# The four AgroAgent voters on the proposition:
#   "APPROVE surfacing the recommended crop to the farmer."
VOTERS = ["crop_agent", "weather_agent", "market_agent", "llm_agent"]

# Thresholds (tunable governance policy)
ML_CONF_VOTE_FLOOR = 0.30   # crop_agent APPROVEs only above this ML confidence
ML_CONF_DRIFT_FLOOR = 0.15  # below this, crop_agent's vote is treated as drift (ignored)
ML_CONF_HARD_FLOOR = 0.20   # policy fails outright below this
MARKET_SCORE_FLOOR = 0.30   # market_agent APPROVEs only above this profitability score

BLOCK_TOKENS = {"invalid input", "no suitable crop"}


# ---------------------------------------------------------------------------
# Crop-domain policy engine (same (agent_id, request) -> (decision, reason)
# contract as coordination.mesh.agent_policy_engine.PolicyEngine, but for crops)
# ---------------------------------------------------------------------------

class CropPolicyEngine:
    """Per-agent reasoning over a crop-recommendation request."""

    def evaluate(self, agent_id: str, request: dict) -> tuple[str, str]:
        conf = request["ml_confidence"]
        suitability = request["weather_suitability"]
        market_score = request["market_score"]
        validation = request["validation_status"]
        blocked = request["llm_blocked"]

        if agent_id == "crop_agent":
            if conf >= ML_CONF_VOTE_FLOOR:
                return "APPROVE", f"ML confidence {conf:.0%} ≥ {ML_CONF_VOTE_FLOOR:.0%} floor"
            return "REJECT", f"ML confidence {conf:.0%} below {ML_CONF_VOTE_FLOOR:.0%} floor"

        if agent_id == "weather_agent":
            if suitability.startswith("poor"):
                return "REJECT", f"weather unsuitable: {suitability}"
            return "APPROVE", f"weather acceptable: {suitability}"

        if agent_id == "market_agent":
            if market_score >= MARKET_SCORE_FLOOR:
                return "APPROVE", f"profitability score {market_score:.2f} ≥ {MARKET_SCORE_FLOOR:.2f}"
            return "REJECT", f"profitability score {market_score:.2f} below {MARKET_SCORE_FLOOR:.2f}"

        if agent_id == "llm_agent":
            if blocked or validation.lower() == "invalid":
                return "REJECT", f"LLM validation = {validation or 'Invalid'} (recommendation blocked)"
            return "APPROVE", f"LLM validation = {validation or 'Valid'}"

        return "REJECT", "unknown agent"


# ---------------------------------------------------------------------------
# Vote derivation: turn an AgroAgent pipeline state into voting inputs
# ---------------------------------------------------------------------------

def _parse_validation_status(llm_response: str) -> str:
    m = re.search(r"validation status\s*:\s*(\w+)", llm_response, re.IGNORECASE)
    return m.group(1).strip() if m else ""


def build_request(state: dict) -> dict:
    """Extract the signals each voter reasons over, from a pipeline result."""
    crop = state["crop_result"]
    weather = state["weather_result"]
    market = state["market_result"]
    final = state["final_recommendation"]

    recommended = final["recommended_crop"]
    blocked = recommended.strip().lower() in BLOCK_TOKENS

    # market profitability score for the recommended crop (0 if blocked / unranked)
    market_score = 0.0
    if not blocked:
        for r in market["ranked_crops"]:
            if r["crop"].lower() == recommended.lower():
                market_score = float(r["profitability_score"])
                break

    return {
        "recommended_crop": recommended,
        "ml_top_crop": crop["top_prediction"],
        "ml_confidence": float(crop["confidence"]),
        "weather_suitability": weather.get("suitability", ""),
        "weather_source": weather.get("source", "mock"),
        "market_best_crop": market["best_market_crop"],
        "market_score": market_score,
        "validation_status": _parse_validation_status(final.get("llm_response", "")),
        "llm_blocked": blocked,
    }


def _drift_context(agent_id: str, request: dict) -> dict:
    """
    Drift-aware voting: an agent whose signal is unreliable is flagged so the
    DriftAwareQuorum ignores its APPROVE vote.
      - weather_agent drifts when weather is simulated (no live API)
      - crop_agent drifts when ML confidence is near-random
    """
    if agent_id == "weather_agent" and request["weather_source"] != "openweathermap":
        return {"drift": True, "reason": "weather is mock/simulated"}
    if agent_id == "crop_agent" and request["ml_confidence"] < ML_CONF_DRIFT_FLOOR:
        return {"drift": True, "reason": "ML confidence near-random"}
    return {"drift": False}


# ---------------------------------------------------------------------------
# The Governor: consensus + trust + validation + approval + audit
# ---------------------------------------------------------------------------

class CoordinationGovernor:
    """Applies PrivateVault coordination governance to one AgroAgent result."""

    def __init__(self, trust_state_path: str = TRUST_STATE_PATH,
                 audit_path: str = AUDIT_LEDGER_PATH):
        self._trust = TrustEngine(storage_path=trust_state_path)
        self._policy = CropPolicyEngine()
        self._audit_path = audit_path

    # -- public ---------------------------------------------------------

    def govern(self, state: dict) -> dict:
        request = build_request(state)
        action_id = "rec_" + hashlib.sha256(
            json.dumps(request, sort_keys=True).encode()
        ).hexdigest()[:12]

        # 1. TRUST SCORING — seed a per-run registry from persistent weights
        registry = TrustRegistry()
        trust_in = {a: float(self._trust.get_weight(a)) for a in VOTERS}
        for a, w in trust_in.items():
            registry.set_score(a, w)

        # CONTENT-THREAT SCAN — the knowledge context that feeds the LLM may be
        # poisoned (false facts) or injected (adversarial instructions). If so the
        # llm_agent is treated as COMPROMISED: its vote is quarantined (drift) and
        # the policy gate hard-fails, because its validation can no longer be trusted.
        rag_ctx = state.get("rag_context", "") or ""
        context_flags = detect_injection(rag_ctx) + detect_poison(rag_ctx)
        context_compromised = bool(context_flags)

        # 2. DECISION VALIDATION — each agent reasons under policy
        votes: list[dict] = []
        for agent in VOTERS:
            decision, reason = self._policy.evaluate(agent, request)
            ctx = _drift_context(agent, request)
            if agent == "llm_agent" and context_compromised:
                ctx = {"drift": True,
                       "reason": f"context compromised: {len(context_flags)} injection/poison flag(s)"}
            votes.append({"agent_id": agent, "decision": decision,
                          "reason": reason, "context": ctx})

        # 3. AGENT CONSENSUS — trust-weighted, drift-aware quorum
        active_trust = sum(
            trust_in[v["agent_id"]] for v in votes if not v["context"].get("drift")
        )
        threshold = round(0.5 * active_trust, 4)   # majority of *active* trust weight
        quorum = DriftAwareQuorum(threshold=threshold, trust_registry=registry)
        for v in votes:
            quorum.submit_vote(action_id, v["agent_id"], v["decision"], "sig", context=v["context"])
        consensus = MeshDecisionEngine(quorum).evaluate(action_id)["decision"]

        wc = compute_weighted_consensus([
            {"agent_id": v["agent_id"],
             "vote": v["decision"] == "APPROVE" and not v["context"].get("drift"),
             "weight": trust_in[v["agent_id"]]}
            for v in votes
        ])

        # 4. EXECUTION APPROVAL — system-level policy gate (demo_one_flow pattern)
        policy_pass, policy_reason = self._system_policy(request)
        if context_compromised:
            policy_pass = False
            policy_reason = (f"compromised context — {len(context_flags)} injection/poison "
                             f"flag(s); LLM validation untrusted")
        final_status = "ALLOW" if (consensus == "APPROVE" and policy_pass) else "BLOCK"

        # 5. TRUST UPDATE — agents rewarded/penalised vs. the consensus outcome
        trust_out = {}
        for v in votes:
            correct = (v["decision"] == consensus)
            outcome = {"correct": correct,
                       "policy_violation": (not policy_pass and v["decision"] == "APPROVE")}
            trust_out[v["agent_id"]] = self._trust.update_trust(v["agent_id"], outcome)

        # Supporting evidence (for audit / explainability)
        weather = state.get("weather_result", {})
        m_conf = re.search(r"confidence level\s*:\s*(\w+)",
                           state["final_recommendation"].get("llm_response", ""), re.IGNORECASE)
        evidence = {
            "ml_top_crops": [(c["crop"], c["confidence"])
                             for c in state["crop_result"]["top_crops"]],
            "ml_confidence": request["ml_confidence"],
            "llm_confidence_level": m_conf.group(1) if m_conf else None,
            "llm_model": state["final_recommendation"].get("model_used"),
            "validation_status": request["validation_status"] or "Valid",
            "weather": {"source": weather.get("source"),
                        "suitability": weather.get("suitability"),
                        "temperature": weather.get("temperature"),
                        "humidity": weather.get("humidity")},
            "market": {"best_crop": request["market_best_crop"],
                       "score": request["market_score"]},
            "rag_excerpt": (state.get("rag_context", "") or "")[:240].replace("\n", " "),
            "context_threat_flags": context_flags,
        }

        # 6. AUDIT LOGGING — hash-chained, append-only ledger entry
        audit = self._append_audit({
            "action_id": action_id,
            "recommended_crop": request["recommended_crop"],
            "votes": votes,
            "consensus": consensus,
            "weighted_consensus": wc,
            "threshold": threshold,
            "policy_pass": policy_pass,
            "policy_reason": policy_reason,
            "final_status": final_status,
            "trust_in": trust_in,
            "trust_out": trust_out,
            "trust_snapshot_hash": self._trust.snapshot_hash(),
            "evidence": evidence,
        })

        # governed crop: surfaced only when ALLOWed
        governed_crop = request["recommended_crop"] if final_status == "ALLOW" else "Withheld — needs review"

        return {
            "action_id": action_id,
            "request": request,
            "votes": votes,
            "consensus": consensus,
            "consensus_score": round(wc.get("score", 0.0), 4),
            "threshold": threshold,
            "active_trust": round(active_trust, 4),
            "policy_pass": policy_pass,
            "policy_reason": policy_reason,
            "context_compromised": context_compromised,
            "context_flags": context_flags,
            "final_status": final_status,
            "governed_crop": governed_crop,
            "ungoverned_crop": request["recommended_crop"],
            "trust_in": trust_in,
            "trust_out": trust_out,
            "evidence": evidence,
            "audit": audit,
        }

    # -- private --------------------------------------------------------

    @staticmethod
    def _system_policy(request: dict) -> tuple[bool, str]:
        """Hard system gate, analogous to the MAX_DISCOUNT check in demo_one_flow."""
        if request["llm_blocked"]:
            return False, f"recommendation blocked by validation ({request['recommended_crop']})"
        if request["ml_confidence"] < ML_CONF_HARD_FLOOR:
            return False, f"ML confidence {request['ml_confidence']:.0%} below hard floor {ML_CONF_HARD_FLOOR:.0%}"
        return True, "within policy limits"

    def _append_audit(self, payload: dict) -> dict:
        """Append a tamper-evident, hash-chained record to the JSONL ledger."""
        prev_hash = self._last_hash()
        ts = datetime.now(timezone.utc).isoformat()
        body = json.dumps({**payload, "timestamp": ts, "prev_hash": prev_hash},
                          sort_keys=True)
        entry_hash = hashlib.sha256((prev_hash + body).encode()).hexdigest()
        record = {"timestamp": ts, "prev_hash": prev_hash,
                  "entry_hash": entry_hash, "payload": payload}
        with open(self._audit_path, "a") as f:
            f.write(json.dumps(record) + "\n")
        return {"entry_hash": entry_hash, "prev_hash": prev_hash, "timestamp": ts}

    def _last_hash(self) -> str:
        if not os.path.exists(self._audit_path):
            return "GENESIS"
        last = "GENESIS"
        with open(self._audit_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        last = json.loads(line)["entry_hash"]
                    except Exception:
                        pass
        return last


    # -- crop-level conflict resolution --------------------------------

    def govern_conflict(self, nominations: dict[str, str],
                        quorum_frac: float = 0.5) -> dict:
        """
        Resolve a conflict where agents nominate DIFFERENT crops, using
        PrivateVault's trust-weighted consensus.

        nominations : {agent_id: nominated_crop}
        Returns a resolution record (consensus crop or ESCALATE) and audits it.
        """
        trust_in = {a: float(self._trust.get_weight(a)) for a in nominations}
        resolution = resolve_crop_conflict(nominations, trust_in, quorum_frac)

        # trust update: agents aligned with the winning crop are "correct"
        winner = resolution["winner"]
        trust_out = {}
        for agent, crop in nominations.items():
            correct = (winner is not None and crop == winner)
            trust_out[agent] = self._trust.update_trust(
                agent, {"correct": correct, "policy_violation": False})

        audit = self._append_audit({
            "mode": "conflict_resolution",
            "nominations": nominations,
            "trust_in": trust_in,
            "resolution": resolution,
            "trust_out": trust_out,
            "trust_snapshot_hash": self._trust.snapshot_hash(),
        })
        return {**resolution, "trust_in": trust_in, "trust_out": trust_out, "audit": audit}


def resolve_crop_conflict(nominations: dict[str, str], trust_in: dict[str, float],
                          quorum_frac: float = 0.5) -> dict:
    """
    Trust-weighted plurality over nominated crops, via PrivateVault's
    coordination.mesh.weighted_consensus.compute_weighted_consensus.

    For each candidate crop, agents nominating it vote True (weighted by trust);
    the crop with the most trust-weight wins IF it clears the quorum fraction of
    total trust — otherwise the conflict is ESCALATEd (no safe consensus).
    """
    candidates = sorted(set(nominations.values()))
    per_crop = {}
    for c in candidates:
        wc = compute_weighted_consensus([
            {"agent_id": a, "vote": (nom == c), "weight": float(trust_in.get(a, 1.0))}
            for a, nom in nominations.items()
        ])
        per_crop[c] = {
            "support_weight": round(wc["score"] * wc["total_weight"], 4),
            "support_frac":   round(wc["score"], 4),
            "backers":        [a for a, nom in nominations.items() if nom == c],
        }
    winner = max(per_crop, key=lambda c: per_crop[c]["support_weight"])
    win_frac = per_crop[winner]["support_frac"]
    resolved = win_frac >= quorum_frac
    return {
        "candidates":     candidates,
        "per_crop":       per_crop,
        "winner":         winner if resolved else None,
        "consensus_frac": win_frac,
        "quorum_frac":    quorum_frac,
        "status":         "RESOLVED" if resolved else "ESCALATE",
    }


def verify_audit_chain(audit_path: str = AUDIT_LEDGER_PATH) -> bool:
    """Recompute the hash chain and confirm it is unbroken (tamper-evidence)."""
    if not os.path.exists(audit_path):
        return True
    prev = "GENESIS"
    with open(audit_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            body = json.dumps({**rec["payload"], "timestamp": rec["timestamp"],
                               "prev_hash": prev}, sort_keys=True)
            expected = hashlib.sha256((prev + body).encode()).hexdigest()
            if expected != rec["entry_hash"] or rec["prev_hash"] != prev:
                return False
            prev = rec["entry_hash"]
    return True
