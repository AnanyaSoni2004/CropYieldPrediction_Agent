# Phase 2 — PrivateVault Coordination Integration

**Objective:** integrate PrivateVault's [`coordination/`](https://github.com/LOLA0786/PrivateVault.ai/tree/main/coordination)
layer into AgroAgent and compare system behavior **before vs. after** PrivateVault.

The AgroAgent pipeline is wrapped — **not modified**. The existing LangGraph
orchestration runs exactly as in Phase 1; PrivateVault's coordination layer is
applied to its output as a governance boundary.

- Wrapper code: [governance/](../governance/)
- Run the comparison: `python governance/compare_before_after.py`
- Raw results: [governance/comparison_results.json](../governance/comparison_results.json)

---

## 1. What was integrated (from `coordination/`)

| Capability | PrivateVault `coordination/` component used | Role in AgroAgent |
|------------|---------------------------------------------|-------------------|
| **Trust scoring** | `coordination.trust.trust_engine.TrustEngine` | Persistent, decaying per-agent trust weight (JSON-backed), seeded into each run |
| **Agent consensus** | `coordination.mesh.drift_aware_quorum.DriftAwareQuorum` + `decision_engine.MeshDecisionEngine` + `weighted_consensus.compute_weighted_consensus` | Trust-weighted, drift-aware quorum over the agents' votes |
| **Trust registry** | `coordination.mesh.trust_registry.TrustRegistry` | Per-run weight lookup for the quorum |
| **Decision validation** | `CropPolicyEngine` (our crop-domain adapter, same `evaluate(agent_id, request)` contract as `coordination.mesh.agent_policy_engine.PolicyEngine`) | Each agent reasons under policy to cast APPROVE / REJECT |
| **Execution approval** | `demo_one_flow` pattern: `ALLOW iff consensus == APPROVE AND policy PASS` | Withholds a recommendation that fails the gate |
| **Audit logging** | SHA-256 hash-chained append-only JSONL ledger + `TrustEngine.snapshot_hash()` | Tamper-evident record of every governed decision |

> Two `coordination/` files are intentionally **not** used: `mesh/merkle_anchor.py`
> (imports a `merkle` module that is absent from the repo) and
> `trust/update_after_decision.py` (instantiates a `TrustEngine` at import time with a
> CWD-relative path). Their logic is reproduced safely inside
> [governance/pv_coordination.py](../governance/pv_coordination.py).

### Domain mapping

PrivateVault's coordination demos are finance-flavored (`pricing_agent`,
`risk_agent`, `revenue_agent` voting on discount amounts). AgroAgent's agents map
onto the same trust-weighted-quorum model as **four voters on the proposition
"surface the recommended crop to the farmer"**:

| AgroAgent signal | Voter | Votes APPROVE when |
|------------------|-------|--------------------|
| RandomForest confidence | `crop_agent` | ML confidence ≥ 30% (drift-ignored if < 15%) |
| Live weather suitability | `weather_agent` | suitability not "poor" (drift-ignored if weather is mock) |
| Market profitability | `market_agent` | recommended crop's profitability score ≥ 0.30 |
| LLM validation | `llm_agent` | validation = Valid and not blocked |

**System policy gate** (hard floor, analogous to the demo's `MAX_DISCOUNT`):
fails if the LLM blocked the input, or if ML confidence < 20%.

---

## 2. Architecture: before vs. after

```
BEFORE (Phase 1)
  soil + location ─► weather ─► crop(ML) ─► market ─► rag ─► decision(LLM) ─► recommended_crop ─► SURFACED

AFTER (Phase 2)
  soil + location ─► [ same pipeline, unchanged ] ─► recommended_crop
                                                          │
                                                          ▼
                          ┌──────────── PrivateVault coordination governor ───────────┐
                          │ 1. seed per-agent TRUST weights (persistent TrustEngine)   │
                          │ 2. each agent casts APPROVE/REJECT under CropPolicyEngine  │
                          │ 3. trust-weighted, drift-aware QUORUM → consensus          │
                          │ 4. system POLICY gate (confidence floor / validation)      │
                          │ 5. ALLOW iff consensus==APPROVE AND policy PASS            │
                          │ 6. update TRUST vs. outcome; append hash-chained AUDIT     │
                          └────────────────────────────────────────────────────────────┘
                                                          │
                                          ALLOW ─► crop SURFACED   /   BLOCK ─► WITHHELD (needs review)
```

---

## 3. Before / after results (live run, 5 representative inputs)

| Case | BEFORE (ungoverned surfaces) | Consensus (score) | Policy | AFTER verdict | Outcome changed |
|------|------------------------------|-------------------|--------|---------------|-----------------|
| valid_humid_coastal (Mumbai) | jute | APPROVE (0.75) | PASS | **ALLOW → jute** | no |
| valid_high_npk_neutral (Bengaluru) | banana | APPROVE (0.75) | PASS | **ALLOW → banana** | no |
| suboptimal_drought (Jaipur) | Invalid Input | REJECT (0.50) | FAIL | **BLOCK → withheld** | **yes** |
| invalid_extreme_ph_low_npk (Delhi) | Invalid Input | REJECT (0.00) | FAIL | **BLOCK → withheld** | **yes** |
| valid_with_user_query (Hyderabad) | maize | APPROVE (1.00) | PASS | **ALLOW → maize** | no |

- **Governed BLOCK verdicts: 2/5.** The ungoverned pipeline surfaces a result for
  all 5; governance independently withholds the 2 unsafe ones.
- **Audit-chain integrity: VERIFIED ✓** — and tamper-tested: editing any ledger
  entry flips verification to BROKEN ✗ (confirmed live).

### What governance adds that the bare pipeline does not

1. **Independent multi-agent consensus, not LLM-only trust.** In
   `valid_humid_coastal` and `valid_high_npk_neutral` the `crop_agent` *rejected*
   (ML confidence 28%) yet consensus still APPROVEd on the trust-weighted strength of
   the weather/market/LLM agents — the disagreement is resolved by quorum rather than
   silently ignored. In `suboptimal_drought` the market + LLM agents *outvoted* the
   crop + weather agents to REJECT.
2. **A confidence floor the LLM cannot override.** The system policy fails any
   recommendation below 20% ML confidence even if the LLM declared it "Valid" — a
   guardrail the Phase-1 system lacks (Phase 1 reported "High" confidence on 28–48%
   ML predictions).
3. **Drift-aware voting.** A mock/simulated weather reading, or near-random ML
   confidence (< 15%), is flagged so the quorum ignores that agent's APPROVE vote
   instead of counting an unreliable signal.
4. **Evolving accountability.** Each agent's trust weight is rewarded/penalised
   against the consensus outcome and persists across runs (visible decay toward the
   engine's 0.7 equilibrium across the 5 cases).
5. **Tamper-evident audit trail.** Every decision is hash-chained
   (`entry_hash = SHA256(prev_hash + payload)`) — Phase 1 had no durable decision record.

### Behavior that did **not** change

For the three valid cases the governed system surfaces the **same crop** as before
(jute, banana, maize). Governance is a safety boundary, not a recommender — it never
substitutes a different crop, only ALLOWs or WITHHOLDs.

---

## 4. Metrics: before vs. after

| Dimension | Before (Phase 1) | After (Phase 2, governed) |
|-----------|------------------|----------------------------|
| Decision authority | ML top crop, LLM may block | + trust-weighted quorum of 4 agents + policy gate |
| Unsafe outputs surfaced | all (no gate) | withheld on consensus REJECT or policy FAIL (2/5 here) |
| Confidence guardrail | none (LLM self-reported) | hard 20% ML floor, LLM cannot override |
| Agent disagreement handling | logged only (Phase-1 finding: 100% ML≠market) | resolved via weighted quorum |
| Trust / accountability | none | persistent per-agent trust, decaying, outcome-driven |
| Audit trail | none | SHA-256 hash-chained, verifiable, tamper-evident |
| Added latency | — | negligible (~5 ms; all coordination logic is local/offline) |

End-to-end latency is essentially unchanged: the coordination layer is pure local
computation (no network, no API keys), adding single-digit milliseconds on top of the
~1–2 s pipeline.

---

## 5. Files

| File | Purpose |
|------|---------|
| [governance/pv_coordination.py](../governance/pv_coordination.py) | Bridge to `coordination/`; `CropPolicyEngine`, vote derivation, `CoordinationGovernor`, audit chain + verifier |
| [governance/governed_orchestrator.py](../governance/governed_orchestrator.py) | `run_governed()` — runs the unchanged pipeline, then governs its output |
| [governance/compare_before_after.py](../governance/compare_before_after.py) | Before/after comparison harness |
| [governance/state/trust_state.json](../governance/state/) | Persistent per-agent trust weights (PrivateVault `TrustEngine`) |
| [governance/state/audit_ledger.jsonl](../governance/state/) | Hash-chained, append-only audit ledger |

### Notes / limitations

- The governed BLOCK set currently coincides with the LLM's `Invalid Input` cases,
  because none of the 5 sample inputs land in the 15–20% ML-confidence band where the
  policy floor would block a *validated* recommendation. The mechanism is independent
  of the LLM (point 2 above); it simply wasn't the binding constraint on these inputs.
- `CropPolicyEngine` thresholds (`ML_CONF_VOTE_FLOOR=0.30`, `ML_CONF_HARD_FLOOR=0.20`,
  `MARKET_SCORE_FLOOR=0.30`) are explicit, tunable governance policy in
  [pv_coordination.py](../governance/pv_coordination.py).
- Trust weights persist in `governance/state/`; delete that file to reset to the
  neutral 1.0 baseline before a fresh demonstration.
