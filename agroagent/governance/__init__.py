"""
AgroAgent Governance Layer
==========================
Wraps the AgroAgent multi-agent orchestration with PrivateVault's
`coordination/` layer (trust-weighted quorum, drift-aware voting, persistent
trust scoring, policy validation, and a hash-chained audit ledger).

Public entry points:
    from governance.governed_orchestrator import run_governed
    from governance.pv_coordination import CoordinationGovernor
"""
