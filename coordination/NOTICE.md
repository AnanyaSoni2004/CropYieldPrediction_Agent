# Vendored from PrivateVault.ai

These modules are a minimal subset of PrivateVault's `coordination/` layer,
vendored into this repo so the deployed app (Streamlit Cloud) has them without the
full PrivateVault.ai checkout. Source: https://github.com/LOLA0786/PrivateVault.ai
License: Apache-2.0. Only the files used by `governance/pv_coordination.py` are included:
drift_aware_quorum, trust_registry, decision_engine, weighted_consensus (mesh) and
trust_engine (trust).
