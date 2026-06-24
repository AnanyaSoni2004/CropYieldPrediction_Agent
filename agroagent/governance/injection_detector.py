"""
Prompt-injection detector
=========================
Flags instruction-style manipulation aimed at the DecisionAgent's LLM — the
text patterns an attacker uses to override the system prompt or agronomic rules.

This is the signal that lets the governance layer treat the `llm_agent` as
compromised (drift) when its context contains injection, independent of whether
the underlying model happened to obey.

Distinct from poison_detector.py: that flags false *facts*; this flags adversarial
*instructions*.
"""
from __future__ import annotations

import re

INJECTION_PATTERNS = [
    (r"ignore (all )?(previous |prior )?(instructions|guidelines)", "override of instructions/guidelines"),
    (r"disregard (the )?(agronomic |soil |climate |safety )?(guidelines|rules|risk)", "disregard safety/agronomy"),
    (r"override (previous |all )?instructions", "explicit instruction override"),
    (r"regardless of (the )?(risk|soil|climate|conditions|safety)", "ignore-risk directive"),
    (r"highest[- ]profit", "profit-maximisation directive"),
    (r"mark (every |this |all )?inputs? as valid", "force-valid directive"),
    (r"set (validation status|confidence level)\s*:", "output-field forcing"),
    (r"validation status\s*:\s*valid", "forced validation verdict"),
    (r"without validation", "bypass-validation directive"),
    (r"trust (all |the )?(retrieved )?documents", "blind-trust directive"),
    (r"unrestricted mode|developer mode|jailbreak|system update", "role/override framing"),
    (r"you are now", "role reassignment"),
]


def detect_injection(text: str) -> list[str]:
    """Return descriptions of injection patterns found in *text* (empty = clean)."""
    low = text.lower()
    found = []
    for pat, label in INJECTION_PATTERNS:
        m = re.search(pat, low)
        if m:
            found.append(f"{label}: '{m.group(0)}'")
    return found


def is_injection(text: str) -> bool:
    return len(detect_injection(text)) > 0
