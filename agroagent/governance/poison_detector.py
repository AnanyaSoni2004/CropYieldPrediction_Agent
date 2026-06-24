"""
Content-anomaly / poison detector for RAG passages
==================================================
PrivateVault's coordination layer scores *agents*, not documents — so to let
consensus "identify suspicious inputs" we need a signal that flags an agent
carrying anomalous retrieved context. This module provides that signal.

`detect_poison(text)` returns a list of agronomy-grounded red flags. Any flag
marks the knowledge-base agent's context as untrustworthy → the drift-aware
quorum then ignores its vote and trust scoring penalises it.

The rules are deliberately explainable (not ML): implausible nutrient doses,
hazardous/banned substances, physically impossible yields, and unsafe blanket
advice — the classic shapes of poisoned agricultural guidance.
"""
from __future__ import annotations

import re

# Agronomic safe-bounds and banned terms (intentionally conservative).
MAX_SAFE_DOSE_KG_HA = 300       # single-application N/P/K beyond this is implausible
MAX_PLAUSIBLE_YIELD_Q_HA = 150  # quintals/ha; real cereal yields are well under this
BANNED_SUBSTANCES = [
    "ddt", "endosulfan", "bleach", "diesel", "mercury", "arsenic",
    "lindane", "parathion", "monocrotophos",
]
UNSAFE_PHRASES = [
    r"no protective equipment",
    r"zero irrigation", r"no water", r"stop (?:all )?irrigation",
    r"double the (?:recommended )?dose", r"triple your (?:harvest|yield)",
    r"unlimited", r"without any fertilizer",
]


def detect_poison(text: str) -> list[str]:
    """Return a list of red-flag descriptions found in *text* (empty = clean)."""
    flags: list[str] = []
    low = text.lower()

    # 1. Implausible fertilizer / nutrient doses
    for m in re.finditer(r"(\d{2,5})\s*kg\s*/?\s*ha", low):
        dose = int(m.group(1))
        if dose > MAX_SAFE_DOSE_KG_HA:
            flags.append(f"implausible nutrient dose: {dose} kg/ha "
                         f"(safe ≤ {MAX_SAFE_DOSE_KG_HA})")

    # 2. Hazardous / banned substances
    for sub in BANNED_SUBSTANCES:
        if re.search(rf"\b{re.escape(sub)}\b", low):
            flags.append(f"hazardous/banned substance referenced: '{sub}'")

    # 3. Physically impossible yield claims
    for m in re.finditer(r"(\d{2,5})\s*quintal", low):
        y = int(m.group(1))
        if y > MAX_PLAUSIBLE_YIELD_Q_HA:
            flags.append(f"implausible yield claim: {y} quintals/ha "
                         f"(plausible ≤ {MAX_PLAUSIBLE_YIELD_Q_HA})")

    # 4. Unsafe blanket advice
    for pat in UNSAFE_PHRASES:
        if re.search(pat, low):
            flags.append(f"unsafe advice pattern: '{re.search(pat, low).group(0)}'")

    return flags


def is_suspicious(text: str) -> bool:
    return len(detect_poison(text)) > 0
