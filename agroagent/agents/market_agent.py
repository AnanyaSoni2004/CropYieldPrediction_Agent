"""
Market Agent
------------
Reads market price data and ranks candidate crops by profitability.
"""
import os
from typing import Any

import pandas as pd

MARKET_CSV = os.path.join(os.path.dirname(__file__), "..", "data", "market_prices.csv")

# Numeric score for demand levels (used in profitability ranking)
DEMAND_SCORE = {"high": 3, "medium": 2, "low": 1}
TREND_SCORE  = {"rising": 2, "stable": 1, "falling": 0}


class MarketAgent:
    """Scores crops by current market prices, demand, and price trend."""

    def __init__(self):
        self._df = pd.read_csv(MARKET_CSV)

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def analyse(self, candidate_crops: list[str]) -> dict[str, Any]:
        """
        Return market analysis for *candidate_crops*.

        Parameters
        ----------
        candidate_crops : list of crop names to evaluate

        Returns
        -------
        {
            "best_market_crop":  str,
            "ranked_crops":      [{"crop": str, "price_per_quintal": int,
                                   "demand": str, "trend": str,
                                   "profitability_score": float}, ...],
            "market_insights":   str,
        }
        """
        rows = []
        for crop in candidate_crops:
            match = self._df[self._df["crop"].str.lower() == crop.lower()]
            if match.empty:
                # Use average price for unknown crops
                avg_price = int(self._df["price_per_quintal"].mean())
                rows.append(
                    {
                        "crop": crop,
                        "price_per_quintal": avg_price,
                        "demand": "medium",
                        "trend": "stable",
                        "profitability_score": self._score(avg_price, "medium", "stable"),
                    }
                )
            else:
                r = match.iloc[0]
                rows.append(
                    {
                        "crop": crop,
                        "price_per_quintal": int(r["price_per_quintal"]),
                        "demand": r["demand_level"],
                        "trend":  r["trend"],
                        "profitability_score": self._score(
                            r["price_per_quintal"], r["demand_level"], r["trend"]
                        ),
                    }
                )

        ranked = sorted(rows, key=lambda x: x["profitability_score"], reverse=True)
        best   = ranked[0]["crop"] if ranked else "unknown"

        return {
            "best_market_crop": best,
            "ranked_crops":     ranked,
            "market_insights":  self._build_insights(ranked),
        }

    # ------------------------------------------------------------------
    # private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _score(price: float, demand: str, trend: str) -> float:
        """Compute a normalised profitability score [0, 1]."""
        max_price = 15_000
        price_norm  = min(price / max_price, 1.0)
        demand_norm = DEMAND_SCORE.get(demand, 1) / 3
        trend_norm  = TREND_SCORE.get(trend,  1) / 2
        return round(0.5 * price_norm + 0.3 * demand_norm + 0.2 * trend_norm, 4)

    @staticmethod
    def _build_insights(ranked: list[dict]) -> str:
        if not ranked:
            return "No market data available."
        best = ranked[0]
        lines = [
            f"{best['crop'].title()} leads in profitability "
            f"(₹{best['price_per_quintal']}/quintal, "
            f"{best['demand']} demand, {best['trend']} trend)."
        ]
        if len(ranked) > 1:
            runner = ranked[1]
            lines.append(
                f"{runner['crop'].title()} is a strong alternative "
                f"(₹{runner['price_per_quintal']}/quintal, {runner['trend']} trend)."
            )
        return " ".join(lines)
