"""
Knowledge Base Builder
-----------------------
Reads crop_yield_dataset.csv and produces labelled chunks for ingestion
into the ChromaDB vector store.
"""
import csv
import os
import statistics
from typing import Iterator

DATASET_FILE = os.path.join(
    os.path.dirname(__file__), "..", "crop_yield_dataset.csv"
)

NUMERIC_COLS = [
    "Temperature (C)",
    "Rainfall (mm)",
    "Humidity (%)",
    "Sunlight (hours)",
    "Soil pH",
    "Soil Nitrogen (%)",
    "Soil Phosphorus (ppm)",
    "Soil Potassium (ppm)",
    "Altitude (m)",
    "Wind Speed (m/s)",
    "Crop Yield (tons/ha)",
]


def _fmt(value: float) -> str:
    return f"{value:.2f}"


def iter_chunks(
    chunk_size: int = 400,
    overlap: int = 50,
) -> Iterator[dict]:
    """
    Yield {'id': str, 'text': str, 'metadata': dict} dicts.

    Produces two kinds of chunks:
      1. Per-crop aggregate summary with min/mean/max statistics.
      2. Individual row chunks describing exact growing conditions and yield.
    """
    with open(DATASET_FILE, encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))

    by_crop: dict[str, list[dict]] = {}
    for row in rows:
        by_crop.setdefault(row["Crop"], []).append(row)

    doc_id = 0

    # Per-crop summary chunks
    for crop, crop_rows in sorted(by_crop.items()):
        stats = {
            col: {
                "mean": statistics.mean(float(r[col]) for r in crop_rows),
                "min":  min(float(r[col]) for r in crop_rows),
                "max":  max(float(r[col]) for r in crop_rows),
            }
            for col in NUMERIC_COLS
        }

        text = "\n".join([
            f"{crop} growing conditions and yield summary ({len(crop_rows)} records):",
            f"  Temperature: avg {_fmt(stats['Temperature (C)']['mean'])} °C"
            f" (range {_fmt(stats['Temperature (C)']['min'])}–{_fmt(stats['Temperature (C)']['max'])} °C)",
            f"  Rainfall: avg {_fmt(stats['Rainfall (mm)']['mean'])} mm"
            f" (range {_fmt(stats['Rainfall (mm)']['min'])}–{_fmt(stats['Rainfall (mm)']['max'])} mm)",
            f"  Humidity: avg {_fmt(stats['Humidity (%)']['mean'])} %"
            f" (range {_fmt(stats['Humidity (%)']['min'])}–{_fmt(stats['Humidity (%)']['max'])} %)",
            f"  Sunlight: avg {_fmt(stats['Sunlight (hours)']['mean'])} hrs"
            f" (range {_fmt(stats['Sunlight (hours)']['min'])}–{_fmt(stats['Sunlight (hours)']['max'])} hrs)",
            f"  Soil pH: avg {_fmt(stats['Soil pH']['mean'])}"
            f" (range {_fmt(stats['Soil pH']['min'])}–{_fmt(stats['Soil pH']['max'])})",
            f"  Soil Nitrogen: avg {_fmt(stats['Soil Nitrogen (%)']['mean'])} %"
            f" (range {_fmt(stats['Soil Nitrogen (%)']['min'])}–{_fmt(stats['Soil Nitrogen (%)']['max'])} %)",
            f"  Soil Phosphorus: avg {_fmt(stats['Soil Phosphorus (ppm)']['mean'])} ppm"
            f" (range {_fmt(stats['Soil Phosphorus (ppm)']['min'])}–{_fmt(stats['Soil Phosphorus (ppm)']['max'])} ppm)",
            f"  Soil Potassium: avg {_fmt(stats['Soil Potassium (ppm)']['mean'])} ppm"
            f" (range {_fmt(stats['Soil Potassium (ppm)']['min'])}–{_fmt(stats['Soil Potassium (ppm)']['max'])} ppm)",
            f"  Altitude: avg {_fmt(stats['Altitude (m)']['mean'])} m"
            f" (range {_fmt(stats['Altitude (m)']['min'])}–{_fmt(stats['Altitude (m)']['max'])} m)",
            f"  Wind Speed: avg {_fmt(stats['Wind Speed (m/s)']['mean'])} m/s"
            f" (range {_fmt(stats['Wind Speed (m/s)']['min'])}–{_fmt(stats['Wind Speed (m/s)']['max'])} m/s)",
            f"  Crop Yield: avg {_fmt(stats['Crop Yield (tons/ha)']['mean'])} tons/ha"
            f" (range {_fmt(stats['Crop Yield (tons/ha)']['min'])}–{_fmt(stats['Crop Yield (tons/ha)']['max'])} tons/ha)",
        ])

        yield {
            "id":       f"doc_{doc_id}",
            "text":     text,
            "metadata": {"crop": crop, "chunk_type": "summary"},
        }
        doc_id += 1
