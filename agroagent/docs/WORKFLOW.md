# AgroAgent — Multi-Agent Workflow Documentation & Measurement

This document describes the existing multi-agent crop-recommendation workflow and
reports measurements taken by running the live pipeline (Groq, OpenWeatherMap, and
ChromaDB all active) via [evaluation/measure_workflow.py](../evaluation/measure_workflow.py).

All numbers below come from a real run on 5 representative inputs. Re-run with:

```bash
python evaluation/measure_workflow.py
# → prints summary, writes evaluation/measurement_results.json
```

---

## 1. Workflow topology

The system is a **fixed, linear LangGraph pipeline** (no branching, no loops, no
conditional edges). It is defined in [agents/orchestrator.py](../agents/orchestrator.py):

```
START
  └─► weather          (WeatherAgent)         → live temp/humidity + suitability label
        └─► crop_prediction  (CropPredictionAgent)  → top-3 crops + confidence (RandomForest)
              └─► market      (MarketAgent)          → profitability ranking of those crops
                    └─► rag    (RAGKnowledgeAgent)    → cultivation knowledge for top crop
                          └─► decision (DecisionAgent) → LLM synthesises final recommendation
                                └─► END
```

Each node is a pure function `AgentState → dict` that reads from and writes to a shared
`AgentState` TypedDict. The graph is compiled once into a module-level singleton and
invoked through `run(soil_data, location, user_query)`.

Entry points: `POST /recommend` in [api/main.py](../api/main.py) and the Streamlit UI
in [streamlit_app.py](../streamlit_app.py).

### Important data-flow detail: live weather overrides user input

In `crop_prediction_node`, **if a live OpenWeatherMap response is available, the user's
submitted `temperature` and `humidity` are discarded and replaced with live values for
the location** before ML inference and before LLM validation:

```python
if weather.get("source") == "openweathermap":
    soil["temperature"] = weather["temperature"]
    soil["humidity"]    = weather["humidity"]
```

This was clearly visible in measurement — e.g. the "Jaipur drought" case submitted
`temperature=42, humidity=25` but the model actually saw `34.19°C, 37%` (live Jaipur
weather). The submitted `N, P, K, ph, rainfall` are **not** overridden.

---

## 2. The five agents (decisions & tool calls)

| Agent | Tool / resource used | Decision it commits to |
|-------|---------------------|------------------------|
| **WeatherAgent** | OpenWeatherMap HTTP GET (mock fallback if no key) | live temperature, humidity, and a suitability label (`excellent`/`good`/`poor – …`) |
| **CropPredictionAgent** | RandomForest `predict_proba` on 7 features (N,P,K,temp,humidity,ph,rainfall) | top-1 crop + top-3 crops with confidence scores |
| **MarketAgent** | reads `data/market_prices.csv` (pandas) | profitability ranking of the candidate crops; `best_market_crop` |
| **RAGKnowledgeAgent** | ChromaDB vector search + Groq LLM (extractive fallback) | retrieved cultivation-knowledge context (and an answer to a user query, if any) |
| **DecisionAgent** | Groq `llama-3.3-70b-versatile` (→ OpenAI → rule-based fallback) | final structured recommendation: validation status, environmental assessment, recommended crop, confidence level |

### Decision authority

The DecisionAgent's LLM is **not** free to pick a crop. Per
[agents/decision_agent.py](../agents/decision_agent.py):

- The **ML top prediction is authoritative** for the recommended crop on valid input.
- The LLM may only *override* it to `Invalid Input` or `No Suitable Crop`.
- Otherwise the LLM's role is to *explain and caveat*, not to substitute its own choice
  (`recommended = crop_result["top_prediction"]`).

So the final crop is always either the ML top crop **or** a block verdict.

---

## 3. Measurements

### 3.1 Agent decisions (live run)

| Case | Weather (live) | ML top (conf.) | Market best | Final | Validation | Reported confidence |
|------|----------------|----------------|-------------|-------|-----------|---------------------|
| valid_humid_coastal (Mumbai) | excellent, 28.0°C | **jute** (0.475) | coffee | jute | Valid | High |
| valid_high_npk_neutral (Bengaluru) | excellent, 29.1°C | **banana** (0.275) | maize | banana | Valid | High |
| suboptimal_drought (Jaipur) | good, 34.2°C | mango (0.515) | coffee | **Invalid Input** | Invalid | Low |
| invalid_extreme_ph_low_npk (Delhi) | poor, 39.9°C | mothbeans (0.28) | kidneybeans | **Invalid Input** | Invalid | Low |
| valid_with_user_query (Hyderabad) | excellent, 28.4°C | **maize** (0.36) | coffee | maize | Valid | Medium |

### 3.2 Agent disagreements

Disagreement is measured along four axes. Counts across the 5 cases:

| Disagreement type | Cases | Frequency |
|-------------------|-------|-----------|
| **ML top ≠ Market best crop** | all 5 | **5/5 (100%)** |
| **LLM blocked ML** (`Invalid Input`) | drought, invalid | 2/5 |
| **Final ≠ ML** on valid input | none | 0/5 |
| **Weather "poor" but still recommended** | none | 0/5 |

**Key finding — the Market agent has no influence on the chosen crop.** In every case
the market's most-profitable crop differed from the ML pick (e.g. ML `jute` vs market
`coffee`), yet the final crop always followed ML. The MarketAgent output is surfaced as
advisory text inside the LLM prompt but never changes the recommended crop. This is by
design (ML is authoritative) but means market profitability is effectively decorative
in the final decision.

**Key finding — LLM validation is stricter than the documented hard rules.** The
"drought" case had all parameters inside the prompt's stated valid ranges
(rainfall 300 mm ∈ [100, 4000], live temp 34°C ∈ [0, 50], ph 6.0 ∈ [4, 9]) yet the LLM
returned `Validation Status: Invalid`. The hard rule set would have passed it; the LLM
rejected it on agronomic judgement. This is a **consistency risk**: the same input can
be `Valid` or `Invalid` depending on LLM sampling (temperature 0.3).

### 3.3 Tool calls

Tool calls are **deterministic** per recommendation:

| Tool | Calls (no user query) | Calls (with user query) |
|------|----------------------|-------------------------|
| OpenWeatherMap HTTP GET | 1 | 1 |
| RandomForest `predict_proba` | 1 | 1 |
| `market_prices.csv` read | 1 | 1 |
| ChromaDB vector search | 1 | **2** |
| Groq LLM completion | 1 | **2** |
| **Total** | **5** | **7** |

A user follow-up question adds exactly one extra vector search + one extra LLM call
(the RAG node answers the question separately, then prepends it to the context).

### 3.4 Recommendation quality

Scored 0–100 by a composite heuristic (ML confidence 30 + cross-agent agreement 25 +
structured-output completeness 20 + decisiveness 15 + real-LLM-used 10):

| Case | Quality | Main drag on score |
|------|--------:|--------------------|
| valid_humid_coastal | 74 | ML/market disagreement |
| valid_with_user_query | 71 | low ML confidence (0.36), ML/market disagreement |
| valid_high_npk_neutral | 68 | low ML confidence (0.275) |
| suboptimal_drought | 60 | blocked + low confidence |
| invalid_extreme_ph_low_npk | 53 | blocked + low confidence |
| **Average** | **65.2** | |

Observations:
- **ML confidence is low across the board (0.28–0.52)** yet the LLM reports
  "Confidence Level: High" on the top two cases — the LLM's stated confidence is
  decoupled from the model's actual probability.
- All structured-output fields parsed cleanly in 5/5 cases — the LLM reliably follows
  the required output format.
- The real LLM (not the rule-based fallback) was used in 5/5 cases.

### 3.5 Time to final recommendation

| Metric | Value |
|--------|-------|
| Cold-start total (case 1: model + embedding load) | **11.8 s** |
| Steady-state total (no user query) | **1.0 – 1.5 s** |
| Steady-state total (with user query) | **2.1 s** |

Average per-node latency (steady state), where the two LLM-backed nodes dominate:

| Node | Avg latency | Bound by |
|------|------------:|----------|
| weather | 0.19 s | network (OpenWeatherMap) |
| crop_prediction | 0.03 s | local CPU (RF inference) |
| market | 0.01 s | local CPU (CSV/pandas) |
| rag | ~0.3 s steady (2.3 s incl. cold embed load) | ChromaDB + optional LLM |
| decision | 1.0 s | Groq LLM round-trip |

The end-to-end latency is essentially **the Groq LLM call(s) plus the weather HTTP GET**;
the ML, market, and vector steps are negligible. Cold start is dominated by loading the
sentence-transformer embedding model and the pickled RF model on first use.

---

## 4. Summary of findings

1. **Linear, deterministic orchestration** — 5 fixed nodes, no branching; 5 tool calls
   per recommendation (7 with a user query).
2. **Live weather silently overrides user temperature/humidity** when an API key is
   present — user-entered temp/humidity never reach the model or validator.
3. **Market analysis never changes the chosen crop** (100% ML-vs-market divergence, 0%
   influence) — it is advisory text only.
4. **LLM validation is stricter and less consistent than the hard rules** — a within-range
   input was rejected as `Invalid`, a non-deterministic outcome at temperature 0.3.
5. **Reported confidence is decoupled from ML probability** — "High" confidence is
   reported on predictions with 0.28–0.48 model probability.
6. **Latency is LLM-bound** — ~1–2 s steady state, ~12 s cold start.

## 5. Measurement methodology

[evaluation/measure_workflow.py](../evaluation/measure_workflow.py) mirrors the
orchestrator's node order so it can observe every intermediate result, and monkey-patches
the five real tool boundaries (`requests.get`, Groq `Completions.create`,
`RandomForestClassifier.predict_proba`, `VectorStore.query`, `pandas.read_csv`) to count
actual invocations. It does **not** modify any agent. Timing uses `time.perf_counter()`
per node and end-to-end. Disagreement and quality are derived from the captured decisions.
Raw per-case output is saved to `evaluation/measurement_results.json`.
