# AgroAgent — Full Project Documentation

Complete technical documentation for the AgroAgent Smart Crop Prediction & Advisory System.


## 1. Overview

**AgroAgent** is a multi-agent AI system that helps farmers choose the optimal crop for their field conditions. Instead of relying on a single model, it orchestrates five specialised agents — each responsible for a different dimension of the decision:

| Dimension | Agent | Signal |
|-----------|-------|--------|
| **Soil compatibility** | Crop Prediction Agent | ML confidence scores from Random Forest |
| **Weather fitness** | Weather Agent | Real-time temperature, humidity, rainfall suitability |
| **Profitability** | Market Agent | Price per quintal, demand level, price trend |
| **Expert knowledge** | RAG Knowledge Agent | Cultivation practices, fertilizer advice, pest management |
| **Final synthesis** | Decision Agent | LLM-generated advisory combining all signals |

The system is designed with **graceful degradation** — every external dependency (LLM API, weather API) has a built-in fallback, so the system always produces a recommendation.

---

## 2. Architecture

### Pipeline Flow

```
START
  │
  ▼
┌─────────────────────┐
│  Crop Prediction     │  ← Random Forest on soil features (N, P, K, pH, temp, humidity, rainfall)
│  Agent               │  → top-3 crops + confidence + soil summary
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│  Weather Agent       │  ← OpenWeatherMap API (or mock data)
│                      │  → temperature, humidity, wind, suitability label
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│  Market Agent        │  ← market_prices.csv (22 crops)
│                      │  → ranked crops by profitability score
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│  RAG Knowledge       │  ← ChromaDB semantic search over rag/documents/ (9 crop guides)
│  Agent               │  → relevant cultivation context for the predicted crop
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│  Decision Agent      │  ← All previous outputs + LLM (Groq/OpenAI/fallback)
│                      │  → "Recommended Crop: X" + structured reasoning + advice
└─────────┬───────────┘
          ▼
         END
```

### Shared State

All agents communicate through a single `AgentState` TypedDict:

```python
class AgentState(TypedDict, total=False):
    # Inputs
    soil_data:   dict[str, float]   # N, P, K, temperature, humidity, ph, rainfall
    location:    str                # city name or "lat,lon"
    user_query:  str                # optional follow-up question

    # Intermediate results (written by each agent)
    crop_result:    dict[str, Any]
    weather_result: dict[str, Any]
    market_result:  dict[str, Any]
    rag_context:    str

    # Final output
    final_recommendation: dict[str, Any]
```

---

## 3. Agents in Detail

### 3.1 Crop Prediction Agent

**File:** `agents/crop_prediction_agent.py`

**What it does:**
- Loads pre-trained artifacts: `crop_model.pkl` (Random Forest), `scaler.pkl` (StandardScaler), `label_encoder.pkl` (LabelEncoder)
- Takes 7 soil/weather features, scales them, and runs `predict_proba()` to get confidence scores for all 22 crops
- Returns top-3 predictions plus a soil health summary

**Soil summary logic:**
| Parameter | High | Medium | Low |
|-----------|------|--------|-----|
| Nitrogen | > 80 kg/ha | > 40 kg/ha | ≤ 40 kg/ha |
| Phosphorus | > 60 kg/ha | > 30 kg/ha | ≤ 30 kg/ha |
| Potassium | > 60 kg/ha | > 30 kg/ha | ≤ 30 kg/ha |

**pH classification:** strongly acidic (< 5.5) → slightly acidic (< 6.5) → neutral (< 7.5) → slightly alkaline (< 8.5) → strongly alkaline

---

### 3.2 Weather Agent

**File:** `agents/weather_agent.py`

**What it does:**
- If `OPENWEATHER_API_KEY` is set → calls OpenWeatherMap (`/data/2.5/weather`) with city name or lat/lon
- If no API key → returns mock data (25°C, 70% humidity, "partly cloudy")
- Evaluates suitability:

| Condition | Label |
|-----------|-------|
| 15–35°C, 40–85% humidity, no issues | `excellent – ideal growing conditions` |
| No specific issues detected | `good – generally suitable for most crops` |
| Temp > 40°C, < 5°C, humidity > 95%, etc. | `poor – [specific issues]` |

---

### 3.3 Market Agent

**File:** `agents/market_agent.py`

**What it does:**
- Reads `data/market_prices.csv` (22 crops with price, demand, trend)
- For each candidate crop, calculates a **profitability score**:

```
score = 0.5 × (price / 15000) + 0.3 × (demand_score / 3) + 0.2 × (trend_score / 2)
```

Where: `demand_score` = high:3, medium:2, low:1 and `trend_score` = rising:2, stable:1, falling:0

- Returns crops ranked by score + market insights text

---

### 3.4 RAG Knowledge Agent

**File:** `agents/rag_knowledge_agent.py`

**What it does:**
1. On first run, ingests crop knowledge documents from `rag/documents/` into ChromaDB
   - Each `.txt` file is a dedicated agronomic guide for one crop
   - Split on `## ` section headers — each section becomes one chunk (ideal growing conditions, soil requirements, pests, diseases, best practices, expected yield)
   - Embeds using `all-MiniLM-L6-v2` (runs fully offline, no API key needed)
   - Produces ~96 focused chunks across 9 crops
2. For queries, performs semantic search and returns top-N passages
3. If `GROQ_API_KEY` is available → generates a synthesised answer via LLaMA-3
4. If no LLM → returns the most relevant raw passage as an extractive answer

**Knowledge base crops:** Rice, Wheat, Maize, Potato, Tomato, Cotton, Sugarcane, Soybean, Peanut

**Why documents instead of the training dataset:** The knowledge documents contain real agronomic expertise (pest management, soil science, planting calendars, best practices) that the LLM can reason over. Feeding the training CSV into RAG was redundant — the Random Forest already learned the statistical relationships from that data, so converting it back into text summaries added no new information.

---

### 3.5 Decision Agent

**File:** `agents/decision_agent.py`

**What it does:**
- Receives outputs from all 4 previous agents
- **The RF model's top prediction is the authoritative recommended crop.** The LLM's role is to validate the input, explain the recommendation, and surface agronomic caveats — it does not substitute its own crop choice.
- The LLM can only block the RF prediction in two cases: input is genuinely invalid (e.g. temperature 200°C, pH 0) or no crop is suitable at all
- Calls LLM with a structured prompt requiring:

```
Validation Status: <Valid / Invalid>
Issues Found: <list or "None">
Environmental Assessment: <Optimal / Suboptimal / Extreme>
Final Recommendation:
  Recommended Crop: <crop>
Reasoning:
  - Soil: <findings>
  - Weather: <findings>
  - Market: <findings>
ML Prediction Review:
  - Status: <Confirmed / Caveats>
  - Reason: <suitability explanation or caveats>
Suggested Fixes: <improvement suggestions or "None required">
Confidence Level: <High / Medium / Low>
```

**LLM priority:** Groq LLaMA-3-70b → OpenAI GPT-4o-mini → Rule-based fallback

---

## 4. Project Structure

```
CropYieldPrediction_Agent/
│
├── agents/                          # AI Agent modules
│   ├── __init__.py                  # Exports all agent classes
│   ├── orchestrator.py              # LangGraph DAG (5 nodes, sequential)
│   ├── crop_prediction_agent.py     # Random Forest ML inference
│   ├── weather_agent.py             # OWM API + mock fallback
│   ├── market_agent.py              # Profitability scoring & ranking
│   ├── rag_knowledge_agent.py       # ChromaDB RAG pipeline
│   └── decision_agent.py            # LLM synthesis → final advice
│
├── models/                          # ML training & serialised artifacts
│   ├── __init__.py
│   ├── train_model.py               # Train RF classifier (run once)
│   ├── crop_model.pkl               # (generated) 200-tree RF
│   ├── scaler.pkl                   # (generated) StandardScaler
│   └── label_encoder.pkl            # (generated) 22-class encoder
│
├── rag/                             # Retrieval-Augmented Generation
│   ├── __init__.py                  # Exports VectorStore, iter_chunks
│   ├── vector_store.py              # ChromaDB client (cosine similarity)
│   ├── knowledge_base.py            # Reads documents/ → section-level chunks
│   └── documents/                   # Agronomic knowledge guides (one .txt per crop)
│       ├── rice.txt
│       ├── wheat.txt
│       ├── maize.txt
│       ├── potato.txt
│       ├── tomato.txt
│       ├── cotton.txt
│       ├── sugarcane.txt
│       ├── soybean.txt
│       └── peanut.txt
│
├── api/                             # REST API
│   ├── __init__.py
│   └── main.py                      # FastAPI: /recommend, /chat, /crops, /market
│
├── utils/                           # Shared utilities
│   ├── __init__.py                  # Exports loaders + validators
│   ├── data_loader.py               # load_crop_data(), load_market_prices()
│   └── helpers.py                   # validate_soil_input(), format_recommendation()
│
├── data/                            # All data files
│   ├── crop_data.csv                # 2,200 rows — training dataset
│   ├── market_prices.csv            # 22 rows — price/demand/trend
│   └── generate_crop_data.py        # Script to regenerate crop_data.csv
│
├── chroma_db/                       # (auto-created) Persisted vector store
├── streamlit_app.py                 # Streamlit frontend (4 pages)
├── requirements.txt                 # Python dependencies
├── .env.example                     # API key template
├── .gitignore                       # Ignores __pycache__, .pkl, .env, chroma_db
└── AgroAgent_Report.pdf             # Project report document
```

---

## 5. Setup & Installation

### Prerequisites

- Python 3.10 or higher
- pip

### Step 1 — Clone & Install

```bash
git clone https://github.com/AnanyaSoni2004/CropYieldPrediction_Agent.git
cd CropYieldPrediction_Agent
pip install -r requirements.txt
```

**Dependencies installed:**

| Category | Packages |
|----------|----------|
| ML & Data | scikit-learn, numpy, pandas |
| API | fastapi, uvicorn, pydantic |
| LLM | groq, openai |
| Orchestration | langgraph, langchain-core |
| RAG | chromadb, sentence-transformers |
| HTTP | requests |
| Frontend | streamlit, plotly |
| Utilities | python-dotenv |

### Step 2 — API Keys (Optional)

```bash
cp .env.example .env
```

| Key | What For | Required? | Where to Get |
|-----|----------|-----------|-------------|
| `GROQ_API_KEY` | LLM reasoning (LLaMA-3-70b) | Recommended | [console.groq.com](https://console.groq.com) (free) |
| `OPENAI_API_KEY` | Fallback LLM (GPT-4o-mini) | Optional | [platform.openai.com](https://platform.openai.com) |
| `OPENWEATHER_API_KEY` | Live weather data | Optional | [openweathermap.org/api](https://openweathermap.org/api) (free) |

**What happens without keys:**

| Missing Key | Fallback Behaviour |
|-------------|-------------------|
| No LLM keys | Rule-based template recommendation (still accurate, less detailed) |
| No weather key | Mock weather: 25°C, 70% humidity, "partly cloudy" |
| No keys at all | System still works end-to-end with all fallbacks active |

### Step 3 — Train ML Model

```bash
python models/train_model.py
```

Expected output:
```
Loading dataset...
Training Random Forest classifier...

Test Accuracy: 0.9864

Classification Report:
              precision    recall  f1-score   support
    apple       1.00      1.00      1.00        20
   banana       1.00      1.00      1.00        20
   ...
 accuracy                           0.99       440

5-Fold CV Accuracy: 0.9841 ± 0.0048

Feature Importances:
  K              : 0.2134
  humidity       : 0.1987
  ...

Artifacts saved to models/
```

### Step 4 — Launch

```bash
# Option A: Streamlit (recommended)
streamlit run streamlit_app.py

# Option B: FastAPI
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

---

## 6. Streamlit Frontend

The Streamlit app (`streamlit_app.py`, 879 lines) provides a dark-themed glassmorphism UI with 4 pages:

### Page 1: Crop Recommendation

**Inputs:**
- 7 sliders: Nitrogen (0–300), Phosphorus (0–300), Potassium (0–300), pH (0–14), Temperature (-10–55°C), Humidity (0–100%), Rainfall (0–5000mm)
- Text input: Location (city or lat,lon)
- Text input: Optional follow-up question

**Outputs (after running pipeline):**
- Hero card: recommended crop + ML confidence + model used
- Top-3 prediction cards with confidence bars
- Soil analysis: N/P/K status badges + pH + rainfall
- Weather: location, temperature, humidity, suitability badge
- Market: Plotly bar chart (price by crop, coloured by profitability) + insights
- AI Advisory Report: full LLM-generated reasoning

### Page 2: Knowledge Q&A

- 4 preset question buttons (e.g., "What fertilizer should I use for rice?")
- Text area for custom questions
- Slider for number of source documents (1–10)
- Results: RAG answer + expandable source passages with relevance scores

### Page 3: Market Explorer

- Summary metrics: total crops, average price, most expensive, high-demand count
- Filters: demand level + trend (multiselect)
- Plotly bar chart: price comparison, colour-coded by demand
- Plotly scatter: price vs. trend with bubble sizes
- Full data table (sortable)

### Page 4: Dataset Explorer

- Summary: total rows, features, crop types, samples per crop
- Feature selector: histogram + box plot for any of the 7 features
- Correlation heatmap (7×7 matrix)
- Samples-per-crop bar chart
- Descriptive statistics table
- Random sample viewer (5–50 rows)

### UI Design

- **Theme:** Dark gradient background (`#0a1628` → `#0d2818` → `#1a1a2e`)
- **Cards:** Glassmorphism (blur, transparent backgrounds, green borders)
- **Typography:** Inter (Google Fonts), weights 300–800
- **Accents:** Green palette (`#4ade80`, `#059669`, `#6ee7b7`, `#a7f3d0`)
- **Animations:** `fadeInUp` on result cards, hover `translateY` on glass cards
- **Charts:** Plotly with transparent backgrounds and green-toned colour scales

---

## 7. FastAPI Reference

**Base URL:** `http://localhost:8000`
**Docs:** `http://localhost:8000/docs` (Swagger UI)

### POST `/recommend`

Full multi-agent pipeline execution.

**Request body:**
```json
{
  "N": 90,
  "P": 42,
  "K": 43,
  "temperature": 20.8,
  "humidity": 82,
  "ph": 6.5,
  "rainfall": 202,
  "location": "Mumbai",
  "user_query": "What fertilizer should I apply?"
}
```

**Validation rules:**
| Field | Min | Max |
|-------|-----|-----|
| N, P, K | 0 | 300 |
| temperature | -10 | 55 |
| humidity | 0 | 100 |
| ph | 0 | 14 |
| rainfall | 0 | 5000 |

**Response:**
```json
{
  "recommended_crop": "rice",
  "full_recommendation": "Recommended Crop: rice\n\nReasoning:\n- Soil analysis: ...",
  "model_used": "llama3-70b-8192",
  "crop_analysis": { "top_prediction": "rice", "confidence": 0.94, "top_crops": [...], "soil_summary": {...} },
  "weather_analysis": { "location": "Mumbai", "temperature": 30.2, "humidity": 78, "suitability": "..." },
  "market_analysis": { "best_market_crop": "rice", "ranked_crops": [...], "market_insights": "..." }
}
```

---

### POST `/chat`

RAG-powered agricultural Q&A.

**Request:** `{"question": "How to improve soil fertility?", "n_results": 3}`

**Response:** `{"question": "...", "answer": "...", "sources": [{"topic": "...", "snippet": "...", "score": 0.91}]}`

---

### GET `/crops`

Returns all 22 supported crops with market data.

**Response:** `{"total": 22, "crops": [{"crop": "rice", "price_per_quintal": 2200, ...}, ...]}`

---

### GET `/market/{crop}`

Market data for a single crop.

**Example:** `GET /market/coffee`
**Response:** `{"crop": "coffee", "price_per_quintal": 15000, "demand": "high", "trend": "rising", "profitability_score": 0.8}`

---

### GET `/health`

**Response:** `{"status": "ok", "service": "AgroAgent"}`

---

## 8. Datasets

### `data/crop_data.csv` — Training Data

| Property | Detail |
|----------|--------|
| **Rows** | 2,200 (100 per crop) |
| **Features** | N, P, K, temperature, humidity, ph, rainfall |
| **Label** | crop name (22 classes) |
| **Generation** | Synthetic, using normal distributions from per-crop profiles |
| **Script** | `data/generate_crop_data.py` |

**22 supported crops:** rice, maize, chickpea, kidney beans, pigeon peas, moth beans, mung bean, black gram, lentil, pomegranate, banana, mango, grapes, watermelon, muskmelon, apple, orange, papaya, coconut, cotton, jute, coffee

### `data/market_prices.csv` — Market Data

| Crop | Price (₹/quintal) | Demand | Trend |
|------|--------------------|--------|-------|
| coffee | 15,000 | high | rising |
| apple | 9,000 | high | rising |
| pomegranate | 8,000 | high | rising |
| grapes | 7,500 | high | rising |
| cotton | 6,500 | high | rising |
| mango | 6,000 | high | rising |
| mungbean | 6,000 | medium | rising |
| lentil | 5,800 | medium | rising |
| ... | ... | ... | ... |
| watermelon | 1,500 | high | stable |

### `rag/documents/` — Agronomic Knowledge Base

9 crop-specific guides (rice, wheat, maize, potato, tomato, cotton, sugarcane, soybean, peanut) covering ideal growing conditions, soil requirements, pest and disease management, best practices, and expected yield ranges. Each guide is split by `##` section headers into focused chunks (~96 total) for ChromaDB ingestion. These contain real agronomic expertise, not re-derived training data statistics.

---

## 9. Tech Stack

| Component | Technology | Details |
|-----------|-----------|---------|
| **Orchestration** | LangGraph | StateGraph with TypedDict, 5 sequential nodes |
| **ML Model** | scikit-learn | RandomForestClassifier, 200 estimators, ~98.6% accuracy |
| **LLM** | Groq / OpenAI | LLaMA-3-70b-8192 (primary), GPT-4o-mini (fallback) |
| **Vector DB** | ChromaDB | PersistentClient, cosine similarity, auto-populated |
| **Embeddings** | sentence-transformers | all-MiniLM-L6-v2 (offline, no API key) |
| **API** | FastAPI | Pydantic v2 validation, CORS enabled, Uvicorn server |
| **Frontend** | Streamlit | 4-page app, dark glassmorphism theme |
| **Charts** | Plotly | Bar, scatter, histogram, heatmap, all with dark styling |
| **Data** | Pandas + NumPy | CSV loading, data generation, statistical ops |
| **Weather** | OpenWeatherMap | REST API with mock fallback |
| **Config** | python-dotenv | `.env` file for API keys |

---

## 10. How It All Connects

```
User opens Streamlit
      │
      ├── Adjusts sliders (N, P, K, pH, temp, humidity, rainfall)
      ├── Enters location ("Mumbai")
      ├── Clicks "Get Crop Recommendation"
      │
      ▼
streamlit_app.py calls agents/orchestrator.py:run()
      │
      ├── 1. CropPredictionAgent.predict(soil_data)
      │       ├── Loads crop_model.pkl, scaler.pkl, label_encoder.pkl
      │       ├── Scales input → predict_proba() → top-3 crops
      │       └── Returns: {top_prediction, confidence, top_crops, soil_summary}
      │
      ├── 2. WeatherAgent.get_weather("Mumbai")
      │       ├── Calls api.openweathermap.org (or mock)
      │       └── Returns: {location, temperature, humidity, suitability}
      │
      ├── 3. MarketAgent.analyse(["rice", "jute", "maize"])
      │       ├── Reads market_prices.csv
      │       ├── Scores each crop: 0.5×price + 0.3×demand + 0.2×trend
      │       └── Returns: {best_market_crop, ranked_crops, market_insights}
      │
      ├── 4. RAGKnowledgeAgent.retrieve_for_crop("rice")
      │       ├── Semantic search in ChromaDB (indexed from rag/documents/)
      │       └── Returns: section-level context about cultivation, pests, soil, etc.
      │
      └── 5. DecisionAgent.decide(crop, weather, market, rag_context)
              ├── Builds structured prompt with all agent outputs
              ├── Calls Groq LLaMA-3 (or OpenAI, or fallback)
              ├── LLM validates input and provides agronomic reasoning/caveats
              ├── recommended_crop = RF top_prediction (LLM only overrides if input is Invalid)
              └── Returns: {recommended_crop, llm_response, model_used}
      │
      ▼
Streamlit renders results:
      ├── Hero card (crop name + confidence)
      ├── Top-3 prediction cards
      ├── Soil analysis badges
      ├── Weather metrics
      ├── Market bar chart (Plotly)
      └── Full LLM advisory report
```

---
