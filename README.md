# AgroAgent – Smart Crop Prediction & Advisory System

An **Agentic AI** system that collaborates multiple specialised agents to recommend
the best crop for a farmer's field, backed by ML, real-time weather, live market
prices, and a RAG knowledge base.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        LangGraph Orchestrator                       │
│                       (agents/orchestrator.py)                      │
│                                                                     │
│  ┌──────────────┐   ┌───────────────┐   ┌───────────────────────┐  │
│  │  Crop        │──▶│  Weather      │──▶│  Market               │  │
│  │  Prediction  │   │  Agent        │   │  Agent                │  │
│  │  Agent       │   │  (OWM API /   │   │  (market_prices.csv)  │  │
│  │  (RF model)  │   │   mock)       │   │                       │  │
│  └──────────────┘   └───────────────┘   └───────────────────────┘  │
│          │                  │                       │               │
│          └──────────────────┴───────────┬───────────┘               │
│                                         ▼                           │
│                              ┌────────────────────┐                 │
│                              │  RAG Knowledge     │                 │
│                              │  Agent             │                 │
│                              │  (ChromaDB +       │                 │
│                              │   sentence-bert)   │                 │
│                              └────────────────────┘                 │
│                                         │                           │
│                                         ▼                           │
│                              ┌────────────────────┐                 │
│                              │  Decision Agent    │                 │
│                              │  (Groq LLaMA-3 /   │                 │
│                              │   OpenAI GPT)      │                 │
│                              └────────────────────┘                 │
└─────────────────────────────────────────────────────────────────────┘
                                         │
                                         ▼
                              ┌────────────────────┐
                              │   FastAPI Server   │
                              │  /recommend        │
                              │  /chat             │
                              │  /crops            │
                              │  /market/{crop}    │
                              └────────────────────┘
```

---

## Agents

| Agent | File | Role |
|-------|------|------|
| Crop Prediction | `agents/crop_prediction_agent.py` | Random Forest classifier; returns top-N crops with confidence |
| Weather | `agents/weather_agent.py` | Fetches live weather via OpenWeatherMap; evaluates suitability |
| Market | `agents/market_agent.py` | Ranks candidate crops by price, demand, and trend |
| RAG Knowledge | `agents/rag_knowledge_agent.py` | Semantic search over ChromaDB; answers farming queries |
| Decision | `agents/decision_agent.py` | Groq LLaMA-3 / GPT-4o-mini synthesises all signals into final advice |

---

## Project Structure

```
CropPrediction_AgenticAI/
├── agents/
│   ├── crop_prediction_agent.py   # ML inference (Random Forest)
│   ├── weather_agent.py           # OpenWeatherMap integration
│   ├── market_agent.py            # Profitability ranking
│   ├── decision_agent.py          # LLM-powered final recommendation
│   ├── rag_knowledge_agent.py     # ChromaDB RAG agent
│   └── orchestrator.py            # LangGraph multi-agent workflow
├── models/
│   ├── train_model.py             # Training script
│   ├── crop_model.pkl             # Trained RF classifier (generated)
│   ├── scaler.pkl                 # StandardScaler (generated)
│   └── label_encoder.pkl          # LabelEncoder (generated)
├── rag/
│   ├── vector_store.py            # ChromaDB wrapper
│   └── knowledge_base.py          # Knowledge base chunk parser
├── api/
│   └── main.py                    # FastAPI routes
├── utils/
│   ├── data_loader.py             # CSV loaders
│   └── helpers.py                 # Validation & formatting utilities
├── data/
│   ├── crop_data.csv              # 2200-row synthetic crop dataset
│   ├── market_prices.csv          # 22-crop market price table
│   ├── agricultural_knowledge.txt # RAG knowledge base source
│   └── generate_crop_data.py      # Data generation script
├── chroma_db/                     # Persisted vector store (auto-created)
├── .env.example                   # Environment variable template
├── requirements.txt
└── README.md
```

---

## Setup

### 1 – Clone & install dependencies

```bash
git clone <repo-url>
cd CropPrediction_AgenticAI
pip install -r requirements.txt
```

### 2 – Configure environment variables

```bash
cp .env.example .env
# Edit .env and add your API keys
```

| Variable | Required | Source |
|----------|----------|--------|
| `GROQ_API_KEY` | Recommended | https://console.groq.com (free tier) |
| `OPENAI_API_KEY` | Optional | https://platform.openai.com |
| `OPENWEATHER_API_KEY` | Optional | https://openweathermap.org/api (free tier) |

> If no LLM key is provided the system uses a rule-based fallback.  
> If no weather key is provided mock weather data is used.

### 3 – Train the ML model

```bash
python models/train_model.py
```

Expected output:
```
Test Accuracy: 0.9864
5-Fold CV Accuracy: 0.9841 ± 0.0048
Artifacts saved to models/
```

### 4 – Start the API server

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

Interactive docs available at: `http://localhost:8000/docs`

---

## API Reference

### `POST /recommend` – Full crop recommendation

```bash
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "N": 90, "P": 42, "K": 43,
    "temperature": 20.8,
    "humidity": 82,
    "ph": 6.5,
    "rainfall": 202,
    "location": "Mumbai",
    "user_query": "What fertilizer should I apply?"
  }'
```

**Response:**
```json
{
  "recommended_crop": "rice",
  "full_recommendation": "Recommended Crop: rice\n\nReasoning:\n- Soil analysis: High nitrogen levels and slightly acidic pH (6.5) are well-suited for rice...\n- Weather conditions: Mumbai's 20.8°C temperature and 82% humidity are ideal for paddy cultivation...\n- Market trends: Rice commands ₹2200/quintal with high demand and stable prices...\n\nAdditional Advice:\n- Fertilizer usage: Apply urea in three split doses...\n- Best practices: Ensure 5–7 cm standing water during vegetative stage...\n- Risk mitigation: Monitor for blast disease during high humidity periods...",
  "model_used": "llama3-70b-8192",
  "crop_analysis": {
    "top_prediction": "rice",
    "confidence": 0.94,
    "top_crops": [
      {"crop": "rice",   "confidence": 0.94},
      {"crop": "jute",   "confidence": 0.04},
      {"crop": "maize",  "confidence": 0.02}
    ]
  },
  "weather_analysis": {
    "location": "Mumbai",
    "temperature": 30.2,
    "humidity": 78,
    "suitability": "excellent – ideal growing conditions"
  },
  "market_analysis": {
    "best_market_crop": "rice",
    "market_insights": "Rice leads in profitability (₹2200/quintal, high demand, stable trend)."
  }
}
```

---

### `POST /chat` – RAG farming Q&A

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "How do I manage waterlogging in cotton fields?"}'
```

**Response:**
```json
{
  "question": "How do I manage waterlogging in cotton fields?",
  "answer": "Cotton requires deep well-drained black cotton soil. To manage waterlogging: (1) create raised beds or ridges before planting, (2) install field drainage channels, (3) avoid irrigation 7–10 days before anticipated rainfall...",
  "sources": [
    {"topic": "Cotton Cultivation", "snippet": "Cotton grows best at 21-30°C...", "score": 0.91}
  ]
}
```

---

### `GET /market/{crop}` – Market data for a crop

```bash
curl http://localhost:8000/market/coffee
```

---

### `GET /health` – Health check

```bash
curl http://localhost:8000/health
# {"status": "ok", "service": "AgroAgent"}
```

---

## Feature Highlights

- **Multi-agent orchestration** via LangGraph with typed shared state
- **Explainable AI** – LLM provides structured reasoning, not a black-box answer
- **RAG grounding** – ChromaDB + sentence-transformers (fully offline embeddings)
- **Graceful degradation** – works without API keys using mock/fallback responses
- **Input validation** – Pydantic models + range checking on all soil parameters

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Agent Orchestration | LangGraph |
| ML Classifier | Scikit-learn Random Forest |
| LLM | Groq LLaMA-3-70b / OpenAI GPT-4o-mini |
| Vector DB | ChromaDB |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| API | FastAPI + Uvicorn |
| Data | Pandas, NumPy |
| Weather | OpenWeatherMap REST API |
