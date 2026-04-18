"""
AgroAgent FastAPI Application
------------------------------
Endpoints:
  POST /recommend      – Full multi-agent crop recommendation pipeline
  POST /chat           – RAG-powered Q&A on agricultural knowledge
  GET  /health         – Health check
  GET  /crops          – List all supported crops
  GET  /market/{crop}  – Market data for a specific crop
"""
import sys
import os

# Ensure project root is on the path so relative imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from agents.orchestrator import run as run_pipeline
from agents.rag_knowledge_agent import RAGKnowledgeAgent
from agents.market_agent import MarketAgent
from utils.helpers import validate_soil_input

# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class SoilInput(BaseModel):
    N:           float = Field(..., ge=0, le=300, description="Nitrogen (kg/ha)")
    P:           float = Field(..., ge=0, le=300, description="Phosphorus (kg/ha)")
    K:           float = Field(..., ge=0, le=300, description="Potassium (kg/ha)")
    temperature: float = Field(..., ge=-10, le=55, description="Temperature (°C)")
    humidity:    float = Field(..., ge=0,  le=100, description="Relative humidity (%)")
    ph:          float = Field(..., ge=0,  le=14,  description="Soil pH")
    rainfall:    float = Field(..., ge=0,  le=5000,description="Annual rainfall (mm)")
    location:    str   = Field("New Delhi", description="City name or 'lat,lon'")
    user_query:  str   = Field("",          description="Optional follow-up question")


class ChatInput(BaseModel):
    question: str = Field(..., min_length=3, description="Farming question")
    n_results: int = Field(3, ge=1, le=10)


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="AgroAgent – Smart Crop Prediction & Advisory System",
    description=(
        "A multi-agent AI system that recommends the best crop using ML, "
        "real-time weather, market data, and a RAG knowledge base."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy-loaded singletons
_rag_agent    = None
_market_agent = None


def get_rag() -> RAGKnowledgeAgent:
    global _rag_agent
    if _rag_agent is None:
        _rag_agent = RAGKnowledgeAgent()
    return _rag_agent


def get_market() -> MarketAgent:
    global _market_agent
    if _market_agent is None:
        _market_agent = MarketAgent()
    return _market_agent


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health", tags=["System"])
def health_check():
    return {"status": "ok", "service": "AgroAgent"}


@app.post("/recommend", tags=["Recommendation"])
def recommend(data: SoilInput):
    """
    Run the full multi-agent pipeline:
    Crop Prediction → Weather → Market → RAG → Decision (LLM)

    **Example request body:**
    ```json
    {
      "N": 90, "P": 42, "K": 43,
      "temperature": 20.8, "humidity": 82, "ph": 6.5, "rainfall": 202,
      "location": "Mumbai"
    }
    ```
    """
    soil = data.model_dump(exclude={"location", "user_query"})
    valid, err = validate_soil_input(soil)
    if not valid:
        raise HTTPException(status_code=422, detail=err)

    try:
        state = run_pipeline(
            soil_data  = soil,
            location   = data.location,
            user_query = data.user_query,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    rec = state["final_recommendation"]
    return {
        "recommended_crop":   rec["recommended_crop"],
        "full_recommendation": rec["llm_response"],
        "model_used":          rec["model_used"],
        "crop_analysis":       state["crop_result"],
        "weather_analysis":    state["weather_result"],
        "market_analysis":     state["market_result"],
    }


@app.post("/chat", tags=["Knowledge Base"])
def chat(data: ChatInput):
    """
    Ask any farming question. Answers are grounded in the agricultural
    knowledge base using Retrieval-Augmented Generation (RAG).

    **Example:** `{"question": "What fertilizer should I use for rice?"}`
    """
    result = get_rag().query(data.question, n_results=data.n_results)
    return {
        "question": result["question"],
        "answer":   result["answer"],
        "sources":  result["sources"],
    }


@app.get("/crops", tags=["Data"])
def list_crops():
    """Return all crops supported by the market price database."""
    df = get_market()._df
    crops = df.to_dict(orient="records")
    return {"total": len(crops), "crops": crops}


@app.get("/market/{crop}", tags=["Data"])
def crop_market(crop: str):
    """Return market data and profitability score for a specific crop."""
    result = get_market().analyse([crop])
    if not result["ranked_crops"]:
        raise HTTPException(status_code=404, detail=f"Crop '{crop}' not found.")
    return result["ranked_crops"][0]
