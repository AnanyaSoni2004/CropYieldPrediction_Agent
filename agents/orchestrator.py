"""
LangGraph Orchestrator
-----------------------
Defines the multi-agent workflow as a directed graph using LangGraph.

Graph topology:
    START
      └─► crop_prediction_node
            └─► weather_node
                  └─► market_node
                        └─► rag_node
                              └─► decision_node
                                    └─► END

Each node reads from and writes to the shared AgentState TypedDict.
"""
from __future__ import annotations

from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from agents.crop_prediction_agent import CropPredictionAgent
from agents.weather_agent import WeatherAgent
from agents.market_agent import MarketAgent
from agents.decision_agent import DecisionAgent
from agents.rag_knowledge_agent import RAGKnowledgeAgent

# ---------------------------------------------------------------------------
# Shared state schema
# ---------------------------------------------------------------------------

class AgentState(TypedDict, total=False):
    # ---- inputs ----
    soil_data:   dict[str, float]   # N,P,K,temp,humidity,ph,rainfall
    location:    str                # city or "lat,lon"
    user_query:  str                # optional follow-up question

    # ---- intermediate results ----
    crop_result:    dict[str, Any]
    weather_result: dict[str, Any]
    market_result:  dict[str, Any]
    rag_context:    str

    # ---- final output ----
    final_recommendation: dict[str, Any]


# ---------------------------------------------------------------------------
# Node functions  (each node is a plain function AgentState → dict)
# ---------------------------------------------------------------------------

def crop_prediction_node(state: AgentState) -> dict:
    agent  = CropPredictionAgent()
    result = agent.predict(state["soil_data"])
    return {"crop_result": result}


def weather_node(state: AgentState) -> dict:
    agent  = WeatherAgent()
    result = agent.get_weather(state.get("location", "New Delhi"))
    return {"weather_result": result}


def market_node(state: AgentState) -> dict:
    agent  = MarketAgent()
    candidate_crops = [c["crop"] for c in state["crop_result"]["top_crops"]]
    result = agent.analyse(candidate_crops)
    return {"market_result": result}


def rag_node(state: AgentState) -> dict:
    agent   = RAGKnowledgeAgent()
    crop    = state["crop_result"]["top_prediction"]
    query   = state.get("user_query") or f"cultivation practices for {crop}"
    context = agent.retrieve_for_crop(crop)
    # If there is a specific user question, append that answer too
    if state.get("user_query"):
        extra = agent.query(query)
        context = extra["answer"] + "\n\n" + context
    return {"rag_context": context}


def decision_node(state: AgentState) -> dict:
    agent  = DecisionAgent()
    result = agent.decide(
        crop_result    = state["crop_result"],
        weather_result = state["weather_result"],
        market_result  = state["market_result"],
        rag_context    = state.get("rag_context", ""),
    )
    return {"final_recommendation": result}


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    """Build and compile the AgroAgent LangGraph workflow."""
    builder = StateGraph(AgentState)

    builder.add_node("crop_prediction", crop_prediction_node)
    builder.add_node("weather",         weather_node)
    builder.add_node("market",          market_node)
    builder.add_node("rag",             rag_node)
    builder.add_node("decision",        decision_node)

    builder.add_edge(START,             "crop_prediction")
    builder.add_edge("crop_prediction", "weather")
    builder.add_edge("weather",         "market")
    builder.add_edge("market",          "rag")
    builder.add_edge("rag",             "decision")
    builder.add_edge("decision",        END)

    return builder.compile()


# Singleton compiled graph (imported by the API)
graph = build_graph()


def run(soil_data: dict[str, float], location: str = "New Delhi", user_query: str = "") -> AgentState:
    """Execute the full multi-agent pipeline and return the final state."""
    initial: AgentState = {
        "soil_data":  soil_data,
        "location":   location,
        "user_query": user_query,
    }
    return graph.invoke(initial)
