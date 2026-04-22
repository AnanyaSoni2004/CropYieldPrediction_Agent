"""
AgroAgent - Streamlit Frontend
================================
A rich, multi-page Streamlit UI for the AgroAgent crop prediction & advisory system.

Run:
    streamlit run streamlit_app.py
"""
import sys, os

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(__file__))

# Load .env so API keys (GROQ_API_KEY, etc.) are available to agents
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from agents.weather_agent import LocationNotFoundError

# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AgroAgent - Smart Crop Advisory",
    page_icon="A",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS - Dark agricultural theme with glassmorphism
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ---- Global ---- */
html, body, [class*="css"], p, div, h1, h2, h3, h4, h5, h6 {
    font-family: 'Inter', sans-serif;
    color: #f8fafc;
}

/* Fix Streamlit's transparent labels specifically so they are solid white */
.stSlider > div > div > div > label, 
.stTextInput label, 
.stSelectbox label, 
.stMultiSelect label, 
.stTextArea label {
    color: #e2e8f0 !important;
    opacity: 1 !important;
    font-weight: 600 !important;
}

/* Base Dark Background */
.stApp {
    background: linear-gradient(135deg, #0a1628 0%, #0d2818 40%, #1a1a2e 100%);
}

/* ---- Sidebar ---- */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d2818 0%, #0a1628 100%);
    border-right: 1px solid rgba(74, 222, 128, 0.15);
}
section[data-testid="stSidebar"] .stRadio > label {
    color: #e2e8f0 !important;
    font-weight: 600;
}

/* ---- Glassmorphism cards ---- */
.glass-card {
    background: rgba(255,255,255,0.04);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid rgba(74,222,128,0.12);
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 16px;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.glass-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 32px rgba(74,222,128,0.1);
}

/* ---- Hero section ---- */
.hero-banner {
    background: linear-gradient(135deg, #065f46 0%, #064e3b 40%, #0d3b2e 100%);
    border-radius: 20px;
    padding: 40px 36px;
    margin-bottom: 28px;
    border: 1px solid rgba(74,222,128,0.2);
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -20%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(74,222,128,0.08) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-title {
    font-size: 2.2rem;
    font-weight: 800;
    color: #ecfdf5;
    margin: 0 0 8px 0;
    letter-spacing: -0.5px;
}
.hero-sub {
    font-size: 1.05rem;
    color: #a7f3d0;
    margin: 0;
    font-weight: 400;
}

/* ---- Result hero crop card ---- */
.crop-hero {
    background: linear-gradient(135deg, #065f46, #047857);
    border-radius: 20px;
    padding: 32px;
    text-align: center;
    border: 1px solid rgba(74,222,128,0.3);
    animation: fadeInUp 0.6s ease;
}
.crop-hero h1 {
    font-size: 2.8rem;
    color: #ecfdf5;
    margin: 0;
    font-weight: 800;
}
.crop-hero .confidence-badge {
    display: inline-block;
    background: rgba(255,255,255,0.15);
    color: #a7f3d0;
    padding: 6px 20px;
    border-radius: 30px;
    font-size: 0.95rem;
    font-weight: 600;
    margin-top: 10px;
}

/* ---- Section headers ---- */
.section-header {
    font-size: 1.25rem;
    font-weight: 700;
    color: #a7f3d0;
    margin: 28px 0 12px 0;
    padding-bottom: 8px;
    border-bottom: 2px solid rgba(74,222,128,0.2);
    letter-spacing: -0.3px;
}

/* ---- Metric mini cards ---- */
.metric-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(74,222,128,0.12);
    border-radius: 12px;
    padding: 16px;
    text-align: center;
}
.metric-card .label {
    font-size: 0.75rem;
    color: #6ee7b7;
    text-transform: uppercase;
    letter-spacing: 1px;
    font-weight: 600;
}
.metric-card .value {
    font-size: 1.5rem;
    font-weight: 700;
    color: #ecfdf5;
    margin-top: 4px;
}
.metric-card .value span {
    color: #ecfdf5 !important;
}

/* ---- Status badges ---- */
.badge {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 600;
    margin: 2px 4px;
}
.badge-high   { background: rgba(74,222,128,0.2); color: #6ee7b7; }
.badge-medium { background: rgba(250,204,21,0.2); color: #fde68a; }
.badge-low    { background: rgba(248,113,113,0.2); color: #fca5a5; }
.badge-info   { background: rgba(96,165,250,0.2);  color: #93c5fd; }

/* ---- Animations ---- */
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(20px); }
    to   { opacity: 1; transform: translateY(0); }
}
.animate-in {
    animation: fadeInUp 0.5s ease;
}

/* ---- Slider styling ---- */
.stSlider > div > div > div > div {
    background-color: #4ade80 !important;
}
.stSlider [data-baseweb="slider"] {
    margin-top: 4px;
}

/* ---- Form styling ---- */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea,
div[data-baseweb="input"],
div[data-baseweb="textarea"],
div[data-baseweb="base-input"] {
    background-color: #0f172a !important;
    border-color: rgba(74,222,128,0.2) !important;
    border-radius: 10px !important;
}
div[data-baseweb="input"] input,
div[data-baseweb="base-input"] input,
div[data-baseweb="textarea"] textarea,
.stTextInput input,
.stTextArea textarea {
    color: #f8fafc !important;
    -webkit-text-fill-color: #f8fafc !important;
    background-color: transparent !important;
}
::placeholder {
    color: #94a3b8 !important;
    opacity: 0.8 !important;
}
.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus,
div[data-baseweb="input"]:focus-within {
    border-color: #4ade80 !important;
    box-shadow: 0 0 0 2px rgba(74,222,128,0.15) !important;
}

/* ---- Dropdowns & Popovers ---- */
div[data-baseweb="select"],
div[data-baseweb="select"] > div,
div[role="listbox"],
div[role="listbox"] ul,
li[role="option"],
div[data-baseweb="popover"] > div {
    background-color: #0f172a !important;
    color: #f8fafc !important;
}
div[data-baseweb="select"] span,
li[role="option"] span {
    color: #f8fafc !important;
}
li[role="option"]:hover,
li[role="option"][aria-selected="true"] {
    background-color: #1e293b !important;
}

/* MultiSelect tags */
span[data-baseweb="tag"] {
    background-color: #064e3b !important;
    border: 1px solid #047857 !important;
}
span[data-baseweb="tag"] span {
    color: #ecfdf5 !important;
    background-color: transparent !important;
}


/* ---- Plotly bg fix ---- */
.js-plotly-plot .plotly .main-svg { background: transparent !important; }

/* ---- Button ---- */
.stButton > button {
    background: linear-gradient(135deg, #059669, #047857) !important;
    color: #ecfdf5 !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 10px 32px !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 20px rgba(5,150,105,0.4) !important;
}
.stButton > button * {
    color: #ecfdf5 !important;
}

/* ---- Expander ---- */
[data-testid="stExpander"] details {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(74,222,128,0.2) !important;
    border-radius: 10px !important;
    overflow: hidden;
}
[data-testid="stExpander"] summary {
    color: #e2e8f0 !important;
    background: transparent !important;
}
[data-testid="stExpanderDetails"] {
    background: #0a1628 !important;
    color: #e2e8f0 !important;
}
.streamlit-expanderHeader {
    background: rgba(255,255,255,0.03) !important;
    border-radius: 10px !important;
}

/* ---- Tables ---- */
.stDataFrame { border-radius: 12px; overflow: hidden; }

/* ---- Hide hamburger menu ---- */
#MainMenu { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------

def _validate_location(loc: str) -> str | None:
    """Return an error message string if location is invalid, else None."""
    loc = loc.strip()
    if not loc:
        return "Location cannot be empty. Please enter a city name, e.g. 'Mumbai'."
    if len(loc) < 2:
        return f"'{loc}' is too short to be a valid location. Please enter a city name, e.g. 'Mumbai'."
    return None


@st.cache_resource(show_spinner=False)
def load_orchestrator():
    """Import and warm up the orchestrator (loads ML model + ChromaDB)."""
    from agents.orchestrator import run as run_pipeline
    return run_pipeline


@st.cache_resource(show_spinner=False)
def load_rag_agent():
    from agents.rag_knowledge_agent import RAGKnowledgeAgent
    return RAGKnowledgeAgent()


@st.cache_data(show_spinner=False)
def load_market_csv():
    return pd.read_csv(os.path.join(os.path.dirname(__file__), "data", "market_prices.csv"))


@st.cache_data(show_spinner=False)
def load_crop_csv():
    return pd.read_csv(os.path.join(os.path.dirname(__file__), "data", "crop_data.csv"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def badge(text: str, level: str = "info") -> str:
    return f'<span class="badge badge-{level}">{text}</span>'


def metric_card(label: str, value: str) -> str:
    return f"""
    <div class="metric-card">
        <div class="label">{label}</div>
        <div class="value">{value}</div>
    </div>
    """


def soil_badge(status: str) -> str:
    lvl = {"high": "high", "medium": "medium", "low": "low"}.get(status, "info")
    return badge(status.title(), lvl)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 20px 0 10px 0;">
        <div style="font-size:2.2rem; font-weight:800; color:#a7f3d0; letter-spacing:-0.5px;">AgroAgent</div>
        <div style="font-size:0.8rem; color:#6ee7b7; margin-top:2px;">Smart Crop Advisory System</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    page = st.radio(
        "Navigate",
        ["Crop Recommendation", "Knowledge Q&A", "Market Explorer", "Dataset Explorer"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("""
    <div style="padding:12px; background:rgba(255,255,255,0.03); border-radius:12px; border:1px solid rgba(74,222,128,0.1);">
        <div style="font-size:0.7rem; color:#6ee7b7; text-transform:uppercase; letter-spacing:1px; font-weight:600; margin-bottom:6px;">Powered By</div>
        <div style="font-size:0.78rem; color:#94a3b8; line-height:1.6;">
            LangGraph Orchestration<br>
            Random Forest ML<br>
            Groq LLaMA-3 / GPT-4o<br>
            ChromaDB RAG<br>
            OpenWeatherMap
        </div>
    </div>
    """, unsafe_allow_html=True)


# ===========================================================================
# PAGE 1: CROP RECOMMENDATION
# ===========================================================================
if page == "Crop Recommendation":

    st.markdown("""
    <div class="hero-banner">
        <p class="hero-title">Crop Recommendation</p>
        <p class="hero-sub">Enter your soil and environmental parameters to get an AI-powered crop recommendation backed by ML, weather data, market analysis, and agricultural knowledge.</p>
    </div>
    """, unsafe_allow_html=True)

    # ---- Input form with SLIDERS ----
    with st.container():
        st.markdown('<div class="section-header">Soil & Environment Parameters</div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            n_val = st.slider("Nitrogen (N) kg/ha", min_value=0.0, max_value=300.0, value=90.0, step=1.0)
            p_val = st.slider("Phosphorus (P) kg/ha", min_value=0.0, max_value=300.0, value=42.0, step=1.0)
            k_val = st.slider("Potassium (K) kg/ha", min_value=0.0, max_value=300.0, value=43.0, step=1.0)
            ph_val = st.slider("Soil pH", min_value=0.0, max_value=14.0, value=6.5, step=0.1)
        with col2:
            temp_val = st.slider("Temperature (C)", min_value=-10.0, max_value=55.0, value=20.8, step=0.1)
            hum_val = st.slider("Humidity (%)", min_value=0.0, max_value=100.0, value=82.0, step=1.0)
            rain_val = st.slider("Rainfall (mm)", min_value=0.0, max_value=5000.0, value=202.0, step=1.0)

        col_loc, col_q = st.columns([1, 2])
        with col_loc:
            location = st.text_input(
                "Location",
                value="New Delhi",
                help="Enter a city name, e.g. 'Mumbai', 'London', 'New York'.",
            )
        with col_q:
            user_query = st.text_input("Additional question (optional)", placeholder="e.g. What fertilizer should I apply?")

    # ---- Submit button ----
    submitted = st.button("Get Crop Recommendation", use_container_width=True)

    if submitted:
        loc_error = _validate_location(location)
        if loc_error:
            st.error(loc_error)
            st.stop()

        soil_data = {
            "N": n_val, "P": p_val, "K": k_val,
            "temperature": temp_val, "humidity": hum_val,
            "ph": ph_val, "rainfall": rain_val,
        }

        from utils.helpers import validate_soil_input
        valid, soil_error = validate_soil_input(soil_data)
        if not valid:
            st.error(f"Invalid input: {soil_error}")
            st.stop()

        with st.spinner("Running multi-agent pipeline: Crop ML > Weather > Market > RAG > LLM Decision"):
            try:
                run_pipeline = load_orchestrator()
                state = run_pipeline(
                    soil_data=soil_data,
                    location=location,
                    user_query=user_query,
                )
            except LocationNotFoundError as e:
                st.error(f"Location error: {e}")
                st.stop()
            except FileNotFoundError as e:
                st.error(f"Error: {e}")
                st.info("Run `python models/train_model.py` to generate the ML model artifacts first.")
                st.stop()
            except Exception as e:
                st.error(f"Pipeline error: {e}")
                st.stop()

        rec = state["final_recommendation"]
        crop_result = state["crop_result"]
        weather_result = state["weather_result"]
        market_result = state["market_result"]

        if weather_result.get("warning"):
            st.warning(f"Weather: {weather_result['warning']}")

        st.markdown("---")

        is_invalid = rec["recommended_crop"].lower() in ("invalid input", "no suitable crop")

        # ---- Hero: Recommended Crop ----
        if is_invalid:
            st.markdown(f"""
            <div class="crop-hero animate-in" style="background:linear-gradient(135deg,#7f1d1d,#991b1b);">
                <div style="font-size:0.9rem;color:#fca5a5;text-transform:uppercase;letter-spacing:2px;font-weight:600;">Recommendation</div>
                <h1 style="color:#fecaca;">{rec['recommended_crop'].title()}</h1>
                <div class="confidence-badge" style="background:rgba(239,68,68,0.2);color:#fca5a5;">
                    Input validation failed — see report below
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="crop-hero animate-in">
                <div style="font-size:0.9rem;color:#a7f3d0;text-transform:uppercase;letter-spacing:2px;font-weight:600;">Recommended Crop</div>
                <h1>{rec['recommended_crop'].title()}</h1>
                <div class="confidence-badge">{crop_result['confidence']*100:.1f}% ML Confidence  |  {rec['model_used']}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("")

        if not is_invalid:
            # ---- Top Predictions ----
            st.markdown('<div class="section-header">Top Crop Predictions</div>', unsafe_allow_html=True)
            pred_cols = st.columns(len(crop_result["top_crops"]))
            medals = ["1st", "2nd", "3rd"]
            for i, crop_info in enumerate(crop_result["top_crops"]):
                with pred_cols[i]:
                    conf_pct = crop_info["confidence"] * 100
                    rank_label = medals[i] if i < 3 else f"{i+1}th"
                    st.markdown(f"""
                    <div class="glass-card" style="text-align:center;">
                        <div style="font-size:1.1rem;font-weight:700;color:#6ee7b7;">{rank_label}</div>
                        <div style="font-size:1.2rem;font-weight:700;color:#ecfdf5;margin:6px 0;">{crop_info['crop'].title()}</div>
                        <div style="font-size:0.85rem;color:#6ee7b7;">{conf_pct:.1f}% confidence</div>
                        <div style="margin-top:8px;background:rgba(74,222,128,0.1);border-radius:8px;height:8px;overflow:hidden;">
                            <div style="width:{conf_pct}%;height:100%;background:linear-gradient(90deg,#4ade80,#059669);border-radius:8px;"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            # ---- Soil Summary ----
            st.markdown('<div class="section-header">Soil Analysis</div>', unsafe_allow_html=True)
            soil_sum = crop_result["soil_summary"]
            sc1, sc2, sc3, sc4, sc5 = st.columns(5)
            with sc1:
                st.markdown(metric_card("Nitrogen", f"{soil_badge(soil_sum['nitrogen'])}"), unsafe_allow_html=True)
            with sc2:
                st.markdown(metric_card("Phosphorus", f"{soil_badge(soil_sum['phosphorus'])}"), unsafe_allow_html=True)
            with sc3:
                st.markdown(metric_card("Potassium", f"{soil_badge(soil_sum['potassium'])}"), unsafe_allow_html=True)
            with sc4:
                st.markdown(metric_card("pH Status", f"{badge(soil_sum['ph_status'].title(), 'info')}"), unsafe_allow_html=True)
            with sc5:
                st.markdown(metric_card("Rainfall", f"<span style='color:#ecfdf5;font-size:1rem;'>{soil_sum['rainfall_mm']} mm</span>"), unsafe_allow_html=True)

            # ---- Weather ----
            st.markdown('<div class="section-header">Weather Analysis</div>', unsafe_allow_html=True)
            wc1, wc2, wc3, wc4 = st.columns(4)
            with wc1:
                st.markdown(metric_card("Location", f"<span style='color:#ecfdf5;font-size:1rem;'>{weather_result['location']}</span>"), unsafe_allow_html=True)
            with wc2:
                st.markdown(metric_card("Temperature", f"<span style='color:#ecfdf5;font-size:1.3rem;'>{weather_result['temperature']}C</span>"), unsafe_allow_html=True)
            with wc3:
                st.markdown(metric_card("Humidity", f"<span style='color:#ecfdf5;font-size:1.3rem;'>{weather_result['humidity']}%</span>"), unsafe_allow_html=True)
            with wc4:
                suit = weather_result.get("suitability", "N/A")
                suit_level = "high" if "excellent" in suit else "medium" if "good" in suit or "moderate" in suit else "low"
                st.markdown(metric_card("Suitability", badge(suit.split("–")[0].strip().title(), suit_level)), unsafe_allow_html=True)

            # ---- Market ----
            st.markdown('<div class="section-header">Market Analysis</div>', unsafe_allow_html=True)
            market_ranked = market_result.get("ranked_crops", [])
            if market_ranked:
                market_df = pd.DataFrame(market_ranked)
                market_df.columns = [c.replace("_", " ").title() for c in market_df.columns]

                fig_market = go.Figure(data=[
                    go.Bar(
                        x=[c["crop"].title() for c in market_ranked],
                        y=[c["price_per_quintal"] for c in market_ranked],
                        marker=dict(
                            color=[c["profitability_score"] for c in market_ranked],
                            colorscale=[[0, "#064e3b"], [0.5, "#059669"], [1, "#4ade80"]],
                            showscale=True,
                            colorbar=dict(title=dict(text="Score", font=dict(color="#94a3b8")), tickfont=dict(color="#94a3b8")),
                        ),
                        text=[f"Rs.{c['price_per_quintal']}" for c in market_ranked],
                        textposition="outside",
                        textfont=dict(color="#cbd5e1", size=12),
                    )
                ])
                fig_market.update_layout(
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#94a3b8"),
                    xaxis=dict(title="Crop", color="#cbd5e1", gridcolor="rgba(74,222,128,0.05)"),
                    yaxis=dict(title="Price (Rs./quintal)", color="#cbd5e1", gridcolor="rgba(74,222,128,0.08)"),
                    margin=dict(l=20, r=20, t=30, b=20),
                    height=340,
                )
                st.plotly_chart(fig_market, use_container_width=True)

                st.markdown(f"""
                <div class="glass-card" style="font-size:0.9rem;color:#94a3b8;">
                    <strong style="color:#a7f3d0;">Market Insights:</strong> {market_result.get('market_insights', '')}
                </div>
                """, unsafe_allow_html=True)

        # ---- AI Advisory Report (always shown) ----
        st.markdown('<div class="section-header">AI Advisory Report</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="glass-card animate-in" style="line-height:1.8; color:#ecfdf5; font-size:0.92rem;">
            {rec['llm_response'].replace(chr(10), '<br>')}
        </div>
        """, unsafe_allow_html=True)


# ===========================================================================
# PAGE 2: KNOWLEDGE Q&A
# ===========================================================================
elif page == "Knowledge Q&A":

    st.markdown("""
    <div class="hero-banner">
        <p class="hero-title">Agricultural Knowledge Q&A</p>
        <p class="hero-sub">Ask any farming question. Answers are grounded in our agricultural knowledge base using Retrieval-Augmented Generation (RAG) with ChromaDB.</p>
    </div>
    """, unsafe_allow_html=True)

    # Preset question suggestions
    st.markdown('<div class="section-header">Try these questions</div>', unsafe_allow_html=True)
    presets = [
        "What fertilizer should I use for rice?",
        "How do I manage waterlogging in cotton fields?",
        "What is the best season to grow wheat?",
        "How to improve soil fertility for vegetables?",
    ]
    preset_cols = st.columns(len(presets))
    for i, preset_q in enumerate(presets):
        with preset_cols[i]:
            btn_label = f"{preset_q[:35]}..." if len(preset_q) > 35 else preset_q
            if st.button(btn_label, key=f"preset_{i}", use_container_width=True):
                st.session_state["rag_question"] = preset_q

    st.markdown("")
    question = st.text_area(
        "Your farming question",
        value=st.session_state.get("rag_question", ""),
        height=100,
        placeholder="Type your agricultural question here...",
    )

    n_results = st.slider("Number of source documents to retrieve", 1, 10, 3)

    if st.button("Search Knowledge Base", use_container_width=True) and question.strip():
        with st.spinner("Searching agricultural knowledge base..."):
            try:
                rag = load_rag_agent()
                result = rag.query(question, n_results=n_results)
            except Exception as e:
                st.error(f"RAG error: {e}")
                st.stop()

        st.markdown("---")

        # Answer
        st.markdown('<div class="section-header">Answer</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="glass-card animate-in" style="line-height:1.8; color:#ecfdf5; font-size:0.95rem;">
            {result['answer'].replace(chr(10), '<br>')}
        </div>
        """, unsafe_allow_html=True)

        # Sources
        st.markdown('<div class="section-header">Sources</div>', unsafe_allow_html=True)
        for src in result.get("sources", []):
            score = src.get("score", 0)
            score_color = "#4ade80" if score > 0.8 else "#facc15" if score > 0.5 else "#f87171"
            with st.expander(f"{src.get('topic', 'General')} - Relevance: {score:.2%}"):
                st.markdown(f"""
                <div style="color:#ecfdf5; font-size:0.88rem; line-height:1.6;">
                    <span style="color:{score_color};font-weight:600;">Relevance: {score:.2%}</span><br><br>
                    {src.get('snippet', '')}
                </div>
                """, unsafe_allow_html=True)
    elif not question.strip():
        st.info("Enter a farming question above or click one of the preset suggestions.")


# ===========================================================================
# PAGE 3: MARKET EXPLORER
# ===========================================================================
elif page == "Market Explorer":

    st.markdown("""
    <div class="hero-banner">
        <p class="hero-title">Market Explorer</p>
        <p class="hero-sub">Browse current market prices, demand levels, and price trends for all 22 supported crops.</p>
    </div>
    """, unsafe_allow_html=True)

    df_market = load_market_csv()

    # ---- Summary metrics ----
    st.markdown('<div class="section-header">Market Overview</div>', unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(metric_card("Total Crops", f"<span style='color:#ecfdf5;font-size:1.4rem;'>{len(df_market)}</span>"), unsafe_allow_html=True)
    with m2:
        avg_price = int(df_market["price_per_quintal"].mean())
        st.markdown(metric_card("Avg Price", f"<span style='color:#ecfdf5;font-size:1.4rem;'>Rs.{avg_price}</span>"), unsafe_allow_html=True)
    with m3:
        highest = df_market.loc[df_market["price_per_quintal"].idxmax()]
        st.markdown(metric_card("Most Expensive", f"<span style='color:#ecfdf5;font-size:1rem;'>{highest['crop'].title()}<br>Rs.{int(highest['price_per_quintal'])}</span>"), unsafe_allow_html=True)
    with m4:
        high_demand = (df_market["demand_level"] == "high").sum()
        st.markdown(metric_card("High Demand", f"<span style='color:#ecfdf5;font-size:1.4rem;'>{high_demand} crops</span>"), unsafe_allow_html=True)

    st.markdown("")

    # ---- Filter controls ----
    fc1, fc2 = st.columns(2)
    with fc1:
        demand_filter = st.multiselect("Filter by Demand", options=["high", "medium", "low"], default=["high", "medium", "low"])
    with fc2:
        trend_filter = st.multiselect("Filter by Trend", options=["rising", "stable", "falling"], default=["rising", "stable", "falling"])

    df_filtered = df_market[
        (df_market["demand_level"].isin(demand_filter)) &
        (df_market["trend"].isin(trend_filter))
    ].copy()
    df_filtered = df_filtered.sort_values("price_per_quintal", ascending=False)

    # ---- Bar chart ----
    st.markdown('<div class="section-header">Price Comparison</div>', unsafe_allow_html=True)
    color_map = {"high": "#4ade80", "medium": "#facc15", "low": "#f87171"}
    fig_prices = px.bar(
        df_filtered,
        x="crop",
        y="price_per_quintal",
        color="demand_level",
        color_discrete_map=color_map,
        labels={"crop": "Crop", "price_per_quintal": "Price (Rs./quintal)", "demand_level": "Demand"},
        text="price_per_quintal",
    )
    fig_prices.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#94a3b8"),
        xaxis=dict(tickangle=-45, gridcolor="rgba(74,222,128,0.05)"),
        yaxis=dict(gridcolor="rgba(74,222,128,0.08)"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#94a3b8")),
        margin=dict(l=20, r=20, t=30, b=60),
        height=420,
    )
    fig_prices.update_traces(texttemplate="Rs.%{text}", textposition="outside", textfont_size=10)
    st.plotly_chart(fig_prices, use_container_width=True)

    # ---- Trend scatter ----
    st.markdown('<div class="section-header">Price vs Trend</div>', unsafe_allow_html=True)
    trend_order = {"falling": 0, "stable": 1, "rising": 2}
    df_filtered["trend_num"] = df_filtered["trend"].map(trend_order)
    fig_scatter = px.scatter(
        df_filtered,
        x="trend_num",
        y="price_per_quintal",
        size="price_per_quintal",
        color="demand_level",
        color_discrete_map=color_map,
        hover_name="crop",
        labels={"trend_num": "Trend", "price_per_quintal": "Price (Rs./quintal)"},
        size_max=30,
    )
    fig_scatter.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#94a3b8"),
        xaxis=dict(
            tickvals=[0, 1, 2],
            ticktext=["Falling", "Stable", "Rising"],
            gridcolor="rgba(74,222,128,0.05)",
        ),
        yaxis=dict(gridcolor="rgba(74,222,128,0.08)"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#94a3b8")),
        margin=dict(l=20, r=20, t=30, b=20),
        height=380,
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    # ---- Data table ----
    st.markdown('<div class="section-header">Full Market Data</div>', unsafe_allow_html=True)
    display_df = df_filtered[["crop", "price_per_quintal", "demand_level", "trend"]].copy()
    display_df.columns = ["Crop", "Price (Rs./quintal)", "Demand", "Trend"]
    display_df["Crop"] = display_df["Crop"].str.title()
    st.dataframe(display_df, use_container_width=True, hide_index=True)


# ===========================================================================
# PAGE 4: DATASET EXPLORER
# ===========================================================================
elif page == "Dataset Explorer":

    st.markdown("""
    <div class="hero-banner">
        <p class="hero-title">Dataset Explorer</p>
        <p class="hero-sub">Explore the crop recommendation training dataset - 2,200 rows with soil, weather, and label data across 22 crop types.</p>
    </div>
    """, unsafe_allow_html=True)

    df_crop = load_crop_csv()

    # ---- Summary ----
    st.markdown('<div class="section-header">Dataset Summary</div>', unsafe_allow_html=True)
    d1, d2, d3, d4 = st.columns(4)
    with d1:
        st.markdown(metric_card("Total Rows", f"<span style='color:#ecfdf5;font-size:1.4rem;'>{len(df_crop):,}</span>"), unsafe_allow_html=True)
    with d2:
        st.markdown(metric_card("Features", f"<span style='color:#ecfdf5;font-size:1.4rem;'>7</span>"), unsafe_allow_html=True)
    with d3:
        st.markdown(metric_card("Crop Types", f"<span style='color:#ecfdf5;font-size:1.4rem;'>{df_crop['label'].nunique()}</span>"), unsafe_allow_html=True)
    with d4:
        samples_per = len(df_crop) // df_crop["label"].nunique()
        st.markdown(metric_card("Samples/Crop", f"<span style='color:#ecfdf5;font-size:1.4rem;'>~{samples_per}</span>"), unsafe_allow_html=True)

    st.markdown("")

    # ---- Feature distributions ----
    st.markdown('<div class="section-header">Feature Distributions</div>', unsafe_allow_html=True)
    features = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
    feature_labels = {
        "N": "Nitrogen (kg/ha)", "P": "Phosphorus (kg/ha)", "K": "Potassium (kg/ha)",
        "temperature": "Temperature (C)", "humidity": "Humidity (%)",
        "ph": "pH", "rainfall": "Rainfall (mm)",
    }
    selected_feature = st.selectbox("Select feature to explore", features, format_func=lambda x: feature_labels[x])

    fig_dist = px.histogram(
        df_crop,
        x=selected_feature,
        color="label",
        marginal="box",
        nbins=40,
        labels={selected_feature: feature_labels[selected_feature], "label": "Crop"},
        opacity=0.75,
    )
    fig_dist.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#94a3b8"),
        xaxis=dict(gridcolor="rgba(74,222,128,0.05)"),
        yaxis=dict(gridcolor="rgba(74,222,128,0.08)"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#cbd5e1", size=10)),
        margin=dict(l=20, r=20, t=30, b=20),
        height=450,
        barmode="overlay",
    )
    st.plotly_chart(fig_dist, use_container_width=True)

    # ---- Correlation heatmap ----
    st.markdown('<div class="section-header">Feature Correlation Matrix</div>', unsafe_allow_html=True)
    corr = df_crop[features].corr()
    fig_corr = px.imshow(
        corr,
        x=[feature_labels[f] for f in features],
        y=[feature_labels[f] for f in features],
        color_continuous_scale=[[0, "#064e3b"], [0.5, "#0a1628"], [1, "#4ade80"]],
        aspect="auto",
        text_auto=".2f",
    )
    fig_corr.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#cbd5e1", size=11),
        margin=dict(l=20, r=20, t=20, b=20),
        height=450,
    )
    st.plotly_chart(fig_corr, use_container_width=True)

    # ---- Crop distribution ----
    st.markdown('<div class="section-header">Samples per Crop</div>', unsafe_allow_html=True)
    crop_counts = df_crop["label"].value_counts().reset_index()
    crop_counts.columns = ["Crop", "Count"]
    fig_crops = px.bar(
        crop_counts,
        x="Crop",
        y="Count",
        color="Count",
        color_continuous_scale=[[0, "#064e3b"], [0.5, "#059669"], [1, "#4ade80"]],
        text="Count",
    )
    fig_crops.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#94a3b8"),
        xaxis=dict(tickangle=-45, gridcolor="rgba(74,222,128,0.05)"),
        yaxis=dict(gridcolor="rgba(74,222,128,0.08)"),
        margin=dict(l=20, r=20, t=30, b=60),
        height=400,
        showlegend=False,
    )
    fig_crops.update_traces(textposition="outside", textfont_size=10)
    st.plotly_chart(fig_crops, use_container_width=True)

    # ---- Descriptive stats ----
    st.markdown('<div class="section-header">Descriptive Statistics</div>', unsafe_allow_html=True)
    desc = df_crop[features].describe().round(2)
    st.dataframe(desc, use_container_width=True)

    # ---- Sample data ----
    st.markdown('<div class="section-header">Sample Data</div>', unsafe_allow_html=True)
    n_rows = st.slider("Number of sample rows", 5, 50, 10)
    st.dataframe(df_crop.sample(n_rows, random_state=42), use_container_width=True, hide_index=True)
