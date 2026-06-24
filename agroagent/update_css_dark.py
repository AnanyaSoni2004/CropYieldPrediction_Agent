import re

with open("streamlit_app.py", "r") as f:
    content = f.read()

new_css = """<style>
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
.stTextArea > div > div > textarea {
    background: rgba(255,255,255,0.05) !important;
    border-color: rgba(74,222,128,0.2) !important;
    color: #e2e8f0 !important;
    border-radius: 10px !important;
}
.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
    border-color: #4ade80 !important;
    box-shadow: 0 0 0 2px rgba(74,222,128,0.15) !important;
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
.streamlit-expanderHeader {
    background: rgba(255,255,255,0.03) !important;
    border-radius: 10px !important;
}

/* ---- Tables ---- */
.stDataFrame { border-radius: 12px; overflow: hidden; }

/* ---- Hide hamburger menu ---- */
#MainMenu { visibility: hidden; }
</style>"""

content = re.sub(r'<style>.*?</style>', new_css, content, flags=re.DOTALL)

with open("streamlit_app.py", "w") as f:
    f.write(content)
