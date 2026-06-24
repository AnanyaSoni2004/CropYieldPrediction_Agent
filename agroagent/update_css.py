import re

with open("streamlit_app.py", "r") as f:
    content = f.read()

new_css = """<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ---- Global ---- */
html, body, [class*="css"], p, span, div, label, h1, h2, h3, h4, h5, h6 {
    font-family: 'Inter', sans-serif !important;
    color: #000000 !important;
    opacity: 1 !important;
}

/* Fix Streamlit's transparent labels specifically */
.stSlider > div > div > div > label, 
.stTextInput label, 
.stSelectbox label, 
.stMultiSelect label, 
.stTextArea label {
    color: #000000 !important;
    opacity: 1 !important;
    font-weight: 600 !important;
}

.stApp {
    background: #f4f7f6;
}

/* ---- Sidebar ---- */
section[data-testid="stSidebar"] {
    background: #ffffff;
    border-right: 1px solid rgba(0, 0, 0, 0.1);
}
section[data-testid="stSidebar"] .stRadio > label {
    color: #000000 !important;
    font-weight: 600;
}

/* ---- Glassmorphism cards ---- */
.glass-card {
    background: #ffffff;
    border: 1px solid rgba(0, 0, 0, 0.08);
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 16px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.03);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.glass-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(0,0,0,0.06);
}

/* ---- Hero section ---- */
.hero-banner {
    background: linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%);
    border-radius: 20px;
    padding: 40px 36px;
    margin-bottom: 28px;
    border: 1px solid rgba(74,222,128,0.4);
    position: relative;
    overflow: hidden;
}
.hero-title {
    font-size: 2.2rem;
    font-weight: 800;
    color: #000000;
    margin: 0 0 8px 0;
    letter-spacing: -0.5px;
}
.hero-sub {
    font-size: 1.05rem;
    color: #000000;
    margin: 0;
    font-weight: 500;
}

/* ---- Result hero crop card ---- */
.crop-hero {
    background: linear-gradient(135deg, #bbf7d0, #86efac);
    border-radius: 20px;
    padding: 32px;
    text-align: center;
    border: 1px solid rgba(74,222,128,0.4);
    animation: fadeInUp 0.6s ease;
}
.crop-hero h1 {
    font-size: 2.8rem;
    color: #000000;
    margin: 0;
    font-weight: 800;
}
.crop-hero .confidence-badge {
    display: inline-block;
    background: rgba(0,0,0,0.05);
    color: #000000;
    padding: 6px 20px;
    border-radius: 30px;
    font-size: 0.95rem;
    font-weight: 700;
    margin-top: 10px;
}

/* ---- Section headers ---- */
.section-header {
    font-size: 1.25rem;
    font-weight: 700;
    color: #000000;
    margin: 28px 0 12px 0;
    padding-bottom: 8px;
    border-bottom: 2px solid rgba(0,0,0,0.1);
    letter-spacing: -0.3px;
}

/* ---- Metric mini cards ---- */
.metric-card {
    background: #ffffff;
    border: 1px solid rgba(0,0,0,0.08);
    border-radius: 12px;
    padding: 16px;
    text-align: center;
    box-shadow: 0 2px 8px rgba(0,0,0,0.02);
}
.metric-card .label {
    font-size: 0.75rem;
    color: #000000;
    text-transform: uppercase;
    letter-spacing: 1px;
    font-weight: 700;
}
.metric-card .value {
    font-size: 1.5rem;
    font-weight: 800;
    color: #000000;
    margin-top: 4px;
}
.metric-card .value span {
    color: #000000 !important;
}

/* ---- Status badges ---- */
.badge {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 700;
    margin: 2px 4px;
}
.badge-high   { background: rgba(74,222,128,0.3); color: #000000; }
.badge-medium { background: rgba(250,204,21,0.3); color: #000000; }
.badge-low    { background: rgba(248,113,113,0.3); color: #000000; }
.badge-info   { background: rgba(96,165,250,0.3);  color: #000000; }

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
    background-color: #059669 !important;
}
.stSlider [data-baseweb="slider"] {
    margin-top: 4px;
}

/* ---- Form styling ---- */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
    background: #ffffff !important;
    border-color: rgba(0,0,0,0.2) !important;
    color: #000000 !important;
    border-radius: 10px !important;
}
.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
    border-color: #059669 !important;
    box-shadow: 0 0 0 2px rgba(5,150,105,0.2) !important;
}

/* ---- Plotly bg fix ---- */
.js-plotly-plot .plotly .main-svg { background: transparent !important; }

/* ---- Button ---- */
.stButton > button {
    background: #059669 !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 10px 32px !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    transition: all 0.2s ease !important;
    opacity: 1 !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 20px rgba(5,150,105,0.3) !important;
}
.stButton > button * {
    color: #ffffff !important;
}

/* ---- Expander ---- */
.streamlit-expanderHeader {
    background: #ffffff !important;
    border: 1px solid rgba(0,0,0,0.05) !important;
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
