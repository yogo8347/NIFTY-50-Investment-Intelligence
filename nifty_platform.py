"""
NIFTY-50 Investment Intelligence Platform  v2
=============================================
Entry point.  Run with:  streamlit run nifty_platform.py
"""

import os
import streamlit as st

st.set_page_config(
    page_title="NIFTY-50 Intelligence",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Default paths ──────────────────────────────────────────────────────────────
_DEFAULT_DATA_PATH   = r"C:\Users\yojit\Desktop\nifty_streamlit_app\nifty_app\data\nifty50"
_DEFAULT_INDEX_PATH  = r"C:\Users\yojit\Desktop\nifty_streamlit_app\nifty_app\data\index"
_DEFAULT_SCRIP_PATH  = r"C:\Users\yojit\Desktop\nifty_streamlit_app\nifty_app\data\nse_scrip"

# ── Theme ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, .stApp { font-family: 'Inter', sans-serif; background: #0d1117; }

[data-testid="stSidebar"]   { background: #0d1117; border-right: 1px solid #21262d; }
[data-testid="stSidebar"] * { color: #e6edf3 !important; }
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stTextInput label {
    color: #8b949e !important; font-size: 0.72rem;
    text-transform: uppercase; letter-spacing: 0.08em;
}

[data-testid="metric-container"] {
    background: #161b22; border: 1px solid #21262d;
    border-radius: 10px; padding: 1rem 1.2rem;
    transition: border-color 0.2s;
}
[data-testid="metric-container"]:hover { border-color: #58a6ff; }
[data-testid="metric-container"] label {
    color: #8b949e !important; font-size: 0.72rem;
    text-transform: uppercase; letter-spacing: 0.06em;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #e6edf3 !important; font-family: 'JetBrains Mono', monospace;
    font-size: 1.4rem; font-weight: 600;
}
[data-testid="stMetricDelta"] { font-size: 0.8rem; }

.main .block-container { background: #0d1117; padding-top: 1.5rem; max-width: 1400px; }

h1 { color: #e6edf3 !important; font-size: 1.7rem !important; font-weight: 700; letter-spacing: -0.02em; }
h2 { color: #e6edf3 !important; font-size: 1.2rem !important; font-weight: 600;
     border-bottom: 1px solid #21262d; padding-bottom: 0.4rem; margin-top: 1.5rem; }
h3 { color: #8b949e !important; font-size: 1rem !important; font-weight: 500; }
p, li, span { color: #c9d1d9; }

.stButton > button {
    background: #238636; color: #fff; border: none; border-radius: 6px;
    font-weight: 600; font-size: 0.85rem; padding: 0.5rem 1.4rem;
    transition: background 0.2s, transform 0.1s;
}
.stButton > button:hover  { background: #2ea043; transform: translateY(-1px); }
.stButton > button:active { transform: translateY(0); }

[data-testid="stDataFrame"] { border: 1px solid #21262d; border-radius: 8px; overflow: hidden; }

[data-testid="stTabs"] [role="tab"]                       { color: #8b949e; font-weight: 500; }
[data-testid="stTabs"] [role="tab"][aria-selected="true"] { color: #58a6ff; border-bottom-color: #58a6ff; }

hr { border-color: #21262d; }
.stAlert { border-radius: 8px; }
.stNumberInput input, .stTextInput input, .stTextArea textarea {
    background: #161b22 !important; border: 1px solid #30363d !important;
    color: #e6edf3 !important; border-radius: 6px !important;
    font-family: 'JetBrains Mono', monospace; font-size: 0.82rem;
}
.stNumberInput input:focus, .stTextInput input:focus {
    border-color: #58a6ff !important; box-shadow: 0 0 0 3px rgba(88,166,255,0.15) !important;
}
[data-testid="stExpander"] { background: #161b22; border: 1px solid #21262d; border-radius: 8px; }
.stProgress > div > div { background-color: #238636; }
[data-testid="stSelectbox"] > div > div {
    background: #161b22 !important; border: 1px solid #30363d !important;
    color: #e6edf3 !important;
}
[data-testid="stSlider"] [data-testid="stTickBar"] { color: #8b949e; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar header ─────────────────────────────────────────────────────────────
st.sidebar.markdown("""
<div style='text-align:center; padding:0.8rem 0 0.4rem 0;'>
  <span style='font-size:2.4rem;'>📈</span><br>
  <span style='color:#e6edf3;font-size:1.15rem;font-weight:700;letter-spacing:-0.01em;'>NIFTY-50 Platform</span><br>
  <span style='color:#8b949e;font-size:0.7rem;letter-spacing:0.06em;text-transform:uppercase;'>Investment Intelligence v2</span>
</div>
""", unsafe_allow_html=True)
st.sidebar.markdown("---")

# ── Navigation ─────────────────────────────────────────────────────────────────
PAGES = [
    "🏠  Overview",
    "📊  EDA & Market Insights",
    "🤖  Stock Predictor",
    "💼  Portfolio Builder",
    "⚖️  Risk Assessment",
]
page = st.sidebar.radio("Navigate", PAGES, label_visibility="collapsed")

st.sidebar.markdown("---")

# ── Data Setup ─────────────────────────────────────────────────────────────────
with st.sidebar.expander("📂  Data Setup", expanded=True):
    st.markdown(
        "<span style='color:#8b949e;font-size:0.72rem;'>"
        "Edit paths only if you move the data folders.</span>",
        unsafe_allow_html=True,
    )
    data_path  = st.text_input("NIFTY-50 CSV Folder",  value=_DEFAULT_DATA_PATH,
                                help="One CSV per stock (e.g. RELIANCE.csv)")
    index_path = st.text_input("Index CSV Folder",      value=_DEFAULT_INDEX_PATH,
                                help="NIFTY 50.csv, INDIA VIX.csv, NIFTY BANK.csv")
    scrip_path = st.text_input("NSE Scrip CSV Folder",  value=_DEFAULT_SCRIP_PATH,
                                help="1,756 NSE stock CSVs — used in Stock Predictor")

st.session_state["data_path"]  = data_path
st.session_state["index_path"] = index_path
st.session_state["scrip_path"] = scrip_path

# ── Load core data ─────────────────────────────────────────────────────────────
from utils.data_loader import load_stock_folder, load_index_file

@st.cache_data(show_spinner=False)
def get_stocks(path: str):
    if not os.path.exists(path):
        return None, {}
    return load_stock_folder(path)

@st.cache_data(show_spinner=False)
def get_index(folder: str, filename: str):
    return load_index_file(folder, filename)

@st.cache_data(show_spinner=False)
def get_scrip_symbols(path: str):
    """Return sorted list of all NSE scrip symbols (filenames) — fast, no data loaded."""
    if not os.path.exists(path):
        return []
    return sorted([
        os.path.splitext(f)[0]
        for f in os.listdir(path)
        if f.endswith(".csv")
    ])

@st.cache_data(show_spinner=False)
def get_single_scrip(path: str, symbol: str):
    """Load a single NSE scrip CSV on demand."""
    import pandas as pd
    fpath = os.path.join(path, f"{symbol}.csv")
    if not os.path.exists(fpath):
        return None
    try:
        df = pd.read_csv(fpath)
        df.columns = df.columns.str.strip()
        if "Date" not in df.columns or "Close" not in df.columns:
            return None
        df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
        df = df.dropna(subset=["Date"])
        if "Series" in df.columns:
            df = df[df["Series"] == "EQ"]
        df = df.sort_values("Date").reset_index(drop=True)
        if "Symbol" not in df.columns:
            df["Symbol"] = symbol
        return df
    except Exception:
        return None

with st.spinner("Loading market data…"):
    nifty_df, nifty_stocks = get_stocks(data_path)
    nifty50_index          = get_index(index_path, "NIFTY 50.csv")
    india_vix              = get_index(index_path, "INDIA VIX.csv")
    nifty_bank             = get_index(index_path, "NIFTY BANK.csv")
    scrip_symbols          = get_scrip_symbols(scrip_path)

st.session_state["nifty_df"]       = nifty_df
st.session_state["nifty_stocks"]   = nifty_stocks
st.session_state["nifty50_index"]  = nifty50_index
st.session_state["india_vix"]      = india_vix
st.session_state["nifty_bank"]     = nifty_bank
st.session_state["scrip_symbols"]  = scrip_symbols
st.session_state["scrip_path"]     = scrip_path

# ── Sidebar status ─────────────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.markdown("#### 📊 Data Status")

if nifty_df is None:
    data_loaded = False
    st.sidebar.error("❌ Stock folder not found.")
    st.sidebar.caption(f"`{data_path}`")
elif len(nifty_stocks) == 0:
    data_loaded = False
    st.sidebar.error("❌ No valid CSVs found.")
else:
    data_loaded = True
    n_stocks = len(nifty_stocks)
    idx_ok   = nifty50_index is not None
    vix_ok   = india_vix     is not None
    bank_ok  = nifty_bank    is not None
    scrip_ok = len(scrip_symbols) > 0

    c1, c2 = st.sidebar.columns(2)
    c1.metric("NIFTY-50", str(n_stocks))
    c2.metric("Index",    "✓" if idx_ok else "✗")

    if scrip_ok:
        st.sidebar.success(f"✅ NSE Scrip: {len(scrip_symbols):,} stocks")
    else:
        st.sidebar.caption("⚠️ NSE Scrip folder not found")

    for fname, ok in [("NIFTY 50.csv", idx_ok), ("INDIA VIX.csv", vix_ok), ("NIFTY BANK.csv", bank_ok)]:
        if not ok:
            st.sidebar.caption(f"⚠️ {fname} not found")

    if nifty_df is not None:
        d_min = nifty_df["Date"].min().strftime("%d %b %Y")
        d_max = nifty_df["Date"].max().strftime("%d %b %Y")
        st.sidebar.caption(f"📅 {d_min} → {d_max}")

# ── Sidebar footer ─────────────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.markdown(
    "<div style='color:#484f58;font-size:0.67rem;text-align:center;line-height:1.7;'>"
    "Dataset: <a href='https://www.kaggle.com/datasets/rohanrao/nifty50-stock-market-data'"
    " style='color:#58a6ff;' target='_blank'>Kaggle / rohanrao</a><br>"
    "For educational use only · Not financial advice"
    "</div>",
    unsafe_allow_html=True,
)

# ── Route to pages ─────────────────────────────────────────────────────────────
from page_modules import overview, eda, predictor, portfolio, risk

if   page == "🏠  Overview":
    overview.render(nifty_df, nifty_stocks, data_loaded)
elif page == "📊  EDA & Market Insights":
    eda.render(nifty_df, nifty_stocks, nifty50_index, india_vix, nifty_bank, data_loaded)
elif page == "🤖  Stock Predictor":
    predictor.render(nifty_stocks, scrip_symbols, scrip_path, data_loaded)
elif page == "💼  Portfolio Builder":
    portfolio.render(nifty_df, nifty_stocks, nifty50_index, data_loaded)
elif page == "⚖️  Risk Assessment":
    risk.render(nifty_stocks, nifty50_index, data_loaded)
