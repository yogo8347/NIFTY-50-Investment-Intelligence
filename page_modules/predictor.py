"""
page_modules/predictor.py  v2
==============================
Stock Predictor — supports NIFTY-50 stocks AND 1,756 NSE Scrip stocks.
NSE scrip stocks are loaded on-demand (single CSV) so startup stays fast.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
import os

try:
    from ta.momentum import RSIIndicator
    from ta.trend    import MACD
    from ta.volatility import BollingerBands
    _TA_OK = True
except ImportError:
    _TA_OK = False

BG   = "#0d1117"
CARD = "#161b22"
GRID = "#21262d"
BLUE = "#58a6ff"
GREEN = "#2ea043"
RED   = "#da3633"
TEXT  = "#c9d1d9"
MUTED = "#8b949e"


# ── Feature engineering ───────────────────────────────────────────────────────
def _engineer_features(df_stock: pd.DataFrame) -> pd.DataFrame:
    df = df_stock.copy().sort_values("Date").reset_index(drop=True)

    for lag in [1, 3, 5, 10, 20]:
        df[f"Close_lag{lag}"]  = df["Close"].shift(lag)
        df[f"Return_{lag}d"]   = df["Close"].pct_change(lag)

    df["MA_10"]          = df["Close"].rolling(10).mean()
    df["MA_20"]          = df["Close"].rolling(20).mean()
    df["MA_50"]          = df["Close"].rolling(50).mean()
    df["MA_ratio_10_50"] = df["MA_10"] / df["MA_50"]
    df["MA_ratio_20_50"] = df["MA_20"] / df["MA_50"]

    if "High" in df.columns and "Low" in df.columns:
        df["High_Low_Range"] = (df["High"] - df["Low"]) / df["Close"]
    else:
        df["High_Low_Range"] = 0.0

    if "Open" in df.columns:
        df["Close_vs_Open"] = (df["Close"] - df["Open"]) / df["Open"]
    else:
        df["Close_vs_Open"] = 0.0

    df["Volatility_10d"] = df["Return_1d"].rolling(10).std()
    df["Volatility_20d"] = df["Return_1d"].rolling(20).std()

    if _TA_OK:
        df["RSI_14"]      = RSIIndicator(df["Close"], window=14).rsi()
        macd_obj          = MACD(df["Close"])
        df["MACD"]        = macd_obj.macd()
        df["MACD_signal"] = macd_obj.macd_signal()
        df["MACD_hist"]   = macd_obj.macd_diff()
        bb                = BollingerBands(df["Close"])
        df["BB_upper"]    = bb.bollinger_hband()
        df["BB_lower"]    = bb.bollinger_lband()
        df["BB_width"]    = (df["BB_upper"] - df["BB_lower"]) / bb.bollinger_mavg()
        df["BB_pct"]      = (df["Close"] - df["BB_lower"]) / (df["BB_upper"] - df["BB_lower"])
    else:
        for c in ["RSI_14", "MACD", "MACD_signal", "MACD_hist",
                  "BB_upper", "BB_lower", "BB_width", "BB_pct"]:
            df[c] = np.nan

    if "Volume" in df.columns:
        df["Volume_lag1"]  = df["Volume"].shift(1)
        df["Volume_MA10"]  = df["Volume"].rolling(10).mean()
        df["Volume_ratio"] = df["Volume"] / df["Volume_MA10"]
    else:
        df["Volume_lag1"] = df["Volume_MA10"] = df["Volume_ratio"] = 0.0

    df["Target_Close"] = df["Close"].shift(-1)
    return df.dropna().reset_index(drop=True)


_FEATURE_COLS = [
    "Close_lag1", "Close_lag3", "Close_lag5", "Close_lag10", "Close_lag20",
    "Return_1d", "Return_3d", "Return_5d", "Return_10d", "Return_20d",
    "MA_10", "MA_20", "MA_50", "MA_ratio_10_50", "MA_ratio_20_50",
    "High_Low_Range", "Close_vs_Open",
    "Volatility_10d", "Volatility_20d",
    "RSI_14", "MACD", "MACD_signal", "MACD_hist",
    "BB_upper", "BB_lower", "BB_width", "BB_pct",
    "Volume_lag1", "Volume_MA10", "Volume_ratio",
]


def _compute_metrics(y_true, y_pred, prev_close):
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    act_dir  = np.sign(np.array(y_true)  - np.array(prev_close))
    pred_dir = np.sign(np.array(y_pred)  - np.array(prev_close))
    dir_acc  = np.mean(act_dir == pred_dir) * 100
    return {"MAE": mae, "RMSE": rmse, "R2": r2, "DirAcc_%": dir_acc}


def _load_scrip_stock(scrip_path: str, symbol: str):
    """Load a single NSE scrip stock CSV on demand."""
    fpath = os.path.join(scrip_path, f"{symbol}.csv")
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


# ── Page ──────────────────────────────────────────────────────────────────────
def render(nifty_stocks, scrip_symbols, scrip_path, data_loaded):
    st.markdown("## 🤖 Stock Predictor")
    st.markdown("*Temporal train/test split · Linear Regression vs XGBoost · SHAP explainability*")
    st.markdown("---")

    if not data_loaded:
        st.error("❌ No data loaded.")
        return

    try:
        from sklearn.linear_model import LinearRegression
        from sklearn.preprocessing import StandardScaler
        from xgboost import XGBRegressor
    except ImportError as e:
        st.error(f"Missing library: {e}. Run `pip install scikit-learn xgboost`")
        return

    # ── Source selector ───────────────────────────────────────────────────────
    nifty_syms = sorted(nifty_stocks.keys())
    has_scrip  = len(scrip_symbols) > 0

    col_src, col_sym, col_test = st.columns([1, 2, 1])

    with col_src:
        source = st.radio(
            "Data Source",
            ["NIFTY-50 (49 stocks)", "NSE Scrip (1,756 stocks)"],
            disabled=not has_scrip,
            help="NSE Scrip gives access to all listed NSE stocks. Loaded on demand — no slowdown.",
        )

    use_scrip = has_scrip and "NSE Scrip" in source

    with col_sym:
        if use_scrip:
            # Search box for 1756 stocks
            search = st.text_input("Search NSE Symbol", value="", placeholder="e.g. INFY, HDFCBANK…").upper().strip()
            matches = [s for s in scrip_symbols if search in s] if search else scrip_symbols
            if not matches:
                st.warning("No matching symbol found.")
                return
            symbol = st.selectbox("Select Symbol", matches)
        else:
            symbol = st.selectbox(
                "Select Stock", nifty_syms,
                index=nifty_syms.index("HDFCBANK") if "HDFCBANK" in nifty_syms else 0,
            )

    with col_test:
        test_pct = st.slider("Test Set %", 10, 40, 20, step=5)

    if not st.button("🚀 Train Models", type="primary"):
        if use_scrip:
            st.info("🔍 Search for any NSE stock above, then click **Train Models**.")
        else:
            st.info("Select a stock and click **Train Models** to run the predictor.")
        return

    # ── Load data ─────────────────────────────────────────────────────────────
    if use_scrip:
        with st.spinner(f"Loading {symbol} from NSE Scrip…"):
            df_raw = _load_scrip_stock(scrip_path, symbol)
        if df_raw is None:
            st.error(f"Could not load {symbol} from NSE Scrip folder.")
            return
        source_label = "NSE Scrip"
    else:
        df_raw = nifty_stocks.get(symbol)
        if df_raw is None:
            st.error(f"{symbol} not found.")
            return
        source_label = "NIFTY-50"

    st.caption(f"📂 Source: **{source_label}** · Symbol: **{symbol}** · Rows: **{len(df_raw):,}**")

    # ── Feature engineering ───────────────────────────────────────────────────
    with st.spinner(f"Engineering features for {symbol}…"):
        df_feat = _engineer_features(df_raw)
        df_feat[_FEATURE_COLS] = (
            df_feat[_FEATURE_COLS]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0)
        )

    if len(df_feat) < 300:
        st.error(f"Not enough data for {symbol} after feature engineering ({len(df_feat)} rows). Need at least 300.")
        return

    split_idx = int(len(df_feat) * (1 - test_pct / 100))
    train, test = df_feat.iloc[:split_idx], df_feat.iloc[split_idx:]

    X_train, y_train = train[_FEATURE_COLS], train["Target_Close"]
    X_test,  y_test  = test[_FEATURE_COLS],  test["Target_Close"]
    prev_close_test  = test["Close"]

    st.caption(
        f"Train: **{len(train)}** rows "
        f"({train['Date'].min().date()} → {train['Date'].max().date()}) | "
        f"Test: **{len(test)}** rows "
        f"({test['Date'].min().date()} → {test['Date'].max().date()})"
    )

    # ── Linear Regression ─────────────────────────────────────────────────────
    with st.spinner("Training Linear Regression…"):
        scaler = StandardScaler()
        lr     = LinearRegression()
        lr.fit(scaler.fit_transform(X_train), y_train)
        lr_pred    = lr.predict(scaler.transform(X_test))
        lr_metrics = _compute_metrics(y_test, lr_pred, prev_close_test)

    # ── XGBoost ───────────────────────────────────────────────────────────────
    with st.spinner("Training XGBoost…"):
        xgb = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=4,
                           subsample=0.8, colsample_bytree=0.8, random_state=42,
                           verbosity=0)
        xgb.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        xgb_pred    = xgb.predict(X_test)
        xgb_metrics = _compute_metrics(y_test, xgb_pred, prev_close_test)

    # ── Metrics ───────────────────────────────────────────────────────────────
    st.markdown("### 📊 Model Performance")
    c1, c2, c3, c4, c5, c6, c7, c8 = st.columns(8)
    c1.metric("LR — MAE",     f"₹{lr_metrics['MAE']:.2f}")
    c2.metric("LR — RMSE",    f"₹{lr_metrics['RMSE']:.2f}")
    c3.metric("LR — R²",      f"{lr_metrics['R2']:.4f}")
    c4.metric("LR — Dir Acc", f"{lr_metrics['DirAcc_%']:.1f}%")
    c5.metric("XGB — MAE",    f"₹{xgb_metrics['MAE']:.2f}")
    c6.metric("XGB — RMSE",   f"₹{xgb_metrics['RMSE']:.2f}")
    c7.metric("XGB — R²",     f"{xgb_metrics['R2']:.4f}")
    c8.metric("XGB — Dir Acc",f"{xgb_metrics['DirAcc_%']:.1f}%")

    # ── Prediction chart ──────────────────────────────────────────────────────
    st.markdown("### 📈 Actual vs Predicted")
    test_dates = test["Date"]

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=("Linear Regression", "XGBoost"),
                        vertical_spacing=0.08)

    for row, (name, pred) in enumerate(
        [("Linear Regression", lr_pred), ("XGBoost", xgb_pred)], start=1
    ):
        fig.add_trace(go.Scatter(x=test_dates, y=y_test.values,
                                 line=dict(color=BLUE, width=1.5),
                                 name="Actual" if row == 1 else "Actual ",
                                 showlegend=True), row=row, col=1)
        fig.add_trace(go.Scatter(x=test_dates, y=pred,
                                 line=dict(color=RED, width=1.2, dash="dash"),
                                 name=f"{name} Prediction"), row=row, col=1)

    fig.update_layout(paper_bgcolor=BG, plot_bgcolor=CARD, font_color=TEXT,
                      height=550, margin=dict(l=10, r=10, t=30, b=10))
    fig.update_xaxes(gridcolor=GRID)
    fig.update_yaxes(gridcolor=GRID, title_text="Close Price (₹)")
    st.plotly_chart(fig, use_container_width=True)

    # ── XGBoost Feature Importance ────────────────────────────────────────────
    st.markdown("### 🔍 XGBoost Feature Importance")
    fi = pd.Series(xgb.feature_importances_, index=_FEATURE_COLS).sort_values(ascending=True).tail(15)
    fig_fi = px.bar(fi.reset_index(), x=fi.values, y=fi.index, orientation="h",
                    color=fi.values, color_continuous_scale="teal",
                    template="plotly_dark")
    fig_fi.update_layout(paper_bgcolor=BG, plot_bgcolor=CARD, font_color=TEXT,
                         height=380, margin=dict(l=10, r=10, t=10, b=10),
                         showlegend=False, coloraxis_showscale=False,
                         xaxis_title="Importance Score", yaxis_title=None)
    st.plotly_chart(fig_fi, use_container_width=True)

    # ── SHAP ──────────────────────────────────────────────────────────────────
    st.markdown("### 🧠 SHAP Explainability (XGBoost)")
    try:
        import shap
        with st.spinner("Computing SHAP values…"):
            explainer   = shap.TreeExplainer(xgb)
            shap_values = explainer.shap_values(X_test)

        mean_shap = np.abs(shap_values).mean(axis=0)
        shap_imp  = pd.Series(mean_shap, index=_FEATURE_COLS).sort_values(ascending=True).tail(15)

        fig_shap = px.bar(
            shap_imp.reset_index(), x=shap_imp.values, y=shap_imp.index,
            orientation="h", color=shap_imp.values,
            color_continuous_scale="Reds", template="plotly_dark",
            labels={"x": "Mean |SHAP Value| (₹)", "index": "Feature"},
        )
        fig_shap.update_layout(paper_bgcolor=BG, plot_bgcolor=CARD, font_color=TEXT,
                               height=380, coloraxis_showscale=False,
                               margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_shap, use_container_width=True)
        st.caption(
            "SHAP values represent each feature's average monetary contribution (₹) "
            "to the model's prediction vs the baseline. Larger bar = more influential."
        )
    except ImportError:
        st.info("Install SHAP for explainability: `pip install shap`")
    except Exception as e:
        st.warning(f"SHAP skipped: {e}")
