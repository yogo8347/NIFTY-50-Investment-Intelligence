# 📈 NIFTY-50 Investment Intelligence Platform

An AI-powered investment decision-support system built using historical NIFTY-50 market data (2000–2021).

---

## 🧭 Project Overview

This platform transforms raw historical stock market data into actionable investment intelligence. It covers:

- **Exploratory Data Analysis** — price history, VIX, correlations, volatility heatmaps, sector analysis
- **Stock Predictor Engine** — Linear Regression vs XGBoost with SHAP explainability
- **Portfolio Construction** — Markowitz MPT with Monte Carlo + scipy exact optimisation (Conservative / Balanced / Aggressive)
- **Risk Assessment** — Sharpe, Sortino, Max Drawdown, VaR, CVaR, Beta, Anomaly Detection, Market Regime
- **NSE Scrip Support** — Extended predictor covering 1,756 NSE-listed stocks

---

## 📂 Dataset

Download from Kaggle:

- **NIFTY-50 Stock Data (primary):** https://www.kaggle.com/datasets/rohanrao/nifty50-stock-market-data
- **NSE Scrip Extended Data:** https://www.kaggle.com/datasets/stoicstatic/india-stock-data-nse-1990-2020

After downloading, place the data as follows:

```
data/
├── nifty50/        ← individual stock CSVs (RELIANCE.csv, TCS.csv …)
├── index/          ← NIFTY 50.csv, INDIA VIX.csv, NIFTY BANK.csv
└── nse_scrip/      ← 1,756 NSE stock CSVs (optional, for extended predictor)
```

---

## 🗂️ Repository Structure

```
nifty_app/
├── nifty_platform.py       ← Main entry point (Streamlit app)
├── page_modules/
│   ├── overview.py         ← Landing page with market snapshot
│   ├── eda.py              ← EDA & Market Insights
│   ├── predictor.py        ← Stock Predictor (LR + XGBoost + SHAP)
│   ├── portfolio.py        ← Portfolio Builder (MPT + Monte Carlo)
│   └── risk.py             ← Risk Assessment + Anomaly Detection
├── utils/
│   └── data_loader.py      ← Data loading, cleaning, risk metric functions
├── requirements.txt
└── README.md
```

---

## ⚙️ Environment Setup

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/nifty50-investment-intelligence.git
cd nifty50-investment-intelligence
```

### 2. Create and activate a virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 Running the Application

```bash
streamlit run nifty_platform.py
```

Then open your browser at: **http://localhost:8501**

On first launch, set your data folder paths in the **Data Setup** panel in the sidebar.

---

## 📦 Dependencies

See `requirements.txt`. Key libraries:

| Library | Purpose |
|---|---|
| `streamlit` | Web dashboard framework |
| `pandas`, `numpy` | Data processing |
| `plotly` | Interactive visualisations |
| `scikit-learn` | Linear Regression, StandardScaler |
| `xgboost` | Gradient boosted tree predictor |
| `scipy` | Portfolio optimisation |
| `ta` | Technical indicators (RSI, MACD, Bollinger Bands) |
| `shap` | Model explainability |

---

## 📊 Platform Pages

### 🏠 Overview
- KPI metrics: stocks tracked, total records, date range
- Top 5 gainers and laggards (1-year return)
- Sector breakdown and return heatmap

### 📊 EDA & Market Insights
- Multi-stock price history with event markers (COVID, Demonetisation)
- INDIA VIX vs NIFTY 50 vs NIFTY BANK
- Daily return correlation matrix
- Rolling volatility heatmap (2016–2021)
- Sector risk/return scatter
- Volume and return distribution analysis

### 🤖 Stock Predictor
- Feature engineering: 30 features including lag prices, returns, MAs, RSI, MACD, Bollinger Bands
- Temporal train/test split (no data leakage)
- Linear Regression vs XGBoost comparison
- Metrics: MAE, RMSE, R², Directional Accuracy
- XGBoost feature importance
- SHAP global explainability
- Supports all 1,756 NSE scrip stocks (loaded on demand)

### 💼 Portfolio Builder
- Monte Carlo simulation (up to 10,000 portfolios)
- scipy exact optimisation: Min Volatility, Max Sharpe, Max Return
- Sector concentration constraints
- Efficient frontier visualisation
- Holdings breakdown by stock and sector
- Backtest: growth of ₹1,00,000 from 2000–2021
- Rolling 252-day Sharpe ratio
- Drawdown chart
- Benchmark comparison vs NIFTY 50

### ⚖️ Risk Assessment
- Full risk dashboard: Sharpe, Sortino, Calmar, Omega, Recovery Factor
- VaR (Historical + Parametric) and CVaR at 90%, 95%, 99%
- Beta analysis vs NIFTY 50
- Market anomaly detection: Z-score flagging (|Z| > 3σ)
- Market regime classification: Bull / Bear / High Volatility

---

## 📈 Key Results

| Module | Highlight |
|---|---|
| Stock Predictor | XGBoost R² > 0.99 on HDFCBANK test set |
| Portfolio | Balanced portfolio Sharpe Ratio: ~1.05 |
| Backtest | ₹1L → ₹4.19 Cr (Balanced) vs ₹1.16 Cr (NIFTY 50 benchmark) |
| Risk | Best Sharpe: SHREECEM (0.827) |
| Anomaly | 36 trading days flagged as anomalous (|Z| > 3σ) |

---

## ⚠️ Disclaimer

This platform is built for **educational and research purposes only** as part of a Finance Club competition. It does not constitute financial advice. All analysis is based on historical data from 2000–2021.

---

## 📄 Dataset Credit

- Rohan Rao — [NIFTY-50 Stock Market Data](https://www.kaggle.com/datasets/rohanrao/nifty50-stock-market-data) (Kaggle)
- Stoic Static — [India Stock Data NSE 1990–2020](https://www.kaggle.com/datasets/stoicstatic/india-stock-data-nse-1990-2020) (Kaggle)
