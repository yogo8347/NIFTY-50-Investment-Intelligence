import os
import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands
import streamlit as st

RENAME_MAP = {
    'MUNDRAPORT': 'ADANIPORTS', 'HINDLEVER': 'HINDUNILVR', 'HEROHONDA': 'HEROMOTOCO',
    'INFOSYSTCH': 'INFY', 'TELCO': 'TATAMOTORS', 'TISCO': 'TATASTEEL',
    'UTIBANK': 'AXISBANK', 'KOTAKMAH': 'KOTAKBANK', 'BAJAUTOFIN': 'BAJAJFINSV',
    'JSWSTL': 'JSWSTEEL', 'SSLT': 'VEDL', 'SESAGOA': 'VEDL',
    'BHARTI': 'BHARTIARTL', 'ZEETELE': 'ZEEL', 'UNIPHOS': 'UPL',
    'HINDALC0': 'HINDALCO', 'M&M': 'MM',
}

SECTOR_MAP = {
    'HDFCBANK': 'Banking', 'ICICIBANK': 'Banking', 'KOTAKBANK': 'Banking',
    'AXISBANK': 'Banking', 'SBIN': 'Banking', 'INDUSINDBK': 'Banking',
    'BAJFINANCE': 'Financial Services', 'BAJAJFINSV': 'Financial Services', 'HDFC': 'Financial Services',
    'TCS': 'Information Technology', 'INFY': 'Information Technology', 'HCLTECH': 'Information Technology',
    'WIPRO': 'Information Technology', 'TECHM': 'Information Technology',
    'RELIANCE': 'Energy', 'ONGC': 'Energy', 'BPCL': 'Energy', 'IOC': 'Energy',
    'GAIL': 'Energy', 'COALINDIA': 'Energy', 'NTPC': 'Energy', 'POWERGRID': 'Energy',
    'HINDUNILVR': 'FMCG', 'ITC': 'FMCG', 'BRITANNIA': 'FMCG', 'NESTLEIND': 'FMCG',
    'SUNPHARMA': 'Pharmaceuticals', 'DRREDDY': 'Pharmaceuticals', 'CIPLA': 'Pharmaceuticals',
    'MARUTI': 'Automobile', 'TATAMOTORS': 'Automobile', 'BAJAJ-AUTO': 'Automobile',
    'HEROMOTOCO': 'Automobile', 'EICHERMOT': 'Automobile',
    'TATASTEEL': 'Metals', 'HINDALCO': 'Metals', 'JSWSTEEL': 'Metals',
    'GRASIM': 'Metals', 'VEDL': 'Metals',
    'BHARTIARTL': 'Telecom', 'ZEEL': 'Telecom',
    'ADANIPORTS': 'Infrastructure',
    'ULTRACEMCO': 'Cement', 'SHREECEM': 'Cement',
    'ASIANPAINT': 'Consumer Goods', 'TITAN': 'Consumer Goods',
    'LT': 'Engineering',
    'UPL': 'Chemicals',
}

SKIP_FILES = {'stock_metadata.csv', 'metadata.csv', 'README.csv'}


@st.cache_data(show_spinner=False)
def load_stock_folder(folder_path, max_files=None):
    all_files = [f for f in os.listdir(folder_path)
                 if f.endswith('.csv') and f not in SKIP_FILES]
    if max_files:
        all_files = all_files[:max_files]

    dfs, failed = [], []
    for fname in all_files:
        fpath = os.path.join(folder_path, fname)
        try:
            df = pd.read_csv(fpath)
            df.columns = df.columns.str.strip()
            if 'Date' not in df.columns or 'Close' not in df.columns:
                continue
            df['Date'] = pd.to_datetime(df['Date'])
            if 'Series' in df.columns:
                df = df[df['Series'] == 'EQ']
            if 'Trades' in df.columns:
                df['Trades'] = df['Trades'].fillna(df['Trades'].median())
            df = df.sort_values('Date').reset_index(drop=True)
            dfs.append(df)
        except Exception as e:
            failed.append((fname, str(e)))

    if not dfs:
        return None, {}

    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df['Symbol'] = combined_df['Symbol'].replace(RENAME_MAP)
    combined_df = combined_df.sort_values(['Symbol', 'Date']).reset_index(drop=True)
    combined_df = combined_df.drop_duplicates(subset=['Symbol', 'Date'], keep='last')
    combined_df['Sector'] = combined_df['Symbol'].map(SECTOR_MAP).fillna('Other')

    stock_dict = {sym: grp.reset_index(drop=True)
                  for sym, grp in combined_df.groupby('Symbol')}
    return combined_df, stock_dict


@st.cache_data(show_spinner=False)
def load_index_file(folder_path, filename):
    fpath = os.path.join(folder_path, filename)
    if not os.path.exists(fpath):
        return None
    df = pd.read_csv(fpath)
    df.columns = df.columns.str.strip()
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
    df = df.dropna(subset=['Date'])
    if 'Close' not in df.columns:
        for alt in ['Closing Index Value', 'CLOSE', 'close']:
            if alt in df.columns:
                df.rename(columns={alt: 'Close'}, inplace=True)
                break
    return df.sort_values('Date').reset_index(drop=True)


def engineer_features(df_stock):
    df = df_stock.copy().sort_values('Date').reset_index(drop=True)

    df['Close_lag1'] = df['Close'].shift(1)
    df['Close_lag3'] = df['Close'].shift(3)
    df['Close_lag5'] = df['Close'].shift(5)
    df['Close_lag10'] = df['Close'].shift(10)
    df['Close_lag20'] = df['Close'].shift(20)

    df['Return_1d'] = df['Close'].pct_change(1)
    df['Return_3d'] = df['Close'].pct_change(3)
    df['Return_5d'] = df['Close'].pct_change(5)
    df['Return_10d'] = df['Close'].pct_change(10)
    df['Return_20d'] = df['Close'].pct_change(20)

    df['MA_10'] = df['Close'].rolling(10).mean()
    df['MA_20'] = df['Close'].rolling(20).mean()
    df['MA_50'] = df['Close'].rolling(50).mean()
    df['MA_ratio_10_50'] = df['MA_10'] / df['MA_50']
    df['MA_ratio_20_50'] = df['MA_20'] / df['MA_50']

    df['High_Low_Range'] = (df['High'] - df['Low']) / df['Close']
    df['Close_vs_Open'] = (df['Close'] - df['Open']) / df['Open']
    df['Volatility_10d'] = df['Return_1d'].rolling(10).std()
    df['Volatility_20d'] = df['Return_1d'].rolling(20).std()

    df['RSI_14'] = RSIIndicator(df['Close'], window=14).rsi()
    macd_obj = MACD(df['Close'])
    df['MACD'] = macd_obj.macd()
    df['MACD_signal'] = macd_obj.macd_signal()
    df['MACD_hist'] = macd_obj.macd_diff()

    bb_obj = BollingerBands(df['Close'])
    df['BB_upper'] = bb_obj.bollinger_hband()
    df['BB_lower'] = bb_obj.bollinger_lband()
    df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / bb_obj.bollinger_mavg()
    df['BB_pct'] = (df['Close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])

    df['Volume_lag1'] = df['Volume'].shift(1)
    df['Volume_MA10'] = df['Volume'].rolling(10).mean()
    df['Volume_ratio'] = df['Volume'] / df['Volume_MA10']

    df['Target_Close'] = df['Close'].shift(-1)
    df = df.dropna().reset_index(drop=True)
    return df


FEATURE_COLS = [
    'Close_lag1', 'Close_lag3', 'Close_lag5', 'Close_lag10', 'Close_lag20',
    'Return_1d', 'Return_3d', 'Return_5d', 'Return_10d', 'Return_20d',
    'MA_10', 'MA_20', 'MA_50', 'MA_ratio_10_50', 'MA_ratio_20_50',
    'High_Low_Range', 'Close_vs_Open', 'Volatility_10d', 'Volatility_20d',
    'RSI_14', 'MACD', 'MACD_signal', 'MACD_hist',
    'BB_upper', 'BB_lower', 'BB_width', 'BB_pct',
    'Volume_lag1', 'Volume_MA10', 'Volume_ratio'
]


def compute_risk_metrics(price_series, risk_free_rate=0.06):
    returns = price_series.pct_change().dropna()
    if len(returns) < 50:
        return None
    ann_return = returns.mean() * 252
    ann_vol = returns.std() * np.sqrt(252)
    sharpe = (ann_return - risk_free_rate) / ann_vol if ann_vol > 0 else np.nan
    downside = returns[returns < 0]
    dv = downside.std() * np.sqrt(252) if len(downside) > 0 else np.nan
    sortino = (ann_return - risk_free_rate) / dv if dv and dv > 0 else np.nan
    cum = (1 + returns).cumprod()
    dd = ((cum - cum.cummax()) / cum.cummax()).min()
    var_95 = np.percentile(returns, 5)
    cvar_95 = returns[returns <= var_95].mean()
    calmar = ann_return / abs(dd) if dd != 0 else np.nan
    return {
        'Ann_Return_%': round(ann_return * 100, 2),
        'Ann_Volatility_%': round(ann_vol * 100, 2),
        'Sharpe_Ratio': round(sharpe, 3),
        'Sortino_Ratio': round(sortino, 3),
        'Max_Drawdown_%': round(dd * 100, 2),
        'VaR_95_%': round(var_95 * 100, 3),
        'CVaR_95_%': round(cvar_95 * 100, 3),
        'Calmar_Ratio': round(calmar, 3),
    }
