import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import streamlit as st

from utils.data_loader import engineer_features, FEATURE_COLS


def compute_metrics(y_true, y_pred, prev_close):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    actual_dir = np.sign(np.array(y_true) - np.array(prev_close))
    pred_dir = np.sign(np.array(y_pred) - np.array(prev_close))
    dir_acc = np.mean(actual_dir == pred_dir) * 100
    return {'MAE': mae, 'RMSE': rmse, 'R2': r2, 'Directional_Accuracy_%': dir_acc}


@st.cache_data(show_spinner=False)
def train_models(symbol, _stock_df):
    df_feat = engineer_features(_stock_df)
    df_feat[FEATURE_COLS] = df_feat[FEATURE_COLS].apply(pd.to_numeric, errors='coerce').fillna(method='ffill').fillna(0)

    n = len(df_feat)
    split_idx = int(n * 0.8)
    train = df_feat.iloc[:split_idx]
    test = df_feat.iloc[split_idx:]

    X_train, y_train = train[FEATURE_COLS], train['Target_Close']
    X_test, y_test = test[FEATURE_COLS], test['Target_Close']
    prev_close_test = test['Close']

    # Linear Regression
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)
    lr = LinearRegression()
    lr.fit(X_train_sc, y_train)
    lr_pred = lr.predict(X_test_sc)
    lr_metrics = compute_metrics(y_test, lr_pred, prev_close_test)

    # XGBoost
    xgb = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=4,
                        subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0)
    xgb.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    xgb_pred = xgb.predict(X_test)
    xgb_metrics = compute_metrics(y_test, xgb_pred, prev_close_test)

    importances = pd.Series(xgb.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)

    return {
        'test_df': test,
        'lr_pred': lr_pred,
        'xgb_pred': xgb_pred,
        'y_test': y_test,
        'prev_close': prev_close_test,
        'lr_metrics': lr_metrics,
        'xgb_metrics': xgb_metrics,
        'feature_importance': importances,
        'train_dates': (train['Date'].min(), train['Date'].max()),
        'test_dates': (test['Date'].min(), test['Date'].max()),
    }
