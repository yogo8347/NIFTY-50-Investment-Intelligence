import numpy as np
import pandas as pd
from scipy.optimize import minimize
import streamlit as st

RISK_FREE_RATE = 0.06


def portfolio_stats(weights, mean_returns, cov_matrix):
    port_return = np.dot(weights, mean_returns) * 252
    port_vol = np.sqrt(weights @ cov_matrix @ weights) * np.sqrt(252)
    sharpe = (port_return - RISK_FREE_RATE) / port_vol if port_vol > 0 else 0
    return port_return, port_vol, sharpe


def optimize_portfolio(returns_df, objective='min_vol'):
    symbols = returns_df.columns.tolist()
    n = len(symbols)
    mean_returns = returns_df.mean()
    cov_matrix = returns_df.cov()

    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    bounds = tuple((0, 1) for _ in range(n))
    init_w = np.array([1 / n] * n)

    if objective == 'min_vol':
        fun = lambda w: np.sqrt(w @ cov_matrix @ w) * np.sqrt(252)
    elif objective == 'max_sharpe':
        fun = lambda w: -(portfolio_stats(w, mean_returns, cov_matrix)[2])
    else:
        fun = lambda w: -(np.dot(w, mean_returns) * 252)

    result = minimize(fun, init_w, method='SLSQP', bounds=bounds,
                      constraints=constraints, options={'maxiter': 1000, 'ftol': 1e-9})

    opt_w = result.x
    ret, vol, sharpe = portfolio_stats(opt_w, mean_returns, cov_matrix)
    return {'Return': ret, 'Volatility': vol, 'Sharpe': sharpe,
            'Weights': opt_w.tolist(), 'Symbols': symbols}


@st.cache_data(show_spinner=False)
def run_monte_carlo(_returns_df, n_simulations=3000):
    symbols = _returns_df.columns.tolist()
    n_stocks = len(symbols)
    mean_returns = _returns_df.mean()
    cov_matrix = _returns_df.cov()

    results = []
    np.random.seed(42)
    for _ in range(n_simulations):
        weights = np.random.dirichlet(np.ones(n_stocks))
        port_return = np.sum(mean_returns * weights) * 252
        port_vol = np.sqrt(weights @ cov_matrix @ weights) * np.sqrt(252)
        sharpe = (port_return - RISK_FREE_RATE) / port_vol
        results.append({'Return': port_return, 'Volatility': port_vol,
                        'Sharpe': sharpe, 'Weights': weights.tolist()})

    return pd.DataFrame(results), symbols


def backtest_portfolio(returns_df, weights, symbols, initial=100_000):
    bt_returns = returns_df[symbols].copy()
    daily = (bt_returns * weights).sum(axis=1)
    nav = (1 + daily).cumprod() * initial
    cum = (1 + daily).cumprod()
    dd = ((cum - cum.cummax()) / cum.cummax()) * 100
    ann_ret = daily.mean() * 252 * 100
    ann_vol = daily.std() * np.sqrt(252) * 100
    sharpe = (daily.mean() * 252 - RISK_FREE_RATE) / (daily.std() * np.sqrt(252))
    max_dd = dd.min()
    return nav, dd, ann_ret, ann_vol, sharpe, max_dd
