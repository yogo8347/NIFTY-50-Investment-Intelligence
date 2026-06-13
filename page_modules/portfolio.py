import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from utils.portfolio import run_monte_carlo, optimize_portfolio, backtest_portfolio, RISK_FREE_RATE
from utils.data_loader import SECTOR_MAP


def render(nifty_df, nifty_stocks, nifty50_index, data_loaded):
    st.markdown("# 💼 Portfolio Builder")
    st.markdown("*Markowitz Modern Portfolio Theory — Efficient Frontier & Backtesting*")
    st.markdown("---")

    if not data_loaded:
        st.warning("Load data first via the sidebar.")
        return

    symbols = sorted(nifty_stocks.keys())

    # ── Settings ──────────────────────────────────────────────────────────────
    with st.expander("⚙️ Portfolio Settings", expanded=True):
        col1, col2, col3 = st.columns(3)
        with col1:
            selected_stocks = st.multiselect("Stocks to include", symbols,
                                              default=symbols[:min(15, len(symbols))],
                                              help="Select stocks for portfolio optimisation")
        with col2:
            n_sims = st.slider("Monte Carlo simulations", 500, 5000, 2000, step=500)
            risk_free = st.number_input("Risk-free rate (%)", 3.0, 10.0, 6.0, 0.5) / 100
        with col3:
            initial_inv = st.number_input("Initial investment (₹)", 10000, 10000000, 100000, step=10000)
            st.markdown("<br>", unsafe_allow_html=True)
            run_btn = st.button("🚀 Build Portfolios", use_container_width=True)

    if not run_btn and 'portfolio_data' not in st.session_state:
        st.info("Configure settings above and click **Build Portfolios**.")
        return

    if run_btn:
        if len(selected_stocks) < 5:
            st.error("Select at least 5 stocks.")
            return

        with st.spinner("Running Monte Carlo simulation & optimisation..."):
            # Build returns matrix
            price_pivot = nifty_df[nifty_df['Symbol'].isin(selected_stocks)].pivot_table(
                index='Date', columns='Symbol', values='Close').sort_index()
            returns_matrix = price_pivot.pct_change().dropna(how='all')
            dense = returns_matrix.dropna(thresh=int(0.6 * len(returns_matrix)), axis=1).fillna(0)

            sim_df, sym_list = run_monte_carlo(dense, n_sims)

            opt_con = optimize_portfolio(dense, 'min_vol')
            opt_bal = optimize_portfolio(dense, 'max_sharpe')
            opt_agg = optimize_portfolio(dense, 'max_return')

            st.session_state['portfolio_data'] = {
                'sim_df': sim_df, 'sym_list': sym_list, 'dense': dense,
                'opt_con': opt_con, 'opt_bal': opt_bal, 'opt_agg': opt_agg,
                'initial_inv': initial_inv,
            }
            st.session_state['nifty50_index_bt'] = nifty50_index

    if 'portfolio_data' not in st.session_state:
        return

    data = st.session_state['portfolio_data']
    sim_df = data['sim_df']
    sym_list = data['sym_list']
    dense = data['dense']
    opt_con, opt_bal, opt_agg = data['opt_con'], data['opt_bal'], data['opt_agg']
    initial_inv = data['initial_inv']

    # ── Efficient Frontier ────────────────────────────────────────────────────
    st.markdown("## Efficient Frontier")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sim_df['Volatility'] * 100, y=sim_df['Return'] * 100,
        mode='markers', marker=dict(color=sim_df['Sharpe'], colorscale='RdYlGn',
                                     size=4, opacity=0.5,
                                     colorbar=dict(title='Sharpe', thickness=12, len=0.7)),
        hovertemplate='Vol: %{x:.1f}%<br>Return: %{y:.1f}%<extra></extra>',
        name='Simulated Portfolios'
    ))

    profile_styles = [
        (opt_con, 'Conservative', '#58a6ff', 'circle'),
        (opt_bal, 'Balanced (Max Sharpe)', '#3fb950', 'diamond'),
        (opt_agg, 'Aggressive', '#f85149', 'triangle-up'),
    ]
    for p, label, color, sym_mk in profile_styles:
        fig.add_trace(go.Scatter(
            x=[p['Volatility'] * 100], y=[p['Return'] * 100],
            mode='markers+text', text=[label],
            textposition='top center', textfont=dict(color=color, size=11),
            marker=dict(color=color, size=16, symbol=sym_mk,
                        line=dict(color='white', width=2)),
            name=label
        ))

    fig.update_layout(paper_bgcolor='#161b22', plot_bgcolor='#0d1117',
                       font_color='#e6edf3', height=460, template='plotly_dark',
                       xaxis_title='Annualised Volatility (Risk) %',
                       yaxis_title='Annualised Return %',
                       legend=dict(orientation='h', yanchor='bottom', y=-0.2))
    st.plotly_chart(fig, use_container_width=True)

    # ── Profile cards ─────────────────────────────────────────────────────────
    st.markdown("## Portfolio Profiles")
    col1, col2, col3 = st.columns(3)
    profiles = [(col1, opt_con, 'Conservative', '🛡️'), (col2, opt_bal, 'Balanced', '⚖️'),
                (col3, opt_agg, 'Aggressive', '🚀')]

    for col, p, label, icon in profiles:
        with col:
            st.markdown(f"### {icon} {label}")
            st.metric("Ann. Return", f"{p['Return']*100:.1f}%")
            st.metric("Ann. Volatility", f"{p['Volatility']*100:.1f}%")
            st.metric("Sharpe Ratio", f"{p['Sharpe']:.3f}")

            weights_s = pd.Series(p['Weights'], index=p['Symbols']).sort_values(ascending=False)
            top5 = weights_s.head(5)
            fig_pie = go.Figure(go.Pie(
                labels=top5.index.tolist() + (['Others'] if len(weights_s) > 5 else []),
                values=top5.values.tolist() + ([weights_s.iloc[5:].sum()] if len(weights_s) > 5 else []),
                hole=0.4, textfont_size=10,
                marker_colors=px.colors.qualitative.Set2[:len(top5)+1]
            ))
            fig_pie.update_layout(paper_bgcolor='#161b22', font_color='#e6edf3',
                                   height=260, showlegend=True, margin=dict(t=10, b=10),
                                   legend=dict(font_size=9))
            st.plotly_chart(fig_pie, use_container_width=True)

    # ── Backtesting ───────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("## Portfolio Backtest")
    st.caption(f"Growth of ₹{initial_inv:,} invested at the start of the available data.")

    fig_bt = go.Figure()
    bt_styles = [(opt_con, 'Conservative', '#58a6ff'), (opt_bal, 'Balanced', '#3fb950'),
                 (opt_agg, 'Aggressive', '#f85149')]

    bt_summaries = []
    for p, label, color in bt_styles:
        nav, dd, ann_ret, ann_vol, sharpe, max_dd = backtest_portfolio(
            dense, np.array(p['Weights']), p['Symbols'], initial_inv)
        fig_bt.add_trace(go.Scatter(x=nav.index, y=nav.values, name=label,
                                     line=dict(color=color, width=2)))
        bt_summaries.append({'Profile': label, 'Final Value': f"₹{nav.iloc[-1]:,.0f}",
                              'Ann Return %': f"{ann_ret:.1f}%", 'Ann Vol %': f"{ann_vol:.1f}%",
                              'Sharpe': f"{sharpe:.3f}", 'Max Drawdown %': f"{max_dd:.1f}%"})

    # Benchmark
    if nifty50_index is not None:
        bench = nifty50_index.set_index('Date')['Close'].reindex(dense.index).ffill().dropna()
        bench_rets = bench.pct_change().fillna(0)
        bench_nav = (1 + bench_rets).cumprod() * initial_inv
        fig_bt.add_trace(go.Scatter(x=bench_nav.index, y=bench_nav.values, name='NIFTY 50 Benchmark',
                                     line=dict(color='#8b949e', width=1.5, dash='dash')))

    fig_bt.add_hline(y=initial_inv, line_dash='dot', line_color='#8b949e', opacity=0.5)
    fig_bt.add_vrect(x0='2020-02-15', x1='2020-04-15', fillcolor='#f85149', opacity=0.07,
                      annotation_text='COVID', annotation_font_color='#f85149')
    fig_bt.update_layout(paper_bgcolor='#161b22', plot_bgcolor='#0d1117',
                          font_color='#e6edf3', height=400, template='plotly_dark',
                          hovermode='x unified', yaxis_title='Portfolio Value (₹)',
                          legend=dict(orientation='h', yanchor='bottom', y=-0.2))
    st.plotly_chart(fig_bt, use_container_width=True)

    st.markdown("### Backtest Summary")
    st.dataframe(pd.DataFrame(bt_summaries), use_container_width=True, hide_index=True)
