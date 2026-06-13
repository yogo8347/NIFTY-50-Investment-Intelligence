import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from scipy.stats import norm as scipy_norm
from utils.data_loader import compute_risk_metrics, SECTOR_MAP


def render(nifty_stocks, nifty50_index, data_loaded):
    st.markdown("# ⚖️ Risk Assessment")
    st.markdown("*Sharpe, Sortino, VaR, CVaR, Max Drawdown, Beta — full risk dashboard*")
    st.markdown("---")

    if not data_loaded:
        st.warning("Load data first via the sidebar.")
        return

    symbols = sorted(nifty_stocks.keys())
    tabs = st.tabs(["📋 Risk Dashboard", "📉 Drawdown Analysis", "📊 VaR / CVaR", "🎯 Beta Analysis"])

    # ── Pre-compute risk table ─────────────────────────────────────────────────
    @st.cache_data(show_spinner=False)
    def build_risk_df(_stocks):
        rows = []
        for sym, df_s in _stocks.items():
            m = compute_risk_metrics(df_s.sort_values('Date')['Close'])
            if m:
                m['Symbol'] = sym
                m['Sector'] = SECTOR_MAP.get(sym, 'Other')
                rows.append(m)
        return pd.DataFrame(rows)

    risk_df = build_risk_df(nifty_stocks)

    # ── Tab 1: Risk Dashboard ─────────────────────────────────────────────────
    with tabs[0]:
        st.markdown("### Risk Dashboard — All Stocks")

        col1, col2 = st.columns([1, 2])
        with col1:
            sort_by = st.selectbox("Sort by", ['Sharpe_Ratio', 'Sortino_Ratio', 'Ann_Return_%',
                                                'Ann_Volatility_%', 'Max_Drawdown_%'])
            asc = st.checkbox("Ascending", value=False)
        with col2:
            sector_filter = st.multiselect("Filter by sector",
                                            sorted(risk_df['Sector'].unique()),
                                            default=list(risk_df['Sector'].unique()))

        filtered = risk_df[risk_df['Sector'].isin(sector_filter)].sort_values(sort_by, ascending=asc)
        display_cols = ['Symbol', 'Sector', 'Ann_Return_%', 'Ann_Volatility_%',
                         'Sharpe_Ratio', 'Sortino_Ratio', 'Max_Drawdown_%', 'VaR_95_%', 'CVaR_95_%', 'Calmar_Ratio']
        st.dataframe(
            filtered[display_cols].style
            .background_gradient(subset=['Sharpe_Ratio', 'Sortino_Ratio'], cmap='Greens')
            .background_gradient(subset=['Max_Drawdown_%'], cmap='Reds_r')
            .format({'Ann_Return_%': '{:.2f}%', 'Ann_Volatility_%': '{:.2f}%',
                      'Sharpe_Ratio': '{:.3f}', 'Sortino_Ratio': '{:.3f}',
                      'Max_Drawdown_%': '{:.2f}%', 'VaR_95_%': '{:.3f}%',
                      'CVaR_95_%': '{:.3f}%', 'Calmar_Ratio': '{:.3f}'}),
            use_container_width=True, hide_index=True, height=400
        )

        st.markdown("---")
        c1, c2 = st.columns(2)

        with c1:
            st.markdown("#### Sharpe Ratio Ranking")
            top_sharpe = filtered.nlargest(12, 'Sharpe_Ratio').sort_values('Sharpe_Ratio')
            colors = ['#238636' if v > 0 else '#da3633' for v in top_sharpe['Sharpe_Ratio']]
            fig = go.Figure(go.Bar(x=top_sharpe['Sharpe_Ratio'], y=top_sharpe['Symbol'],
                                    orientation='h', marker_color=colors,
                                    hovertemplate='%{y}: %{x:.3f}<extra></extra>'))
            fig.add_vline(x=0, line_color='#8b949e', line_dash='dash')
            fig.update_layout(paper_bgcolor='#161b22', plot_bgcolor='#0d1117',
                               font_color='#e6edf3', height=360, template='plotly_dark',
                               xaxis_title='Sharpe Ratio')
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            st.markdown("#### Worst Maximum Drawdowns")
            worst_dd = filtered.nsmallest(12, 'Max_Drawdown_%').sort_values('Max_Drawdown_%', ascending=False)
            fig2 = go.Figure(go.Bar(x=worst_dd['Max_Drawdown_%'], y=worst_dd['Symbol'],
                                     orientation='h', marker_color='#da3633',
                                     hovertemplate='%{y}: %{x:.1f}%<extra></extra>'))
            fig2.update_layout(paper_bgcolor='#161b22', plot_bgcolor='#0d1117',
                                font_color='#e6edf3', height=360, template='plotly_dark',
                                xaxis_title='Max Drawdown %')
            st.plotly_chart(fig2, use_container_width=True)

        # Sector average Sharpe
        st.markdown("#### Average Sharpe by Sector")
        sec_sharpe = risk_df.groupby('Sector')['Sharpe_Ratio'].mean().sort_values()
        sec_colors = ['#238636' if v > 0 else '#da3633' for v in sec_sharpe]
        fig3 = go.Figure(go.Bar(x=sec_sharpe.values, y=sec_sharpe.index, orientation='h',
                                 marker_color=sec_colors,
                                 hovertemplate='%{y}: %{x:.3f}<extra></extra>'))
        fig3.add_vline(x=0, line_color='#8b949e', line_dash='dash')
        fig3.update_layout(paper_bgcolor='#161b22', plot_bgcolor='#0d1117',
                            font_color='#e6edf3', height=280, template='plotly_dark')
        st.plotly_chart(fig3, use_container_width=True)

    # ── Tab 2: Drawdown ───────────────────────────────────────────────────────
    with tabs[1]:
        st.markdown("### Drawdown Analysis")
        dd_sym = st.selectbox("Select stock", symbols, key='dd_sym')
        df_dd = nifty_stocks[dd_sym].sort_values('Date')
        rets = df_dd['Close'].pct_change().dropna()
        cum = (1 + rets).cumprod()
        dd_series = ((cum - cum.cummax()) / cum.cummax()) * 100
        dates_dd = df_dd['Date'].iloc[1:].reset_index(drop=True)

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.5, 0.5])
        fig.add_trace(go.Scatter(x=dates_dd, y=(cum.values - 1) * 100,
                                  fill='tozeroy', fillcolor='rgba(88,166,255,0.07)',
                                  line=dict(color='#58a6ff', width=1.8),
                                  name='Cumulative Return %'), row=1, col=1)
        fig.add_trace(go.Scatter(x=dates_dd, y=dd_series.values,
                                  fill='tozeroy', fillcolor='rgba(248,81,73,0.15)',
                                  line=dict(color='#f85149', width=1.2),
                                  name='Drawdown %'), row=2, col=1)
        fig.add_hline(y=dd_series.min(), row=2, col=1, line_dash='dash', line_color='#ffa500',
                      annotation_text=f'Max DD: {dd_series.min():.1f}%',
                      annotation_font_color='#ffa500')
        fig.update_layout(paper_bgcolor='#161b22', plot_bgcolor='#0d1117',
                           font_color='#e6edf3', height=480, template='plotly_dark',
                           hovermode='x unified')
        st.plotly_chart(fig, use_container_width=True)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Max Drawdown", f"{dd_series.min():.1f}%")
        c2.metric("Total Return", f"{(cum.iloc[-1]-1)*100:.1f}%")
        c3.metric("Ann. Volatility", f"{rets.std()*np.sqrt(252)*100:.1f}%")
        c4.metric("Win Rate", f"{(rets > 0).mean()*100:.1f}%")

    # ── Tab 3: VaR / CVaR ─────────────────────────────────────────────────────
    with tabs[2]:
        st.markdown("### Value at Risk & Expected Shortfall")
        var_sym = st.selectbox("Select stock", symbols, key='var_sym')
        df_var = nifty_stocks[var_sym].sort_values('Date')
        rets_v = df_var['Close'].pct_change().dropna() * 100

        levels = [0.90, 0.95, 0.99]
        var_hist = [np.percentile(rets_v, (1-cl)*100) for cl in levels]
        var_param = [rets_v.mean() + scipy_norm.ppf(1-cl) * rets_v.std() for cl in levels]
        cvar_vals = [rets_v[rets_v <= vl].mean() for vl in var_hist]

        var_table = pd.DataFrame({
            'Confidence': ['90%', '95%', '99%'],
            'Historical VaR %': [round(v, 3) for v in var_hist],
            'Parametric VaR %': [round(v, 3) for v in var_param],
            'CVaR / ES %': [round(v, 3) for v in cvar_vals],
        })
        st.dataframe(var_table, use_container_width=True, hide_index=True)

        # Histogram with VaR
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=rets_v, nbinsx=100, name='Daily Returns',
                                    marker_color='#58a6ff', opacity=0.6))
        for vl, label in [(var_hist[1], 'Hist VaR 95%'), (var_param[1], 'Param VaR 95%'),
                           (cvar_vals[1], 'CVaR 95%')]:
            fig.add_vline(x=vl, line_dash='dash',
                          annotation_text=f'{label}: {vl:.2f}%',
                          annotation_font_size=10)
        fig.update_layout(paper_bgcolor='#161b22', plot_bgcolor='#0d1117',
                           font_color='#e6edf3', height=350, template='plotly_dark',
                           xaxis_title='Daily Return %', yaxis_title='Frequency',
                           xaxis=dict(range=[-8, 8]))
        st.plotly_chart(fig, use_container_width=True)
        st.info("**Interpretation:** If Historical VaR is worse than Parametric VaR, "
                "returns have fat tails — extreme losses occur more often than a normal distribution predicts.")

    # ── Tab 4: Beta ────────────────────────────────────────────────────────────
    with tabs[3]:
        st.markdown("### Beta Analysis vs NIFTY 50")
        if nifty50_index is None:
            st.info("Place 'NIFTY 50.csv' in your index folder to enable beta analysis.")
            return

        @st.cache_data(show_spinner=False)
        def compute_betas(_stocks, _index):
            nifty_ret = _index.set_index('Date')['Close'].pct_change().dropna().rename('NIFTY50')
            rows = []
            for sym, df_s in _stocks.items():
                sr = df_s.set_index('Date')['Close'].pct_change().dropna().rename(sym)
                aligned = pd.concat([sr, nifty_ret], axis=1).dropna()
                if len(aligned) < 100:
                    continue
                cov_mat = np.cov(aligned[sym], aligned['NIFTY50'])
                beta = cov_mat[0, 1] / cov_mat[1, 1]
                corr = np.corrcoef(aligned[sym], aligned['NIFTY50'])[0, 1]
                rows.append({'Symbol': sym, 'Beta': round(beta, 3), 'Market_Corr': round(corr, 3),
                              'Sector': SECTOR_MAP.get(sym, 'Other'),
                              'Category': ('Defensive (β<0.8)' if beta < 0.8 else
                                           'Aggressive (β>1.2)' if beta > 1.2 else
                                           'Market-Neutral (0.8–1.2)')})
            return pd.DataFrame(rows)

        beta_df = compute_betas(nifty_stocks, nifty50_index)

        col1, col2 = st.columns([3, 1])
        with col2:
            st.markdown("**Beta Distribution**")
            for cat in ['Defensive (β<0.8)', 'Market-Neutral (0.8–1.2)', 'Aggressive (β>1.2)']:
                count = (beta_df['Category'] == cat).sum()
                st.metric(cat.split('(')[0].strip(), count)

        with col1:
            beta_sorted = beta_df.sort_values('Beta')
            colors = ['#238636' if b < 0.8 else '#f85149' if b > 1.2 else '#58a6ff'
                      for b in beta_sorted['Beta']]
            fig = go.Figure(go.Bar(x=beta_sorted['Beta'], y=beta_sorted['Symbol'],
                                    orientation='h', marker_color=colors,
                                    hovertemplate='%{y}: β=%{x:.3f}<extra></extra>'))
            fig.add_vline(x=1.0, line_dash='dash', line_color='#8b949e', annotation_text='β=1')
            fig.add_vline(x=0.8, line_dash='dot', line_color='#238636')
            fig.add_vline(x=1.2, line_dash='dot', line_color='#da3633')
            fig.update_layout(paper_bgcolor='#161b22', plot_bgcolor='#0d1117',
                               font_color='#e6edf3', height=480, template='plotly_dark',
                               xaxis_title='Beta vs NIFTY 50',
                               title='Green=Defensive | Blue=Neutral | Red=Aggressive')
            st.plotly_chart(fig, use_container_width=True)

        st.dataframe(
            beta_df.sort_values('Beta', ascending=False)
            [['Symbol', 'Sector', 'Beta', 'Market_Corr', 'Category']],
            use_container_width=True, hide_index=True
        )
