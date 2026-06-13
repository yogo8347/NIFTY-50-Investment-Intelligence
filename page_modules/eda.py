import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from utils.data_loader import engineer_features, SECTOR_MAP


def render(nifty_df, nifty_stocks, nifty50_index, india_vix, nifty_bank, data_loaded):
    st.markdown("# 📊 EDA & Market Insights")
    st.markdown("---")

    if not data_loaded:
        st.warning("Load data first via the sidebar.")
        return

    symbols = sorted(nifty_stocks.keys())
    tabs = st.tabs(["📈 Price History", "🌡️ VIX vs Market", "🔗 Correlations", "📉 Volatility", "🧰 Technical Indicators"])

    # ── Tab 1: Price History ──────────────────────────────────────────────────
    with tabs[0]:
        st.markdown("### Stock Price History")
        col1, col2 = st.columns([2, 1])
        with col1:
            selected = st.multiselect("Select stocks to compare", symbols,
                                      default=['RELIANCE', 'HDFCBANK', 'TCS', 'INFY'][:min(4, len(symbols))])
        with col2:
            normalize = st.checkbox("Normalize to 100 (compare growth)", value=True)

        if selected:
            fig = go.Figure()
            colors = px.colors.qualitative.Set2
            for i, sym in enumerate(selected):
                df_s = nifty_stocks[sym]
                y = df_s['Close']
                if normalize:
                    y = y / y.iloc[0] * 100
                fig.add_trace(go.Scatter(
                    x=df_s['Date'], y=y, name=sym,
                    line=dict(width=1.8, color=colors[i % len(colors)]),
                    hovertemplate=f'<b>{sym}</b><br>Date: %{{x}}<br>{"Index" if normalize else "₹"}: %{{y:.2f}}<extra></extra>'
                ))
            fig.add_vline(x='2020-03-23', line_dash='dash', line_color='#f85149',
                          annotation_text='COVID Crash', annotation_position='top right',
                          annotation_font_color='#f85149')
            fig.update_layout(
                paper_bgcolor='#161b22', plot_bgcolor='#0d1117',
                font_color='#e6edf3', height=420,
                yaxis_title='Normalized Index (base=100)' if normalize else 'Close Price (₹)',
                hovermode='x unified', template='plotly_dark',
                legend=dict(orientation='h', yanchor='bottom', y=-0.25),
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Sector-Wise Average Annualised Returns")
        rows = []
        for sym, df_s in nifty_stocks.items():
            if len(df_s) < 252:
                continue
            ret = df_s['Close'].pct_change().dropna()
            rows.append({'Symbol': sym, 'Sector': SECTOR_MAP.get(sym, 'Other'),
                          'Ann_Return': ret.mean() * 252 * 100,
                          'Ann_Vol': ret.std() * np.sqrt(252) * 100})
        stats_df = pd.DataFrame(rows)
        sec_avg = stats_df.groupby('Sector')['Ann_Return'].mean().sort_values()
        colors_bar = ['#238636' if v > 0 else '#da3633' for v in sec_avg]
        fig2 = go.Figure(go.Bar(x=sec_avg.values, y=sec_avg.index, orientation='h',
                                 marker_color=colors_bar,
                                 hovertemplate='%{y}: %{x:.1f}%<extra></extra>'))
        fig2.update_layout(paper_bgcolor='#161b22', plot_bgcolor='#0d1117',
                            font_color='#e6edf3', height=320,
                            xaxis_title='Avg Annualised Return (%)', template='plotly_dark')
        st.plotly_chart(fig2, use_container_width=True)

    # ── Tab 2: VIX ────────────────────────────────────────────────────────────
    with tabs[1]:
        st.markdown("### INDIA VIX vs NIFTY 50 Index")
        if nifty50_index is None or india_vix is None:
            st.info("Place 'NIFTY 50.csv' and 'INDIA VIX.csv' in your index folder.")
        else:
            vix_nifty = pd.merge(
                india_vix[['Date', 'Close']].rename(columns={'Close': 'VIX'}),
                nifty50_index[['Date', 'Close']].rename(columns={'Close': 'NIFTY50'}),
                on='Date', how='inner'
            )
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                row_heights=[0.6, 0.4], vertical_spacing=0.05)
            fig.add_trace(go.Scatter(x=vix_nifty['Date'], y=vix_nifty['NIFTY50'],
                                      fill='tozeroy', fillcolor='rgba(88,166,255,0.08)',
                                      line=dict(color='#58a6ff', width=1.5), name='NIFTY 50'), row=1, col=1)
            fig.add_trace(go.Scatter(x=vix_nifty['Date'], y=vix_nifty['VIX'],
                                      fill='tozeroy', fillcolor='rgba(248,81,73,0.12)',
                                      line=dict(color='#f85149', width=1.2), name='INDIA VIX'), row=2, col=1)
            fig.add_hline(y=20, row=2, col=1, line_dash='dot', line_color='orange',
                          annotation_text='VIX=20 Fear', annotation_font_color='orange')
            fig.add_hline(y=30, row=2, col=1, line_dash='dot', line_color='#f85149',
                          annotation_text='VIX=30 Panic', annotation_font_color='#f85149')
            fig.update_layout(paper_bgcolor='#161b22', plot_bgcolor='#0d1117',
                               font_color='#e6edf3', height=480, template='plotly_dark',
                               hovermode='x unified')
            st.plotly_chart(fig, use_container_width=True)

            vix_nifty['NIFTY_Ret'] = vix_nifty['NIFTY50'].pct_change()
            vix_nifty['VIX_Chg'] = vix_nifty['VIX'].pct_change()
            corr = vix_nifty[['NIFTY_Ret', 'VIX_Chg']].corr().iloc[0, 1]
            st.info(f"📊 Correlation between NIFTY daily return and VIX change: **{corr:.4f}** — "
                    f"{'Negative as expected: VIX spikes when market falls.' if corr < 0 else 'Positive correlation observed.'}")

    # ── Tab 3: Correlations ───────────────────────────────────────────────────
    with tabs[2]:
        st.markdown("### Stock Return Correlation Matrix")
        corr_stocks = st.multiselect("Select stocks for correlation analysis", symbols,
                                      default=symbols[:min(12, len(symbols))])
        if len(corr_stocks) < 2:
            st.info("Select at least 2 stocks.")
        else:
            price_pivot = nifty_df[nifty_df['Symbol'].isin(corr_stocks)].pivot_table(
                index='Date', columns='Symbol', values='Close')
            ret_pivot = price_pivot.pct_change().dropna(how='all')
            ret_pivot = ret_pivot.dropna(axis=1, thresh=int(0.6 * len(ret_pivot)))
            corr_mat = ret_pivot.corr()
            fig = px.imshow(corr_mat, color_continuous_scale='RdBu_r', zmin=-0.5, zmax=1,
                             text_auto='.2f', aspect='auto', template='plotly_dark')
            fig.update_layout(paper_bgcolor='#161b22', font_color='#e6edf3', height=500)
            fig.update_traces(textfont_size=9)
            st.plotly_chart(fig, use_container_width=True)

            corr_stack = corr_mat.where(np.triu(np.ones(corr_mat.shape), k=1).astype(bool)).stack()
            corr_stack = corr_stack.reset_index()
            corr_stack.columns = ['Stock A', 'Stock B', 'Correlation']
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Most Correlated Pairs**")
                st.dataframe(corr_stack.nlargest(5, 'Correlation').round(4), hide_index=True, use_container_width=True)
            with col2:
                st.markdown("**Least Correlated Pairs**")
                st.dataframe(corr_stack.nsmallest(5, 'Correlation').round(4), hide_index=True, use_container_width=True)

    # ── Tab 4: Volatility ─────────────────────────────────────────────────────
    with tabs[3]:
        st.markdown("### Rolling Volatility Analysis")
        vol_sym = st.selectbox("Select stock", symbols, key='vol_sym')
        window = st.slider("Rolling window (days)", 10, 60, 20)

        df_v = nifty_stocks[vol_sym].copy()
        df_v['Return'] = df_v['Close'].pct_change()
        df_v['RollingVol'] = df_v['Return'].rolling(window).std() * np.sqrt(252) * 100
        df_v = df_v.dropna()

        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.55, 0.45])
        fig.add_trace(go.Scatter(x=df_v['Date'], y=df_v['Close'],
                                  fill='tozeroy', fillcolor='rgba(88,166,255,0.06)',
                                  line=dict(color='#58a6ff', width=1.5), name='Price'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_v['Date'], y=df_v['RollingVol'],
                                  fill='tozeroy', fillcolor='rgba(255,166,0,0.1)',
                                  line=dict(color='#ffa500', width=1.2),
                                  name=f'{window}d Rolling Vol %'), row=2, col=1)
        fig.update_layout(paper_bgcolor='#161b22', plot_bgcolor='#0d1117',
                           font_color='#e6edf3', height=480, template='plotly_dark',
                           hovermode='x unified')
        st.plotly_chart(fig, use_container_width=True)

        # Return distribution
        col1, col2 = st.columns(2)
        with col1:
            fig2 = px.histogram(df_v['Return'] * 100, nbins=80, template='plotly_dark',
                                 color_discrete_sequence=['#58a6ff'],
                                 labels={'value': 'Daily Return %'}, title='Return Distribution')
            fig2.add_vline(x=0, line_dash='dash', line_color='#8b949e')
            fig2.update_layout(paper_bgcolor='#161b22', plot_bgcolor='#0d1117',
                                font_color='#e6edf3', height=280, showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)
        with col2:
            st.markdown("**Distribution Statistics**")
            ret_stats = df_v['Return'].describe()
            metrics = {
                'Skewness': round(df_v['Return'].skew(), 4),
                'Kurtosis': round(df_v['Return'].kurtosis(), 4),
                'Best Day %': round(df_v['Return'].max() * 100, 2),
                'Worst Day %': round(df_v['Return'].min() * 100, 2),
                'Positive Days %': round((df_v['Return'] > 0).mean() * 100, 1),
            }
            for k, v in metrics.items():
                st.metric(k, v)

    # ── Tab 5: Technical Indicators ───────────────────────────────────────────
    with tabs[4]:
        st.markdown("### Technical Indicator Chart")
        ti_sym = st.selectbox("Select stock", symbols, key='ti_sym')
        lookback = st.slider("Lookback period (trading days)", 60, 504, 252)

        df_ti_raw = nifty_stocks[ti_sym].copy()
        if len(df_ti_raw) < 60:
            st.warning("Not enough data for this stock.")
            return
        df_ti = engineer_features(df_ti_raw).tail(lookback)

        fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                             row_heights=[0.45, 0.2, 0.18, 0.17],
                             vertical_spacing=0.03)

        # Price + Bollinger + MAs
        fig.add_trace(go.Scatter(x=df_ti['Date'], y=df_ti['Close'], name='Price',
                                  line=dict(color='#58a6ff', width=2)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_ti['Date'], y=df_ti['MA_20'], name='MA20',
                                  line=dict(color='#ffa500', width=1, dash='dot')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_ti['Date'], y=df_ti['MA_50'], name='MA50',
                                  line=dict(color='#3fb950', width=1, dash='dot')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_ti['Date'], y=df_ti['BB_upper'],
                                  line=dict(color='#8b949e', width=0.8, dash='dash'),
                                  name='BB Upper', showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_ti['Date'], y=df_ti['BB_lower'],
                                  fill='tonexty', fillcolor='rgba(139,148,158,0.08)',
                                  line=dict(color='#8b949e', width=0.8, dash='dash'),
                                  name='BB Bands'), row=1, col=1)

        # Volume
        fig.add_trace(go.Bar(x=df_ti['Date'], y=df_ti['Volume'] / 1e6,
                              marker_color='#58a6ff', opacity=0.5, name='Volume (M)'), row=2, col=1)

        # RSI
        fig.add_trace(go.Scatter(x=df_ti['Date'], y=df_ti['RSI_14'],
                                  line=dict(color='#ff7b72', width=1.5), name='RSI(14)'), row=3, col=1)
        fig.add_hline(y=70, row=3, col=1, line_dash='dot', line_color='#da3633', line_width=1)
        fig.add_hline(y=30, row=3, col=1, line_dash='dot', line_color='#238636', line_width=1)

        # MACD
        macd_colors = ['#238636' if v >= 0 else '#da3633' for v in df_ti['MACD_hist']]
        fig.add_trace(go.Bar(x=df_ti['Date'], y=df_ti['MACD_hist'],
                              marker_color=macd_colors, name='MACD Hist', opacity=0.7), row=4, col=1)
        fig.add_trace(go.Scatter(x=df_ti['Date'], y=df_ti['MACD'],
                                  line=dict(color='#58a6ff', width=1), name='MACD'), row=4, col=1)
        fig.add_trace(go.Scatter(x=df_ti['Date'], y=df_ti['MACD_signal'],
                                  line=dict(color='#ffa500', width=1), name='Signal'), row=4, col=1)

        fig.update_layout(paper_bgcolor='#161b22', plot_bgcolor='#0d1117',
                           font_color='#e6edf3', height=700, template='plotly_dark',
                           hovermode='x unified',
                           legend=dict(orientation='h', yanchor='bottom', y=-0.12, font_size=10))
        fig.update_yaxes(row=3, col=1, range=[0, 100])
        st.plotly_chart(fig, use_container_width=True)
