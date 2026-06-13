import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from utils.data_loader import SECTOR_MAP


def render(nifty_df, nifty_stocks, data_loaded):
    st.markdown("# 📈 NIFTY-50 Investment Intelligence Platform")
    st.markdown("*AI-powered decision support for Indian equity markets*")
    st.markdown("---")

    if not data_loaded:
        st.warning("⚠️ No data loaded. Set the correct folder paths in the sidebar and refresh.")
        st.markdown("""
        **Expected folder structure:**
        ```
        data/
        ├── nifty50/          ← individual stock CSVs (e.g. RELIANCE.csv)
        └── index/            ← index CSVs (NIFTY 50.csv, INDIA VIX.csv)
        ```
        Download dataset from: https://www.kaggle.com/datasets/rohanrao/nifty50-stock-market-data
        """)
        return

    # ── KPI row ───────────────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Stocks Tracked", len(nifty_stocks))
    with col2:
        st.metric("Total Records", f"{len(nifty_df):,}")
    with col3:
        date_range = f"{nifty_df['Date'].min().year}–{nifty_df['Date'].max().year}"
        st.metric("Data Range", date_range)
    with col4:
        sectors = nifty_df['Sector'].nunique() if 'Sector' in nifty_df.columns else "—"
        st.metric("Sectors Covered", sectors)

    st.markdown("---")

    col_left, col_right = st.columns([1.6, 1])

    with col_left:
        st.markdown("## Market Overview — Stock Performance")
        # Build a return summary for all stocks
        rows = []
        for sym, df_s in nifty_stocks.items():
            if len(df_s) < 252:
                continue
            ret = df_s['Close'].pct_change().dropna()
            ann_ret = ret.mean() * 252 * 100
            ann_vol = ret.std() * np.sqrt(252) * 100
            total_ret = (df_s['Close'].iloc[-1] / df_s['Close'].iloc[0] - 1) * 100
            rows.append({
                'Symbol': sym,
                'Sector': SECTOR_MAP.get(sym, 'Other'),
                'Ann Return %': round(ann_ret, 2),
                'Volatility %': round(ann_vol, 2),
                'Total Return %': round(total_ret, 1),
            })
        summary_df = pd.DataFrame(rows).sort_values('Ann Return %', ascending=False)

        fig = px.scatter(
            summary_df,
            x='Volatility %', y='Ann Return %',
            color='Sector', text='Symbol',
            title='Risk vs Return — All NIFTY-50 Stocks',
            template='plotly_dark',
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig.update_traces(textposition='top center', textfont_size=9, marker_size=9)
        fig.update_layout(
            paper_bgcolor='#161b22', plot_bgcolor='#0d1117',
            font_color='#e6edf3', height=420,
            legend=dict(orientation='h', yanchor='bottom', y=-0.35, font_size=10),
            title_font_size=13,
        )
        fig.add_hline(y=0, line_dash='dash', line_color='#8b949e', opacity=0.5)
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.markdown("## Sector Breakdown")
        if 'Sector' in nifty_df.columns:
            sec_counts = nifty_df.groupby('Sector')['Symbol'].nunique().sort_values(ascending=False)
            fig2 = px.bar(
                sec_counts,
                orientation='h',
                template='plotly_dark',
                color=sec_counts.values,
                color_continuous_scale='Teal',
                labels={'value': 'Stocks', 'index': ''},
            )
            fig2.update_layout(
                paper_bgcolor='#161b22', plot_bgcolor='#0d1117',
                font_color='#e6edf3', height=420,
                showlegend=False, coloraxis_showscale=False,
                title_font_size=13, margin=dict(l=10, r=10),
            )
            st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    st.markdown("## Top & Bottom Performers")

    col_t, col_b = st.columns(2)
    with col_t:
        st.markdown("#### 🏆 Top 10 by Annualised Return")
        top10 = summary_df.head(10)[['Symbol', 'Sector', 'Ann Return %', 'Total Return %']]
        st.dataframe(top10.style.background_gradient(subset=['Ann Return %'], cmap='Greens'),
                     use_container_width=True, hide_index=True)
    with col_b:
        st.markdown("#### ⚠️ Bottom 10 by Annualised Return")
        bot10 = summary_df.tail(10)[['Symbol', 'Sector', 'Ann Return %', 'Total Return %']]
        st.dataframe(bot10.style.background_gradient(subset=['Ann Return %'], cmap='Reds_r'),
                     use_container_width=True, hide_index=True)
