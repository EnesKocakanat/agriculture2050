"""
=============================================================
AGRI-2050: Faz 3 - Streamlit Dashboard
=============================================================
Çalıştırma: streamlit run phase3_dashboard.py
=============================================================
"""
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import os, sys, json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

sys.path.insert(0, os.path.dirname(__file__))

# ─── Sayfa ayarları ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AGRI-2050 | Konya Ovası Tahmin Sistemi",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }
.main { background: #0a0f0d; }

.metric-card {
    background: linear-gradient(135deg, #0d1f1a 0%, #0a1a15 100%);
    border: 1px solid #1e4a38;
    border-radius: 12px; padding: 20px;
    text-align: center; margin-bottom: 10px;
}
.metric-value {
    font-size: 2.2rem; font-weight: 700;
    color: #4ade80; font-family: 'JetBrains Mono', monospace;
}
.metric-label { font-size: 0.85rem; color: #6b9980; margin-top: 4px; }
.metric-delta { font-size: 0.9rem; color: #86efac; margin-top: 6px; }

.section-header {
    font-size: 1.1rem; font-weight: 600; color: #d1fae5;
    border-left: 3px solid #4ade80;
    padding-left: 12px; margin: 24px 0 16px;
}
.alert-box {
    background: #1a0f0a; border: 1px solid #b45309;
    border-radius: 8px; padding: 12px 16px;
    color: #fcd34d; font-size: 0.9rem; margin: 8px 0;
}
.success-box {
    background: #0a1a0f; border: 1px solid #16a34a;
    border-radius: 8px; padding: 12px 16px;
    color: #86efac; font-size: 0.9rem; margin: 8px 0;
}
.stSelectbox label, .stSlider label { color: #9ca3af !important; }
</style>
""", unsafe_allow_html=True)

# ─── Veri yükleme / üretme ───────────────────────────────────────────────────
@st.cache_data
def load_or_generate_data():
    data_dir  = os.path.join(os.path.dirname(__file__), "..", "data")
    model_dir = os.path.join(os.path.dirname(__file__), "..", "models")

    if not os.path.exists(f"{data_dir}/merged_dataset.csv"):
        from src.phase1_data_fetch import run_data_pipeline
        run_data_pipeline(data_dir)
    if not os.path.exists(f"{data_dir}/forecast_2025_2030.csv"):
        from phase2_ml_model import run_training_pipeline
        run_training_pipeline(data_dir, model_dir)

    historical = pd.read_csv(f"{data_dir}/merged_dataset.csv")
    forecast   = pd.read_csv(f"{data_dir}/forecast_2025_2030.csv")
    yields_all = pd.read_csv(f"{data_dir}/crop_yields.csv")
    ndvi_ts    = pd.read_csv(f"{data_dir}/satellite_ndvi.csv")

    metrics = {}
    if os.path.exists(f"{model_dir}/metrics.json"):
        with open(f"{model_dir}/metrics.json") as f:
            metrics = json.load(f)

    return historical, forecast, yields_all, ndvi_ts, metrics


# ─── Lojistik hesaplama ───────────────────────────────────────────────────────
def compute_logistics(total_prod_t: int, scenario: str):
    TRUCK_CAPACITY   = 25    # ton
    SILO_CAPACITY    = 50000 # ton
    LOSS_RATE        = 0.04  # %4 hasat kaybı

    net_prod  = total_prod_t * (1 - LOSS_RATE)
    trucks    = int(np.ceil(net_prod / TRUCK_CAPACITY))
    silos     = int(np.ceil(net_prod / SILO_CAPACITY))
    export_t  = int(net_prod * 0.35)
    domestic_t= int(net_prod * 0.65)

    urgency = "YÜKSEKRİSK" if scenario == "kotumser_ssp585" else ("ORTA" if scenario == "orta_ssp245" else "NORMAL")
    return {
        "net_prod_t": int(net_prod),
        "trucks_needed": trucks,
        "silos_needed": silos,
        "export_t": export_t,
        "domestic_t": domestic_t,
        "urgency": urgency,
    }


# ─── Plotly grafikleri ────────────────────────────────────────────────────────
SCENARIO_COLORS = {
    "optimist_ssp126":   "#22c55e",
    "orta_ssp245":       "#f59e0b",
    "kotumser_ssp585":   "#ef4444",
}
SCENARIO_LABELS = {
    "optimist_ssp126":   "İyimser (SSP1-2.6)",
    "orta_ssp245":       "Orta (SSP2-4.5)",
    "kotumser_ssp585":   "Kötümser (SSP5-8.5)",
}

def plot_yield_forecast(historical, forecast):
    fig = go.Figure()

    # Tarihsel
    fig.add_trace(go.Scatter(
        x=historical["year"], y=historical["yield_t_ha"],
        name="Gerçek Rekolte", mode="lines+markers",
        line=dict(color="#60a5fa", width=2.5),
        marker=dict(size=6, color="#93c5fd"),
    ))

    # Senaryo bantları
    for sc_key, color in SCENARIO_COLORS.items():
        sc_df = forecast[forecast["scenario"] == sc_key].sort_values("year")
        if sc_df.empty: continue

        fig.add_trace(go.Scatter(
            x=pd.concat([sc_df["year"], sc_df["year"][::-1]]),
            y=pd.concat([sc_df["yield_upper_80"], sc_df["yield_lower_80"][::-1]]),
           fill="toself", fillcolor=f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.08)",
            line=dict(color="rgba(0,0,0,0)"),
            showlegend=False, hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=sc_df["year"], y=sc_df["yield_t_ha"],
            name=SCENARIO_LABELS[sc_key], mode="lines+markers",
            line=dict(color=color, width=2, dash="dash"),
            marker=dict(size=7),
        ))

    # 2027 dikey çizgi
    fig.add_vline(x=2027, line_dash="dot", line_color="#a78bfa",
                  annotation_text="Hedef: 2027", annotation_position="top")

    fig.update_layout(
        paper_bgcolor="#0a0f0d", plot_bgcolor="#0d1812",
        font=dict(color="#d1fae5", family="Space Grotesk"),
        legend=dict(bgcolor="#0d1812", bordercolor="#1e4a38"),
        xaxis=dict(gridcolor="#1a2e25", title="Yıl"),
        yaxis=dict(gridcolor="#1a2e25", title="Rekolte (ton/hektar)"),
        margin=dict(l=40, r=40, t=20, b=40), height=380,
    )
    return fig


def plot_ndvi_heatmap(ndvi_ts):
    pivot = ndvi_ts.pivot_table(index="month", columns="year", values="ndvi", aggfunc="mean")
    months = ["Oca","Şub","Mar","Nis","May","Haz","Tem","Ağu","Eyl","Eki","Kas","Ara"]

    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=[str(c) for c in pivot.columns],
        y=months[:len(pivot.index)],
        colorscale=[[0,"#1a0f0a"],[0.3,"#854d0e"],[0.6,"#16a34a"],[1,"#4ade80"]],
        colorbar=dict(title="NDVI", tickfont=dict(color="#d1fae5")),
        hoverongaps=False,
    ))
    fig.update_layout(
        paper_bgcolor="#0a0f0d", plot_bgcolor="#0d1812",
        font=dict(color="#d1fae5", family="Space Grotesk"),
        xaxis=dict(title="Yıl"), yaxis=dict(title=""),
        margin=dict(l=40, r=40, t=10, b=40), height=300,
    )
    return fig


def plot_crop_comparison(yields_all):
    latest = yields_all[yields_all["year"] >= 2020].groupby("crop")["yield_t_ha"].mean().reset_index()
    crop_names = {
        "bugday":"Buğday","arpa":"Arpa","seker_pancari":"Şeker P.",
        "kuru_fasulye":"Fasulye","elma":"Elma"
    }
    latest["crop_tr"] = latest["crop"].map(crop_names)
    colors = ["#4ade80","#22d3ee","#f59e0b","#a78bfa","#f472b6"]

    fig = go.Figure(go.Bar(
        x=latest["crop_tr"], y=latest["yield_t_ha"],
        marker_color=colors,
        text=[f"{v:.2f}" for v in latest["yield_t_ha"]],
        textposition="outside", textfont=dict(color="#d1fae5"),
    ))
    fig.update_layout(
        paper_bgcolor="#0a0f0d", plot_bgcolor="#0d1812",
        font=dict(color="#d1fae5", family="Space Grotesk"),
        xaxis=dict(gridcolor="#1a2e25"),
        yaxis=dict(gridcolor="#1a2e25", title="t/ha"),
        margin=dict(l=30, r=30, t=10, b=40), height=280,
        showlegend=False,
    )
    return fig


def plot_logistics_sunburst(logistics):
    fig = go.Figure(go.Sunburst(
        labels=["Toplam Üretim","İhracat","Yurt İçi","Kayıp"],
        parents=["","Toplam Üretim","Toplam Üretim","Toplam Üretim"],
        values=[
            logistics["net_prod_t"] + int(logistics["net_prod_t"] * 0.04/0.96),
            logistics["export_t"],
            logistics["domestic_t"],
            int(logistics["net_prod_t"] * 0.04/0.96),
        ],
        marker=dict(colors=["#0d1812","#4ade80","#22d3ee","#ef4444"]),
        textfont=dict(color="#d1fae5"),
    ))
    fig.update_layout(
        paper_bgcolor="#0a0f0d",
        font=dict(color="#d1fae5"),
        margin=dict(l=0, r=0, t=0, b=0), height=260,
    )
    return fig


# ─── ANA UYGULAMA ─────────────────────────────────────────────────────────────
def main():
    # Başlık
    st.markdown("""
    <div style="background:linear-gradient(90deg,#0d1812,#0a2e1e);
                border:1px solid #1e4a38;border-radius:12px;
                padding:24px 32px;margin-bottom:24px;">
        <h1 style="color:#4ade80;margin:0;font-size:1.9rem;font-weight:700;">
            🌾 AGRI-2050 | Konya Ovası Tahmin Sistemi
        </h1>
        <p style="color:#6b9980;margin:6px 0 0;font-size:0.95rem;">
            Uydu Verisi + Makine Öğrenmesi → 2027 Tahmini Rekolte & Lojistik Planı
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Veri yükle
    with st.spinner("Veriler hazırlanıyor..."):
        historical, forecast, yields_all, ndvi_ts, metrics = load_or_generate_data()

    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### ⚙️ Parametreler")
        selected_scenario = st.selectbox(
            "İklim Senaryosu",
            options=list(SCENARIO_LABELS.keys()),
            format_func=lambda x: SCENARIO_LABELS[x],
            index=1,
        )
        target_year = st.slider("Hedef Yıl", 2025, 2030, 2027)
        area_adjust = st.slider("Ekim Alanı Değişimi (%)", -20, +20, 0)

        st.markdown("---")
        st.markdown("### 📊 Model Performansı")
        if metrics:
            st.metric("Test MAE", f"{metrics.get('test_mae','N/A')} t/ha")
            st.metric("Test R²",  f"{metrics.get('test_r2','N/A')}")
        st.markdown("---")
        st.markdown("### 🛰️ Veri Kaynakları")
        st.markdown("""
        - Sentinel-2 (NDVI)
        - CHIRPS v2.0 (Yağış)
        - ERA5 (Sıcaklık)
        - FAO/TÜİK (Rekolte)
        """)

    # ── Hedef yıl tahmini ────────────────────────────────────────────────────
    sc_forecast = forecast[
        (forecast["scenario"] == selected_scenario) &
        (forecast["year"] == target_year)
    ]

    if sc_forecast.empty:
        st.error("Seçilen yıl/senaryo için veri bulunamadı.")
        return

    row = sc_forecast.iloc[0]
    adj_factor   = 1 + area_adjust / 100
    yield_val    = row["yield_t_ha"]
    total_prod   = int(row["total_prod_t"] * adj_factor)
    logistics    = compute_logistics(total_prod, selected_scenario)

    # Tarihsel ortalama kıyaslama
    hist_avg_yield = historical["yield_t_ha"].mean()
    delta_pct      = (yield_val - hist_avg_yield) / hist_avg_yield * 100

    # ── Üst metrikler ────────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{yield_val:.2f}</div>
            <div class="metric-label">Tahmini Verim (t/ha)</div>
            <div class="metric-delta">{delta_pct:+.1f}% tarihsel ort.</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{total_prod/1e6:.2f}M</div>
            <div class="metric-label">Toplam Üretim (ton)</div>
            <div class="metric-delta">{int(580000*adj_factor/1000):,}K hektar</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{logistics['trucks_needed']:,}</div>
            <div class="metric-label">Kamyon İhtiyacı</div>
            <div class="metric-delta">{logistics['silos_needed']} silo kapasitesi</div>
        </div>""", unsafe_allow_html=True)
    with col4:
        urgency_color = "#ef4444" if logistics["urgency"]=="YÜKSEKRİSK" else ("#f59e0b" if logistics["urgency"]=="ORTA" else "#4ade80")
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color:{urgency_color};font-size:1.4rem;">{logistics['urgency']}</div>
            <div class="metric-label">Risk Seviyesi</div>
            <div class="metric-delta">{SCENARIO_LABELS[selected_scenario]}</div>
        </div>""", unsafe_allow_html=True)

    # ── İkinci satır: Rekolte grafiği + NDVI heatmap ─────────────────────────
    st.markdown('<div class="section-header">📈 Rekolte Projeksiyonu & Senaryo Analizi</div>', unsafe_allow_html=True)
    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.plotly_chart(plot_yield_forecast(historical, forecast),
                        use_container_width=True, key="yield_chart")
    with col_right:
        st.markdown('<div style="color:#9ca3af;font-size:0.85rem;margin-bottom:8px;">🛰️ NDVI Isı Haritası (2010-2024)</div>', unsafe_allow_html=True)
        st.plotly_chart(plot_ndvi_heatmap(ndvi_ts),
                        use_container_width=True, key="ndvi_heatmap")

    # ── Üçüncü satır: Ürün karşılaştırması + Lojistik ───────────────────────
    st.markdown('<div class="section-header">🚛 Lojistik Planlama & Ürün Dağılımı</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns([2, 2, 2])

    with c1:
        st.markdown('<div style="color:#9ca3af;font-size:0.85rem;margin-bottom:8px;">🌿 Ürün Bazlı Verim (2020-2024 Ort.)</div>', unsafe_allow_html=True)
        st.plotly_chart(plot_crop_comparison(yields_all),
                        use_container_width=True, key="crop_bar")

    with c2:
        st.markdown('<div style="color:#9ca3af;font-size:0.85rem;margin-bottom:8px;">📦 Üretim Dağılımı</div>', unsafe_allow_html=True)
        st.plotly_chart(plot_logistics_sunburst(logistics),
                        use_container_width=True, key="sunburst")

    with c3:
        st.markdown('<div style="color:#9ca3af;font-size:0.85rem;margin-bottom:8px;">📋 Lojistik Plan Özeti</div>', unsafe_allow_html=True)
        if logistics["urgency"] == "YÜKSEKRİSK":
            st.markdown(f'<div class="alert-box">⚠️ <b>Kuraklık Uyarısı:</b> Sulama kapasitesi %40 artırılmalı.</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="alert-box">⚠️ Erken hasat planlaması gerekebilir.</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="success-box">✅ Üretim hedefleri karşılanabilir.</div>', unsafe_allow_html=True)

        st.markdown(f"""
        | Kalem | Değer |
        |-------|-------|
        | Net Üretim | {logistics['net_prod_t']:,} ton |
        | İhracat | {logistics['export_t']:,} ton |
        | Yurt içi | {logistics['domestic_t']:,} ton |
        | Kamyon | {logistics['trucks_needed']:,} adet |
        | Silo | {logistics['silos_needed']} ünite |
        """)

    # ── 2027 Detaylı Projeksiyon Tablosu ─────────────────────────────────────
    st.markdown('<div class="section-header">📊 2025-2030 Tüm Senaryo Karşılaştırması</div>', unsafe_allow_html=True)

    pivot = forecast.pivot_table(
        index="year",
        columns="scenario_label",
        values="yield_t_ha"
    ).round(3).reset_index()
    pivot.columns.name = None
    pivot = pivot.rename(columns={"year": "Yıl"})
    st.dataframe(pivot, use_container_width=True, hide_index=True)

    # Footer
    st.markdown("""
    <div style="text-align:center;color:#2d5a44;font-size:0.8rem;margin-top:32px;padding:16px;
                border-top:1px solid #0d2a1e;">
        AGRI-2050 | Uydu Verisi + ML Tahmin Sistemi | Konya Ovası Pilot Bölgesi<br>
        Veri: Sentinel-2, CHIRPS v2.0, ERA5 | Model: Ensemble (RF + GBM + Ridge)
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
