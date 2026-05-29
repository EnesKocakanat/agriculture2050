"""
AGRI-2050: Faz 3b — Türkiye Geneli Dashboard (Temiz Versiyon)
"""
import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

import json


base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CITY_PATH = os.path.join(base, "geo", "turkey-ilçeler", "cities.json")


def load_cities():
    with open(CITY_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

st.set_page_config(page_title="AGRI-2050 | Türkiye", page_icon="🌾", layout="wide")

@st.cache_data
def load_data():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    def csv(p):
        fp = os.path.join(base, p)
        return pd.read_csv(fp) if os.path.exists(fp) else None
    scores    = csv("data/processed/turkey_il_scores.csv")
    precip    = csv("data/raw/precipitation_iller.csv")
    ndvi      = csv("data/raw/satellite_ndvi_iller.csv")
    forecast  = csv("data/processed/forecast_turkey_2025_2050.csv")
    irrig     = csv("data/processed/irrigation_plan.csv")
    rekolt    = csv("data/processed/tuik_rekolt_clean.csv")
    real_yield= csv("data/processed/turkey_real_yield_scores.csv")
    food_sec  = csv("data/processed/food_security_2024_2050.csv")
    dsi       = csv("data/processed/dsi_sulama_clean.csv")
    dsi_artis = csv("data/processed/dsi_verim_artisi.csv")
    return scores, precip, ndvi, forecast, irrig, rekolt, real_yield, food_sec, dsi, dsi_artis

scores, precip, ndvi, forecast, irrig, rekolt, real_yield, food_sec, dsi, dsi_artis = load_data()

precip_yearly = precip.groupby(['il','year'])['precip_mm'].sum().reset_index()
drought = precip_yearly[precip_yearly['precip_mm'] < 270].groupby('il').size().reset_index(name='kuraklık_yil_gercek')
scores = scores.merge(drought, on='il', how='left')
scores['kuraklık_yil_gercek'] = scores['kuraklık_yil_gercek'].fillna(0).astype(int)

koordinatlar = {
    "İstanbul":(41.01,28.97),"Tekirdağ":(41.45,27.51),"Edirne":(41.68,26.56),
    "Kırklareli":(41.73,27.22),"Balıkesir":(39.64,27.88),"Çanakkale":(40.15,26.41),
    "İzmir":(38.42,27.14),"Aydın":(37.84,27.84),"Denizli":(37.77,29.09),
    "Muğla":(37.21,28.36),"Manisa":(38.61,27.43),"Afyonkarahisar":(38.76,30.54),
    "Kütahya":(39.42,29.98),"Uşak":(38.68,29.41),"Bursa":(40.19,29.06),
    "Eskişehir":(39.78,30.52),"Bilecik":(40.15,29.98),"Kocaeli":(40.85,29.88),
    "Sakarya":(40.69,30.43),"Düzce":(40.84,31.16),"Bolu":(40.73,31.61),
    "Yalova":(40.65,29.27),"Ankara":(39.92,32.85),"Konya":(37.87,32.48),
    "Karaman":(37.18,33.22),"Antalya":(36.89,30.70),"Isparta":(37.76,30.55),
    "Burdur":(37.72,30.29),"Adana":(37.00,35.32),"Mersin":(36.81,34.64),
    "Hatay":(36.20,36.16),"Kahramanmaraş":(37.58,36.94),"Osmaniye":(37.07,36.25),
    "Kırıkkale":(39.85,33.51),"Aksaray":(38.37,34.04),"Niğde":(37.97,34.68),
    "Nevşehir":(38.62,34.71),"Kırşehir":(39.14,34.16),"Kayseri":(38.73,35.49),
    "Sivas":(39.75,37.02),"Yozgat":(39.82,34.81),"Zonguldak":(41.45,31.79),
    "Karabük":(41.20,32.62),"Bartın":(41.63,32.34),"Kastamonu":(41.38,33.78),
    "Çankırı":(40.60,33.61),"Sinop":(42.02,35.15),"Samsun":(41.29,36.33),
    "Tokat":(40.31,36.55),"Çorum":(40.55,34.96),"Amasya":(40.65,35.83),
    "Trabzon":(41.00,39.72),"Ordu":(40.98,37.88),"Giresun":(40.91,38.39),
    "Rize":(41.02,40.52),"Artvin":(41.18,41.82),"Gümüşhane":(40.46,39.48),
    "Erzurum":(39.90,41.27),"Erzincan":(39.75,39.49),"Bayburt":(40.26,40.22),
    "Ağrı":(39.72,43.05),"Kars":(40.61,43.10),"Iğdır":(39.92,44.04),
    "Ardahan":(41.11,42.70),"Malatya":(38.35,38.31),"Elazığ":(38.67,39.22),
    "Bingöl":(38.88,40.50),"Tunceli":(39.11,39.55),"Van":(38.49,43.38),
    "Muş":(38.73,41.49),"Bitlis":(38.40,42.11),"Hakkari":(37.57,43.74),
    "Gaziantep":(37.07,37.38),"Adıyaman":(37.76,38.28),"Kilis":(36.72,37.12),
    "Şanlıurfa":(37.16,38.79),"Diyarbakır":(37.91,40.22),"Mardin":(37.31,40.74),
    "Batman":(37.88,41.13),"Şırnak":(37.52,42.46),"Siirt":(37.93,41.95),
}

st.sidebar.title("🌾 AGRI-2050")
st.sidebar.markdown("**Türkiye Tarım Tahmin Sistemi**")
st.sidebar.markdown("---")
st.sidebar.markdown("🛰 GEE Landsat + Sentinel-2 NDVI")
st.sidebar.markdown("🌧 Open-Meteo ERA5 (2004-2024)")
st.sidebar.markdown("📊 TÜİK Rekolte (2004-2024)")
st.sidebar.markdown("💧 DSİ Sulama (2019-2024)")

st.title("🌾 AGRI-2050 — Türkiye Tarım Analizi")
st.caption("81 il · Gerçek uydu + yağış + TÜİK + DSİ verisi · 2004-2024")
st.divider()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Analiz Edilen İl", "81")
c2.metric("Toplam Ekilen Alan", "16.8M ha")
c3.metric("Ortalama NDVI", f"{scores['ndvi_ort'].mean():.3f}")
c4.metric("Kuraklık Riskli İl", int(scores['kuraklık_yil_gercek'].ge(2).sum()))
st.divider()

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11, tab12 = st.tabs([
    "🗺 Türkiye Haritası", "📊 İl Karşılaştırma", "🌧 Yağış & NDVI",
    "🏆 İl Sıralaması", "🔮 2025-2050 Projeksiyon", "💧 Sulama Planı",
    "🌾 Gerçek Verim", "🍞 Gıda Güvenliği", "🚿 DSİ Sulama", "🌱 Tarım Danışmanı",
    " Hava Tahmini","🤖 AI Tarım Asistanı"
])

import plotly.express as px
import pandas as pd

with tab1:
    st.subheader("🗺 Türkiye Tarım Haritası")

    df_map = scores.copy()
    df_map['lat'] = df_map['il'].map(lambda x: koordinatlar.get(x, (39, 35))[0])
    df_map['lon'] = df_map['il'].map(lambda x: koordinatlar.get(x, (39, 35))[1])

    fig_turkey = px.scatter_mapbox(
        df_map, lat='lat', lon='lon',
        size='tarim_skoru', color='tarim_skoru',
        hover_name='il',
        hover_data={'tarim_skoru':':.3f','ndvi_ort':':.3f','yagis_ort':':.0f','lat':False,'lon':False},
        color_continuous_scale='RdYlGn', size_max=30,
        zoom=5.5, center={"lat": 39.0, "lon": 35.0},
        mapbox_style="carto-positron"
    )
    fig_turkey.update_layout(height=480, margin=dict(l=0,r=0,t=10,b=0))
    st.plotly_chart(fig_turkey, use_container_width=True, key="turkey_map")

    il_listesi = sorted(scores['il'].tolist())
    default_idx = il_listesi.index(st.session_state.get("selected_il", "Konya")) if st.session_state.get("selected_il", "Konya") in il_listesi else 0
    secili_il = st.selectbox("📍 İl seç", il_listesi, index=default_idx, key="tab1_il_sec")
    st.session_state["selected_il"] = secili_il

    st.divider()
    st.subheader(f"🧭 {secili_il} İlçeleri")

    try:
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cities_path = os.path.join(base, "geo", "turkey-ilçeler", "cities.json")
        with open(cities_path, encoding="utf-8") as f:
            cities_data = json.load(f)

        city = next((c for c in cities_data if c["name"] == secili_il), None)

        if city is None:
            st.warning(f"{secili_il} için ilçe verisi bulunamadı.")
        else:
            df_ilce = pd.DataFrame([
                {"ilce": t["name"], "lat": t["latitude"], "lon": t["longitude"]}
                for t in city["towns"]
            ])

            # Tarım skoru ekle
            ilce_skor_path = os.path.join(base, "data/processed/ilce_scores.csv")
            if os.path.exists(ilce_skor_path):
                ilce_skor = pd.read_csv(ilce_skor_path)
                df_ilce = df_ilce.merge(
                    ilce_skor[ilce_skor['il']==secili_il][['ilce','tarim_skoru','yagis_ort','ndvi_ort']],
                    on='ilce', how='left'
                )
                df_ilce['tarim_skoru'] = df_ilce['tarim_skoru'].fillna(df_ilce['tarim_skoru'].mean())
                df_ilce['yagis_ort']   = df_ilce['yagis_ort'].fillna(0)
                df_ilce['ndvi_ort']    = df_ilce['ndvi_ort'].fillna(0)
            else:
                df_ilce['tarim_skoru'] = 5.0
                df_ilce['yagis_ort']   = 0.0
                df_ilce['ndvi_ort']    = 0.0

            il_skor = scores[scores['il']==secili_il]['tarim_skoru'].values
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("İlçe Sayısı", len(df_ilce))
            c2.metric("İl Tarım Skoru", f"{il_skor[0]:.3f}" if len(il_skor) else "—")
            c3.metric("En Yüksek İlçe", df_ilce.nlargest(1,'tarim_skoru')['ilce'].values[0] if 'tarim_skoru' in df_ilce else "—")
            c4.metric("En Düşük İlçe",  df_ilce.nsmallest(1,'tarim_skoru')['ilce'].values[0] if 'tarim_skoru' in df_ilce else "—")

            fig_ilce = px.scatter_mapbox(
                df_ilce, lat="lat", lon="lon",
                hover_name="ilce",
                color="tarim_skoru",
                size="tarim_skoru",
                color_continuous_scale="RdYlGn",
                size_max=20,
                hover_data={
                    'tarim_skoru':':.3f',
                    'yagis_ort':':.0f',
                    'ndvi_ort':':.3f',
                    'lat':False,'lon':False
                },
                zoom=8,
                center={"lat": city["latitude"], "lon": city["longitude"]},
                mapbox_style="carto-positron"
            )
            fig_ilce.update_layout(height=420, margin=dict(l=0,r=0,t=10,b=0))
            st.plotly_chart(fig_ilce, use_container_width=True)

            # İlçe tablosu
            df_tablo = df_ilce[['ilce','tarim_skoru','yagis_ort','ndvi_ort']].copy()
            df_tablo.columns = ['İlçe','Tarım Skoru','Yağış Ort (mm)','NDVI Ort']
            df_tablo = df_tablo.sort_values('Tarım Skoru', ascending=False).reset_index(drop=True)
            df_tablo.index += 1
            st.dataframe(df_tablo.style.format({
                'Tarım Skoru':'{:.3f}',
                'Yağış Ort (mm)':'{:.0f}',
                'NDVI Ort':'{:.3f}'
            }), use_container_width=True, height=300)

    except Exception as e:
        st.error(f"İlçe verisi yüklenemedi: {e}")
with tab2:
    st.subheader("İl Bazlı Karşılaştırma")
    top_n = st.slider("Kaç il?", 10, 81, 20, key="tab2_slider")
    df_top = scores.nlargest(top_n, 'tarim_skoru').sort_values('tarim_skoru')
    fig = px.bar(df_top, x='tarim_skoru', y='il', orientation='h',
                 color='tarim_skoru', color_continuous_scale='RdYlGn',
                 labels={'tarim_skoru':'Tarım Skoru','il':'İl'})
    fig.update_layout(height=max(400, top_n*22), showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    col1, col2 = st.columns(2)
    with col1:
        fig = px.scatter(scores, x='yagis_ort', y='ndvi_ort', color='tarim_skoru',
                         hover_name='il', color_continuous_scale='RdYlGn',
                         labels={'yagis_ort':'Yağış (mm)','ndvi_ort':'NDVI'}, title="Yağış vs NDVI")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.box(scores, y='tarim_skoru', points='all', hover_name='il',
                     title="Tarım Skoru Dağılımı")
        st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("Yağış & NDVI Zaman Serisi")
    secili = st.multiselect("İl seçin", sorted(scores['il'].tolist()),
                             default=["Konya","Rize","Ankara","Samsun"], key="tab3_il")
    if secili:
        p_f = precip[precip['il'].isin(secili)].copy()
        p_m = p_f.groupby(['il','year','month'])['precip_mm'].sum().reset_index()
        p_m['tarih'] = pd.to_datetime(p_m[['year','month']].assign(day=1))
        st.plotly_chart(px.line(p_m, x='tarih', y='precip_mm', color='il',
                                title="Aylık Yağış"), use_container_width=True)
        n_f = ndvi[ndvi['il'].isin(secili)].copy()
        if 'ndvi' in n_f.columns:
            n_m = n_f.groupby(['il','year','month'])['ndvi'].mean().reset_index()
            n_m['tarih'] = pd.to_datetime(n_m[['year','month']].assign(day=1))
            st.plotly_chart(px.line(n_m, x='tarih', y='ndvi', color='il',
                                    title="Aylık NDVI"), use_container_width=True)

with tab4:
    st.subheader("81 İl Tarım Skoru Sıralaması")
    df_rank = scores[['il','tarim_skoru','ndvi_ort','yagis_ort','kuraklık_yil_gercek']].sort_values(
        'tarim_skoru', ascending=False).reset_index(drop=True)
    df_rank.index += 1
    df_rank.columns = ['İl','Tarım Skoru','NDVI Ort','Yağış Ort (mm)','Kuraklık Yılı']
    st.dataframe(df_rank.style.format(
        {'Tarım Skoru':'{:.3f}','NDVI Ort':'{:.3f}','Yağış Ort (mm)':'{:.0f}'}),
        use_container_width=True, height=600)
    csv = df_rank.to_csv(index=True).encode('utf-8-sig')
    st.download_button("⬇ CSV İndir", csv, "turkey_tarim_skorlari.csv", "text/csv", key="tab4_dl")

with tab5:
    st.subheader("2025-2050 İklim Senaryosu Projeksiyonları")
    if forecast is None:
        st.warning("Veri bulunamadı. Önce `python src/phase4_forecast_turkey.py` çalıştırın.")
    else:
        sc_sec = st.selectbox("Senaryo", [
            "optimist_ssp126","orta_ssp245","kotumser_ssp585"
        ], format_func=lambda x: {
            "optimist_ssp126":"İyimser (SSP1-2.6)",
            "orta_ssp245":"Orta (SSP2-4.5)",
            "kotumser_ssp585":"Kötümser (SSP5-8.5)"
        }[x], index=1, key="tab5_sc")
        il_sec = st.multiselect("İl seçin", sorted(forecast['il'].unique().tolist()),
            default=["Konya","Ankara","Samsun","Şanlıurfa","Rize"], key="tab5_il")
        if il_sec:
            df_fc = forecast[(forecast['scenario']==sc_sec) & (forecast['il'].isin(il_sec))]
            fig = px.line(df_fc, x='year', y='tarim_skoru', color='il',
                          labels={'tarim_skoru':'Tarım Skoru','year':'Yıl'},
                          title="Seçili İller — Tarım Skoru Projeksiyonu")
            st.plotly_chart(fig, use_container_width=True)
        st.markdown("---")
        st.markdown("**Tüm İller — 2050 Tahmini Değişim (%)**")
        baz = scores[['il','tarim_skoru']].rename(columns={'tarim_skoru':'baz'})
        df_2050 = forecast[(forecast['year']==2050) & (forecast['scenario']==sc_sec)].merge(baz, on='il')
        df_2050['degisim_pct'] = (df_2050['tarim_skoru'] - df_2050['baz']) / df_2050['baz'] * 100
        fig = px.bar(df_2050.sort_values('degisim_pct'), x='degisim_pct', y='il', orientation='h',
                     color='degisim_pct', color_continuous_scale='RdYlGn',
                     labels={'degisim_pct':'Değişim (%)','il':'İl'},
                     title="2050 Tarım Skoru Değişimi")
        fig.update_layout(height=1800, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

with tab6:
    st.subheader("💧 Akıllı Sulama Öneri Sistemi")
    if irrig is None:
        st.warning("Veri bulunamadı. Önce `python src/phase5_irrigation.py` çalıştırın.")
    else:
        risk_renk = {"KRİTİK":"#ef4444","YÜKSEK":"#f97316","ORTA":"#eab308","DÜŞÜK":"#22c55e"}
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Kritik Risk", int((irrig['risk_seviyesi']=='KRİTİK').sum()))
        c2.metric("Yüksek Risk", int((irrig['risk_seviyesi']=='YÜKSEK').sum()))
        c3.metric("Orta Risk",   int((irrig['risk_seviyesi']=='ORTA').sum()))
        c4.metric("Düşük Risk",  int((irrig['risk_seviyesi']=='DÜŞÜK').sum()))
        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            fig = px.bar(irrig.sort_values('sulama_ihtiyaci_mm').tail(30),
                         x='sulama_ihtiyaci_mm', y='il', orientation='h',
                         color='risk_seviyesi', color_discrete_map=risk_renk,
                         labels={'sulama_ihtiyaci_mm':'Sulama İhtiyacı (mm)','il':'İl'},
                         title="En Yüksek Sulama İhtiyacı (İlk 30 İl)")
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            yontem_ozet = irrig['onerilen_yontem'].value_counts().reset_index()
            yontem_ozet.columns = ['Yöntem','İl Sayısı']
            fig = px.pie(yontem_ozet, values='İl Sayısı', names='Yöntem',
                         title="Önerilen Sulama Yöntemi Dağılımı")
            st.plotly_chart(fig, use_container_width=True)
            fig = px.box(irrig, x='onerilen_yontem', y='su_tasarrufu_pct', color='onerilen_yontem',
                         labels={'su_tasarrufu_pct':'Su Tasarrufu (%)','onerilen_yontem':'Yöntem'},
                         title="Yönteme Göre Su Tasarrufu Potansiyeli")
            st.plotly_chart(fig, use_container_width=True)
        st.divider()
        risk_filtre = st.multiselect("Risk filtresi",
            ["KRİTİK","YÜKSEK","ORTA","DÜŞÜK"], default=["KRİTİK","YÜKSEK"], key="tab6_risk")
        df_f = irrig[irrig['risk_seviyesi'].isin(risk_filtre)] if risk_filtre else irrig
        st.dataframe(df_f[['il','bolge','risk_seviyesi','yagis_mm','sulama_ihtiyaci_mm',
                            'onerilen_yontem','yontem_verimliligi_pct','su_tasarrufu_pct',
                            'yatirim_maliyet_ha_tl']].style.format({
            'yagis_mm':'{:.0f}','sulama_ihtiyaci_mm':'{:.0f}',
            'yontem_verimliligi_pct':'{:.0f}%','su_tasarrufu_pct':'{:.1f}%',
            'yatirim_maliyet_ha_tl':'{:,.0f} ₺'
        }), use_container_width=True, height=400)
        csv = df_f.to_csv(index=False).encode('utf-8-sig')
        st.download_button("⬇ CSV İndir", csv, "sulama_plani.csv", "text/csv", key="tab6_dl")

with tab7:
    st.subheader("🌾 Gerçek Verim Analizi (TÜİK 2004-2024)")
    if rekolt is None:
        st.warning("Veri bulunamadı. Önce `python src/phase0b_tuik_rekolt.py` çalıştırın.")
    else:
        son_yil = rekolt['year'].max()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Veri Yılı", f"2004-{int(son_yil)}")
        c2.metric("İl Sayısı", rekolt['il'].nunique())
        c3.metric(f"Toplam Üretim ({int(son_yil)})",
                  f"{rekolt[rekolt['year']==son_yil]['uretim_ton'].sum()/1e6:.1f}M ton")
        c4.metric(f"Ort. Verim ({int(son_yil)})",
                  f"{rekolt[rekolt['year']==son_yil]['verim_t_ha'].mean():.2f} t/ha")
        st.divider()
        tr_trend = rekolt.groupby('year').agg(
            toplam_ton=('uretim_ton','sum'), ort_verim=('verim_t_ha','mean')
        ).reset_index()
        col1, col2 = st.columns(2)
        with col1:
            fig = px.line(tr_trend, x='year', y='ort_verim', markers=True,
                          labels={'ort_verim':'Ort. Verim (t/ha)','year':'Yıl'},
                          title="Türkiye Geneli Buğday Verimi (2004-2024)")
            fig.update_traces(line_color='#22c55e', marker_size=6)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = px.bar(tr_trend, x='year', y='toplam_ton',
                         labels={'toplam_ton':'Üretim (ton)','year':'Yıl'},
                         title="Türkiye Toplam Buğday Üretimi")
            fig.update_traces(marker_color='#3b82f6')
            st.plotly_chart(fig, use_container_width=True)
        st.divider()
        secili = st.multiselect("İl seçin", sorted(rekolt['il'].unique().tolist()),
            default=["Konya","Edirne","Şanlıurfa","Ankara","Samsun"], key="tab7_il")
        if secili:
            fig = px.line(rekolt[rekolt['il'].isin(secili)], x='year', y='verim_t_ha',
                          color='il', markers=True,
                          labels={'verim_t_ha':'Verim (t/ha)','year':'Yıl'},
                          title="İl Bazlı Verim Trendi (2004-2024)")
            st.plotly_chart(fig, use_container_width=True)
        st.divider()
        if real_yield is not None:
            st.markdown("**Model Tahmini vs Gerçek Verim**")
            df_comp = real_yield[['il','verim_gercek_ort','verim_tahmin_ort']].copy()
            df_comp['hata_pct'] = ((df_comp['verim_tahmin_ort'] - df_comp['verim_gercek_ort'])
                                   / df_comp['verim_gercek_ort'] * 100).abs()
            fig = px.scatter(df_comp, x='verim_gercek_ort', y='verim_tahmin_ort',
                             hover_name='il', color='hata_pct', color_continuous_scale='RdYlGn_r',
                             labels={'verim_gercek_ort':'Gerçek (t/ha)',
                                     'verim_tahmin_ort':'Tahmin (t/ha)','hata_pct':'Hata (%)'},
                             title="Model Tahmin Doğruluğu")
            mn = df_comp[['verim_gercek_ort','verim_tahmin_ort']].min().min()
            mx = df_comp[['verim_gercek_ort','verim_tahmin_ort']].max().max()
            fig.add_shape(type='line', x0=mn, y0=mn, x1=mx, y1=mx,
                          line=dict(color='gray', dash='dash'))
            st.plotly_chart(fig, use_container_width=True)
            df_tablo = real_yield[['il','verim_gercek_ort','verim_tahmin_ort',
                                   'precip_ort','ndvi_ort']].copy()
            df_tablo.columns = ['İl','Gerçek Verim (t/ha)','Tahmin Verim (t/ha)',
                                 'Yağış Ort (mm)','NDVI Ort']
            df_tablo = df_tablo.sort_values('Gerçek Verim (t/ha)', ascending=False).reset_index(drop=True)
            df_tablo.index += 1
            st.dataframe(df_tablo.style.format({
                'Gerçek Verim (t/ha)':'{:.3f}','Tahmin Verim (t/ha)':'{:.3f}',
                'Yağış Ort (mm)':'{:.0f}','NDVI Ort':'{:.3f}'
            }), use_container_width=True, height=500)
            csv = df_tablo.to_csv(index=True).encode('utf-8-sig')
            st.download_button("⬇ CSV İndir", csv, "gercek_verim.csv", "text/csv", key="tab7_dl")

with tab8:
    st.subheader("🍞 Gıda Güvenliği — 2024-2050 Arz-Talep Dengesi")
    if food_sec is None:
        st.warning("Veri bulunamadı. Önce `python src/phase6_food_security.py` çalıştırın.")
    else:
        col_s, col_p = st.columns(2)
        with col_s:
            sc_sec = st.selectbox("İklim Senaryosu", [
                "optimist_ssp126","orta_ssp245","kotumser_ssp585"
            ], format_func=lambda x: {
                "optimist_ssp126":"İyimser (SSP1-2.6)",
                "orta_ssp245":"Orta (SSP2-4.5)",
                "kotumser_ssp585":"Kötümser (SSP5-8.5)"
            }[x], index=1, key="tab8_sc")
        with col_p:
            pol_sec = st.selectbox("Politika Senaryosu", [
                "baz","ileri_sulama","yuksek_teknoloji"
            ], format_func=lambda x: {
                "baz":"Baz (mevcut trend)",
                "ileri_sulama":"İleri Sulama",
                "yuksek_teknoloji":"Yüksek Teknoloji + Sulama"
            }[x], index=0, key="tab8_pol")
        df_fs = food_sec[(food_sec['senaryo']==sc_sec) & (food_sec['politika']==pol_sec)]
        r2050 = df_fs[df_fs['year']==2050].iloc[0]
        kky   = r2050['kky_pct']
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("2050 Üretim", f"{r2050['uretim_ton']/1e6:.1f}M ton")
        c2.metric("2050 Talep",  f"{r2050['ic_talep_ton']/1e6:.1f}M ton")
        c3.metric("2050 Denge",  f"{r2050['denge_ton']/1e6:+.1f}M ton",
                  delta_color="normal" if r2050['denge_ton']>=0 else "inverse")
        c4.metric("Kendi Kendine Yeterlilik", f"{kky:.1f}%")
        st.divider()
        fig = px.line(df_fs, x='year', y=['uretim_ton','kullanilabilir_ton','ic_talep_ton'],
                      labels={'value':'Miktar (ton)','year':'Yıl','variable':'Gösterge'},
                      title="Buğday Arz-Talep Dengesi (2024-2050)",
                      color_discrete_map={
                          'uretim_ton':'#22c55e',
                          'kullanilabilir_ton':'#3b82f6',
                          'ic_talep_ton':'#ef4444'
                      })
        newnames = {'uretim_ton':'Toplam Üretim',
                    'kullanilabilir_ton':'Kullanılabilir (ihracat sonrası)',
                    'ic_talep_ton':'İç Talep'}
        fig.for_each_trace(lambda t: t.update(name=newnames.get(t.name, t.name)))
        st.plotly_chart(fig, use_container_width=True)
        col1, col2 = st.columns(2)
        with col1:
            fig = px.area(df_fs, x='year', y='kky_pct',
                          labels={'kky_pct':'KKY (%)','year':'Yıl'},
                          title="Kendi Kendine Yeterlilik Trendi")
            fig.add_hline(y=100, line_dash="dash", line_color="green", annotation_text="Tam yeterlilik")
            fig.add_hline(y=70,  line_dash="dash", line_color="orange", annotation_text="Kritik eşik")
            fig.update_traces(fill='tozeroy', line_color='#3b82f6')
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            df_pol = food_sec[food_sec['senaryo']==sc_sec]
            fig = px.line(df_pol, x='year', y='kky_pct', color='politika_label',
                          labels={'kky_pct':'KKY (%)','year':'Yıl','politika_label':'Politika'},
                          title="Politika Senaryoları Karşılaştırması",
                          color_discrete_sequence=['#ef4444','#f59e0b','#22c55e'])
            fig.add_hline(y=100, line_dash="dash", line_color="gray")
            st.plotly_chart(fig, use_container_width=True)
        st.divider()
        df_2050 = food_sec[food_sec['year']==2050][
            ['senaryo_label','politika_label','uretim_ton','ic_talep_ton','denge_ton','kky_pct']
        ].copy()
        df_2050.columns = ['Senaryo','Politika','Üretim (ton)','Talep (ton)','Denge (ton)','KKY %']
        df_2050 = df_2050.sort_values('KKY %', ascending=False).reset_index(drop=True)
        df_2050.index += 1
        def kky_rengi(val):
            if val >= 90: return 'background-color: #166534; color: white'
            elif val >= 70: return 'background-color: #854d0e; color: white'
            else: return 'background-color: #7f1d1d; color: white'
        st.dataframe(
            df_2050.style.map(kky_rengi, subset=['KKY %']).format({
                'Üretim (ton)':'{:,.0f}','Talep (ton)':'{:,.0f}',
                'Denge (ton)':'{:+,.0f}','KKY %':'{:.1f}%'
            }), use_container_width=True, height=360)
        st.info("💡 Yüksek teknoloji + sulama politikasıyla iyimser senaryoda %99.6 yeterlilik mümkün.")

with tab9:
    st.subheader("🚿 DSİ Gerçek Sulama Verisi (2019-2024)")
    if dsi is None:
        st.warning("Veri bulunamadı. Önce `python src/phase0c_dsi_integration.py` çalıştırın.")
    else:
        son_yil = int(dsi['year'].max())
        df24 = dsi[dsi['year']==son_yil].dropna(subset=['sulanan_ha'])
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Toplam Sulanan Alan", f"{df24['sulanan_ha'].sum()/1e6:.2f}M ha")
        c2.metric("Ort. Sulama Oranı",   f"{df24['sulama_orani'].mean():.2f}")
        c3.metric("En Fazla Sulanan",     df24.nlargest(1,'sulanan_ha')['il'].values[0])
        c4.metric("Hububat Verim Artışı", "%193 (DSİ 2.4)")
        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            fig = px.bar(df24.nlargest(20,'sulanan_ha').sort_values('sulanan_ha'),
                         x='sulanan_ha', y='il', orientation='h',
                         color='sulanan_ha', color_continuous_scale='Blues',
                         labels={'sulanan_ha':'Sulanan Alan (ha)','il':'İl'},
                         title=f"En Fazla Sulanan 20 İl ({son_yil})")
            fig.update_layout(height=500, yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            if real_yield is not None:
                df_sc = df24.merge(real_yield[['il','verim_gercek_ort']], on='il', how='inner')
                fig = px.scatter(df_sc, x='sulama_orani', y='verim_gercek_ort',
                                 hover_name='il', size='sulanan_ha',
                                 color='verim_gercek_ort', color_continuous_scale='RdYlGn',
                                 labels={'sulama_orani':'Sulama Oranı',
                                         'verim_gercek_ort':'Ort. Verim (t/ha)'},
                                 title="Sulama Oranı vs Buğday Verimi")
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
        st.divider()
        st.markdown("**Türkiye Geneli Yıllık Sulama Trendi**")
        df_trend = dsi.groupby('year').agg(
            toplam_sulanan=('sulanan_ha','sum'),
            ort_oran=('sulama_orani','mean')
        ).reset_index()
        col3, col4 = st.columns(2)
        with col3:
            fig = px.bar(df_trend, x='year', y='toplam_sulanan',
                         labels={'toplam_sulanan':'Toplam Sulanan Alan (ha)','year':'Yıl'},
                         title="Yıllık Toplam Sulanan Alan",
                         color='toplam_sulanan', color_continuous_scale='Blues')
            st.plotly_chart(fig, use_container_width=True)
        with col4:
            fig = px.line(df_trend, x='year', y='ort_oran', markers=True,
                          labels={'ort_oran':'Ort. Sulama Oranı','year':'Yıl'},
                          title="Yıllık Ortalama Sulama Oranı")
            fig.update_traces(line_color='#3b82f6', marker_size=8)
            st.plotly_chart(fig, use_container_width=True)
        st.divider()
        if dsi_artis is not None:
            st.markdown("**Hububat: Susuz vs Sululu Verim (DSİ 2.4, 2013-2024)**")
            fig = px.bar(dsi_artis, x='year', y=['susuz_kg_da','sululu_kg_da'],
                         barmode='group',
                         labels={'value':'Verim (kg/da)','year':'Yıl','variable':'Durum'},
                         title="Sulama ile Hububat Verim Artışı",
                         color_discrete_map={'susuz_kg_da':'#ef4444','sululu_kg_da':'#22c55e'})
            newnames = {'susuz_kg_da':'Susuz','sululu_kg_da':'Sululu'}
            fig.for_each_trace(lambda t: t.update(name=newnames.get(t.name,t.name)))
            st.plotly_chart(fig, use_container_width=True)
        st.divider()
        st.markdown("**İl Bazlı DSİ Sulama Detayları**")
        yil_sec = st.selectbox("Yıl seçin", sorted(dsi['year'].unique(), reverse=True), key="tab9_yil")
        df_tablo = dsi[dsi['year']==yil_sec][['il','sulanan_ha','sulama_orani']].copy()
        if real_yield is not None:
            df_tablo = df_tablo.merge(real_yield[['il','verim_gercek_ort']], on='il', how='left')
        df_tablo = df_tablo.sort_values('sulanan_ha', ascending=False).reset_index(drop=True)
        df_tablo.index += 1
        fmt = {'sulanan_ha':'{:,.0f}','sulama_orani':'{:.3f}'}
        if 'verim_gercek_ort' in df_tablo.columns:
            fmt['verim_gercek_ort'] = '{:.3f}'
        st.dataframe(df_tablo.style.format(fmt), use_container_width=True, height=450)
        csv = df_tablo.to_csv(index=True).encode('utf-8-sig')
        st.download_button("⬇ CSV İndir", csv, f"dsi_sulama_{yil_sec}.csv", "text/csv", key="tab9_dl")
        st.info("💡 **Model Bulgusu:** DSİ sulama oranı, ML modelinde buğday verimi için "
                "en belirleyici özellik olarak öne çıkmıştır (%38.8 önem skoru). "
                "Sulama ile hububat verimi ortalama %193 artmaktadır.")



with tab10:
    st.subheader("🌱 Tarım Danışmanı — İlçe Bazlı Ürün & Sulama Önerisi")

    # Veri yükle
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    @st.cache_data
    def load_advisor_data():
        urun_path  = os.path.join(base_path, "data/processed/ilce_urun_onerileri.csv")
        ilce_path  = os.path.join(base_path, "data/processed/ilce_scores.csv")
        sulama_path= os.path.join(base_path, "data/processed/sulama_detay.csv")
        urun_df    = pd.read_csv(urun_path)  if os.path.exists(urun_path)   else None
        ilce_df    = pd.read_csv(ilce_path)  if os.path.exists(ilce_path)   else None
        sulama_df  = pd.read_csv(sulama_path)if os.path.exists(sulama_path) else None
        return urun_df, ilce_df, sulama_df

    urun_df, ilce_df, sulama_df = load_advisor_data()

    if urun_df is None:
        st.warning("Veri bulunamadı. Önce `python src/phase7_crop_advisor.py` çalıştırın.")
    else:
        # ── Seçim ──
        col_il, col_ilce = st.columns(2)
        with col_il:
            il_sec = st.selectbox("🏙️ İl seçin", sorted(urun_df['il'].unique()), key="danisman_il")
        with col_ilce:
            ilce_listesi = sorted(urun_df[urun_df['il']==il_sec]['ilce'].unique())
            ilce_sec = st.selectbox("📍 İlçe seçin", ilce_listesi, key="danisman_ilce")

        # İlçe iklim verisi
        ilce_row = ilce_df[(ilce_df['il']==il_sec) & (ilce_df['ilce']==ilce_sec)]
        if len(ilce_row) > 0:
            ilce_row = ilce_row.iloc[0]
            yagis = ilce_row.get('yagis_ort', 0)
            ndvi  = ilce_row.get('ndvi_ort', 0)
            skor  = ilce_row.get('tarim_skoru', 0)
        else:
            yagis, ndvi, skor = 0, 0, 0

        # ── İklim Özeti ──
        st.divider()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("📍 İlçe", ilce_sec)
        c2.metric("🌧 Yıllık Yağış", f"{yagis:.0f} mm")
        c3.metric("🛰 NDVI Ortalaması", f"{ndvi:.3f}")
        c4.metric("🌾 Tarım Skoru", f"{skor:.2f}/10")

        # İklim yorumu
        if yagis < 300:
            iklim_yorum = "🔴 Çok kurak — sulama zorunlu, kuraklığa dayanıklı ürünler tercih edin"
        elif yagis < 500:
            iklim_yorum = "🟡 Yarı kurak — sulama gerekli, su tasarruflu yöntemler önerilir"
        elif yagis < 900:
            iklim_yorum = "🟢 Yeterli yağış — çoğu ürün yetişir, sulama destekleyici"
        else:
            iklim_yorum = "🔵 Yağışlı iklim — Karadeniz ürünleri (çay, fındık) için ideal"
        st.info(iklim_yorum)

        st.divider()

        # ── Ürün Önerileri ──
        st.markdown("### 🏆 Bu İlçeye En Uygun Ürünler")
        df_sec = urun_df[(urun_df['il']==il_sec) & (urun_df['ilce']==ilce_sec)].copy()

        if len(df_sec) == 0:
            st.warning("Bu ilçe için öneri bulunamadı.")
        else:
            # Uygunluk bar chart
            fig_urun = px.bar(
                df_sec.sort_values('uygunluk_pct'),
                x='uygunluk_pct', y='urun', orientation='h',
                color='uygunluk_pct', color_continuous_scale='RdYlGn',
                labels={'uygunluk_pct':'Uygunluk (%)', 'urun':'Ürün'},
                title=f"{ilce_sec} ({il_sec}) — Ürün Uygunluk Skoru",
                text='uygunluk_pct'
            )
            fig_urun.update_traces(texttemplate='%{text:.0f}%', textposition='outside')
            fig_urun.update_layout(height=350, showlegend=False, xaxis_range=[0,110])
            st.plotly_chart(fig_urun, use_container_width=True)

            # Ürün kartları
            st.markdown("### 📋 Detaylı Ürün Bilgisi")
            cols = st.columns(min(len(df_sec), 3))
            for idx, (_, row) in enumerate(df_sec.iterrows()):
                with cols[idx % 3]:
                    uygunluk_renk = "🟢" if row['uygunluk_pct'] >= 80 else ("🟡" if row['uygunluk_pct'] >= 60 else "🔴")
                    st.markdown(f"""
**{row['emoji']} {row['urun']}**
- {uygunluk_renk} Uygunluk: **%{row['uygunluk_pct']:.0f}**
- 📂 Kategori: {row['kategori']}
- 💰 Kâr Potansiyeli: {row['kar_potansiyel']}
- 🌱 Ekim: {row['ekim_aylari']}
- 🌾 Hasat: {row['hasat_aylari']}
- 💧 Sulama: {row['sulama_yontem']}
- 📝 {row['bakim']}
""")
                    st.markdown("---")

        st.divider()

        # ── Sulama Planı ──
        st.markdown("### 💧 Sulama Yöntemi Karşılaştırması")

        if sulama_df is not None:
            col1, col2, col3 = st.columns(3)
            for i, (_, s) in enumerate(sulama_df.iterrows()):
                with [col1, col2, col3][i % 3]:
                    if s['yontem'] == 'Damla':
                        renk = "🟢"
                        tavsiye = "✅ Bu ilçe için önerilen"
                    elif s['yontem'] == 'Yağmurlama':
                        renk = "🟡"
                        tavsiye = "⚠️ Orta verimli seçenek"
                    else:
                        renk = "🔴"
                        tavsiye = "❌ Önerilmez (su israfı)"

                    st.markdown(f"""
**{renk} {s['yontem']} Sulama**

{tavsiye}
- ⚡ Verimlilik: {s['verimlilik']}
- 💦 Su Tasarrufu: {s['su_tasarrufu']}
- 💰 Kurulum: {s['maliyet_ha']} /ha
- 🌿 Uygun Ürünler: {s['uygun_urunler']}
- ✅ {s['avantaj']}
""")

        st.divider()

        # ── Aylık Sulama Takvimi ──
        st.markdown("### 📅 Yıllık Sulama & Bakım Takvimi")

        en_iyi_urun = df_sec.iloc[0] if len(df_sec) > 0 else None
        if en_iyi_urun is not None:
            aylar = ["Oca","Şub","Mar","Nis","May","Haz","Tem","Ağu","Eyl","Eki","Kas","Ara"]
            # Sulama yoğunluğu (yaz ayları daha yoğun)
            sulama_yogunluk = [20,25,40,60,80,100,100,90,70,45,25,15]
            # Nem ihtiyacı
            nem_ihtiyac = [30,30,50,70,85,95,95,90,75,55,35,25]

            df_takvim = pd.DataFrame({
                "Ay": aylar,
                "Sulama İhtiyacı (%)": sulama_yogunluk,
                "Nem İhtiyacı (%)": nem_ihtiyac
            })

            fig_takvim = px.bar(df_takvim, x="Ay",
                                y=["Sulama İhtiyacı (%)", "Nem İhtiyacı (%)"],
                                barmode='group',
                                title=f"{en_iyi_urun['urun']} — Aylık Sulama Takvimi",
                                color_discrete_map={
                                    "Sulama İhtiyacı (%)": "#3b82f6",
                                    "Nem İhtiyacı (%)": "#22c55e"
                                })
            fig_takvim.update_layout(height=320)
            st.plotly_chart(fig_takvim, use_container_width=True)

        # ── 2050 Projeksiyonu ──
        st.divider()
        st.markdown("### 🔮 2050 İklim Projeksiyonu")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
**{ilce_sec} için 2050 Senaryoları:**

| Senaryo | Yağış Değişimi | Risk |
|---------|---------------|------|
| İyimser (SSP1-2.6) | -%8 | Düşük |
| Orta (SSP2-4.5) | -%15 | Orta |
| Kötümser (SSP5-8.5) | -%30 | Yüksek |
""")
        with col2:
            yagis_2050 = {
                "İyimser": yagis * 0.92,
                "Orta": yagis * 0.85,
                "Kötümser": yagis * 0.70,
            }
            df_2050 = pd.DataFrame({
                "Senaryo": list(yagis_2050.keys()),
                "Yağış (mm)": list(yagis_2050.values())
            })
            fig_2050 = px.bar(df_2050, x="Senaryo", y="Yağış (mm)",
                              color="Senaryo",
                              color_discrete_sequence=["#22c55e","#f59e0b","#ef4444"],
                              title=f"2050 Tahmini Yağış — {ilce_sec}")
            fig_2050.add_hline(y=yagis, line_dash="dash",
                               annotation_text=f"Güncel: {yagis:.0f}mm")
            fig_2050.update_layout(height=280, showlegend=False)
            st.plotly_chart(fig_2050, use_container_width=True)

        st.success(f"💡 **Tavsiye:** {ilce_sec} ilçesinde en güvenli uzun vadeli yatırım "
                   f"**{df_sec.iloc[0]['emoji']} {df_sec.iloc[0]['urun']}** olup "
                   f"**{df_sec.iloc[0]['sulama_yontem']}** sulama yöntemi önerilmektedir.")
        



with tab11:
    st.subheader("🌤 Hava Durumu & Tarımsal Uyarılar")

    import requests as req
    from datetime import datetime, timedelta

    @st.cache_data(ttl=1800)  # 30 dakikada bir güncelle
    def fetch_weather(lat, lon):
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat,
            "longitude": lon,
            "current": [
                "temperature_2m","relative_humidity_2m","precipitation",
                "wind_speed_10m","weather_code","apparent_temperature"
            ],
            "daily": [
                "temperature_2m_max","temperature_2m_min","precipitation_sum",
                "wind_speed_10m_max","weather_code","et0_fao_evapotranspiration"
            ],
            "timezone": "Europe/Istanbul",
            "forecast_days": 7
        }
        resp = req.get(url, params=params, timeout=15)
        return resp.json()

    def weather_code_to_text(code):
        codes = {
            0: ("☀️", "Açık"), 1: ("🌤", "Az bulutlu"), 2: ("⛅", "Parçalı bulutlu"),
            3: ("☁️", "Bulutlu"), 45: ("🌫", "Sisli"), 48: ("🌫", "Kırağılı sis"),
            51: ("🌦", "Hafif çisenti"), 53: ("🌦", "Çisenti"), 55: ("🌧", "Yoğun çisenti"),
            61: ("🌧", "Hafif yağmur"), 63: ("🌧", "Yağmur"), 65: ("🌧", "Şiddetli yağmur"),
            71: ("🌨", "Hafif kar"), 73: ("🌨", "Kar"), 75: ("❄️", "Yoğun kar"),
            80: ("🌦", "Sağanak"), 81: ("🌧", "Kuvvetli sağanak"), 82: ("⛈", "Çok kuvvetli sağanak"),
            95: ("⛈", "Fırtına"), 96: ("⛈", "Dolu fırtınası"), 99: ("⛈", "Şiddetli dolu"),
        }
        return codes.get(code, ("🌡", "Bilinmiyor"))

    # İlçe seç
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    @st.cache_data
    def load_cities_for_weather():
        import json
        path = os.path.join(base_path, "geo", "turkey-ilçeler", "cities.json")
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    cities_weather = load_cities_for_weather()

    col_il, col_ilce = st.columns(2)
    with col_il:
        il_hava = st.selectbox("🏙️ İl seçin", sorted([c["name"] for c in cities_weather]), key="hava_il")
    with col_ilce:
        city_obj = next((c for c in cities_weather if c["name"] == il_hava), None)
        ilce_listesi_hava = [t["name"] for t in city_obj["towns"]] if city_obj else []
        ilce_hava = st.selectbox("📍 İlçe seçin", sorted(ilce_listesi_hava), key="hava_ilce")

    # Koordinat bul
    town_obj = next((t for t in city_obj["towns"] if t["name"] == ilce_hava), None) if city_obj else None
    if town_obj is None:
        st.warning("İlçe koordinatı bulunamadı.")
    else:
        lat_hava = town_obj["latitude"]
        lon_hava = town_obj["longitude"]

        try:
            weather = fetch_weather(lat_hava, lon_hava)
            current = weather["current"]
            daily   = weather["daily"]

            # ── Anlık Hava ──
            st.divider()
            icon, durum = weather_code_to_text(current["weather_code"])
            st.markdown(f"### {icon} {ilce_hava} — Anlık Hava Durumu")

            c1,c2,c3,c4,c5,c6 = st.columns(6)
            c1.metric("🌡 Sıcaklık", f"{current['temperature_2m']:.1f}°C")
            c2.metric("🌡 Hissedilen", f"{current['apparent_temperature']:.1f}°C")
            c3.metric("💧 Nem", f"%{current['relative_humidity_2m']:.0f}")
            c4.metric("🌧 Yağış", f"{current['precipitation']:.1f} mm")
            c5.metric("💨 Rüzgar", f"{current['wind_speed_10m']:.1f} km/h")
            c6.metric("☁️ Durum", durum)

            # ── Tarımsal Uyarılar ──
            st.divider()
            st.markdown("### ⚠️ Tarımsal Uyarılar")

            uyarilar = []
            temps_min = daily["temperature_2m_min"]
            precip_7  = daily["precipitation_sum"]
            toplam_yagis = sum(precip_7)

            # Don riski
            don_gunler = [i for i, t in enumerate(temps_min) if t <= 2]
            if don_gunler:
                gunler_str = ", ".join([
                    (datetime.now() + timedelta(days=d)).strftime("%d %b")
                    for d in don_gunler
                ])
                uyarilar.append(("🥶", "DON RİSKİ", f"Önümüzdeki günlerde don bekleniyor: {gunler_str}. Hassas bitkilerinizi koruyun!", "error"))

            # Kuraklık
            if toplam_yagis < 5:
                uyarilar.append(("🏜️", "KURAKLIK UYARISI", "7 gün boyunca yağış beklenmıyor. Sulama yapmanız önerilir.", "warning"))

            # Aşırı yağış
            asiri_yagis = [i for i, p in enumerate(precip_7) if p > 30]
            if asiri_yagis:
                gunler_str = ", ".join([
                    (datetime.now() + timedelta(days=d)).strftime("%d %b")
                    for d in asiri_yagis
                ])
                uyarilar.append(("🌊", "AŞIRI YAĞIŞ", f"Yoğun yağış bekleniyor: {gunler_str}. Drenaj kontrolü yapın.", "warning"))

            # Yüksek sıcaklık
            temps_max = daily["temperature_2m_max"]
            sicak_gunler = [i for i, t in enumerate(temps_max) if t > 35]
            if sicak_gunler:
                uyarilar.append(("🌡️", "YÜKSEK SICAKLIK", f"35°C üzeri sıcaklık bekleniyor. Sabah/akşam sulama yapın.", "warning"))

            if not uyarilar:
                st.success("✅ Bu hafta tarımsal açıdan olumsuz bir hava koşulu beklenmıyor.")
            else:
                for emoji, baslik, mesaj, tip in uyarilar:
                    if tip == "error":
                        st.error(f"{emoji} **{baslik}:** {mesaj}")
                    else:
                        st.warning(f"{emoji} **{baslik}:** {mesaj}")

            # ── 7 Günlük Tahmin ──
            st.divider()
            st.markdown("### 📅 7 Günlük Hava Tahmini")

            gunler = []
            for i in range(7):
                tarih = (datetime.now() + timedelta(days=i)).strftime("%a %d %b")
                icon_d, durum_d = weather_code_to_text(daily["weather_code"][i])
                gunler.append({
                    "Gün": tarih,
                    "Durum": f"{icon_d} {durum_d}",
                    "Max °C": daily["temperature_2m_max"][i],
                    "Min °C": daily["temperature_2m_min"][i],
                    "Yağış (mm)": daily["precipitation_sum"][i],
                    "Rüzgar (km/h)": daily["wind_speed_10m_max"][i],
                    "ET₀ (mm)": round(daily["et0_fao_evapotranspiration"][i], 1),
                })

            df_gunler = pd.DataFrame(gunler)
            st.dataframe(df_gunler.set_index("Gün"), use_container_width=True, height=280)

            # Sıcaklık grafiği
            col1, col2 = st.columns(2)
            with col1:
                fig_temp = px.line(df_gunler, x="Gün",
                                   y=["Max °C", "Min °C"],
                                   title="7 Günlük Sıcaklık Tahmini",
                                   color_discrete_map={"Max °C":"#ef4444","Min °C":"#3b82f6"},
                                   markers=True)
                fig_temp.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="Don sınırı")
                fig_temp.update_layout(height=300)
                st.plotly_chart(fig_temp, use_container_width=True)

            with col2:
                fig_yagis = px.bar(df_gunler, x="Gün", y="Yağış (mm)",
                                   title="7 Günlük Yağış Tahmini",
                                   color="Yağış (mm)",
                                   color_continuous_scale="Blues")
                fig_yagis.add_hline(y=30, line_dash="dash", line_color="red",
                                    annotation_text="Aşırı yağış eşiği")
                fig_yagis.update_layout(height=300)
                st.plotly_chart(fig_yagis, use_container_width=True)

            # ── Sulama Tavsiyesi ──
            st.divider()
            st.markdown("### 💧 Bu Haftaki Sulama Tavsiyesi")

            et0_toplam = sum(daily["et0_fao_evapotranspiration"])
            net_su = et0_toplam - toplam_yagis

            c1, c2, c3 = st.columns(3)
            c1.metric("💦 Beklenen Yağış", f"{toplam_yagis:.1f} mm")
            c2.metric("🌿 Evapotranspirasyon", f"{et0_toplam:.1f} mm")
            c3.metric("💧 Net Su İhtiyacı", f"{max(0, net_su):.1f} mm",
                      delta_color="inverse" if net_su > 0 else "normal")

            if net_su <= 0:
                st.success("✅ Bu hafta yağış yeterli, sulama yapmana gerek yok.")
            elif net_su < 20:
                st.info(f"ℹ️ Hafif su açığı var ({net_su:.1f}mm). Hafif sulama yeterli.")
            elif net_su < 50:
                st.warning(f"⚠️ Orta su açığı ({net_su:.1f}mm). Haftada 2 kez sulama önerilir.")
            else:
                st.error(f"🚨 Ciddi su açığı ({net_su:.1f}mm). Günlük sulama gerekli!")

        except Exception as e:
            st.error(f"Hava durumu alınamadı: {e}")
            st.info("İnternet bağlantınızı kontrol edin.")


with tab12:
    st.subheader("🤖 AI Tarım Asistanı")
    st.caption("Gerçek veriye dayalı tarım soruları sorun — il, ilçe, ürün, sulama, iklim...")

    from groq import Groq

    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    @st.cache_data
    def load_ai_data():
        ilce_path = os.path.join(base_path, "data/processed/ilce_scores.csv")
        urun_path = os.path.join(base_path, "data/processed/ilce_urun_onerileri.csv")

        ilce_df = pd.read_csv(ilce_path) if os.path.exists(ilce_path) else None
        urun_df = pd.read_csv(urun_path) if os.path.exists(urun_path) else None

        return ilce_df, urun_df

    ilce_df_ai, urun_df_ai = load_ai_data()

    col1, col2 = st.columns(2)

    with col1:
        il_ai = st.selectbox(
            "🏙️ İl seçin",
            sorted(ilce_df_ai['il'].unique()),
            key="ai_il"
        )

    with col2:
        ilce_listesi_ai = sorted(
            ilce_df_ai[ilce_df_ai['il'] == il_ai]['ilce'].unique()
        )

        ilce_ai = st.selectbox(
            "📍 İlçe seçin",
            ilce_listesi_ai,
            key="ai_ilce"
        )

    ilce_row_ai = ilce_df_ai[
        (ilce_df_ai['il'] == il_ai) &
        (ilce_df_ai['ilce'] == ilce_ai)
    ]

    urun_row_ai = (
        urun_df_ai[
            (urun_df_ai['il'] == il_ai) &
            (urun_df_ai['ilce'] == ilce_ai)
        ]
        if urun_df_ai is not None else None
    )

    if len(ilce_row_ai) > 0:

        r = ilce_row_ai.iloc[0]

        yagis = r.get('yagis_ort', 0)
        ndvi = r.get('ndvi_ort', 0)
        skor = r.get('tarim_skoru', 0)

        urun_ozet = ""

        if urun_row_ai is not None and len(urun_row_ai) > 0:

            for _, u in urun_row_ai.head(3).iterrows():

                urun_ozet += (
                    f"- {u['urun']}: "
                    f"%{u['uygunluk_pct']:.0f} uygun, "
                    f"{u['sulama_yontem']}\n"
                )

        context = f"""
Sen AGRI-2050 Türkiye tarım danışmanısın.
Kısa ve pratik cevap ver, Türkçe yaz.

{il_ai}/{ilce_ai} verileri:

- Yıllık yağış: {yagis:.0f}mm
- NDVI: {ndvi:.3f}
- Tarım skoru: {skor:.1f}/10
- İklim:
{"Kurak" if yagis < 300 else
"Yarı kurak" if yagis < 500 else
"Yeterli" if yagis < 900 else
"Yağışlı"}

En uygun ürünler:
{urun_ozet if urun_ozet else "Veri yok"}

Kurallar:
- Veriye dayan
- Kısa ol
- Her cevap sonunda 1 pratik ipucu ver
"""

    else:

        context = (
            f"Türkiye tarım danışmanısın. "
            f"{il_ai}/{ilce_ai} için genel bilgiyle cevap ver. "
            f"Kısa ve pratik ol."
        )

    if "ai_messages" not in st.session_state:
        st.session_state.ai_messages = []

    st.markdown("**💡 Örnek sorular:**")

    ornek_sorular = [
        f"{ilce_ai}'da ne ekeyim?",
        "En iyi sulama yöntemi?",
        "2050'de iklim nasıl olur?",
        "Domates yetişir mi?",
        "Tarım skoru nedir?"
    ]

    cols_o = st.columns(5)

    for i, soru in enumerate(ornek_sorular):

        if cols_o[i].button(
            soru,
            key=f"ornek_{i}",
            use_container_width=True
        ):

            st.session_state.ai_messages.append({
                "role": "user",
                "content": soru
            })

            st.rerun()

    st.divider()

    for msg in st.session_state.ai_messages:

        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input(
        f"{ilce_ai} hakkında tarım sorusu sorun..."
    ):

        st.session_state.ai_messages.append({
            "role": "user",
            "content": prompt
        })

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):

            with st.spinner("Analiz ediliyor..."):

                try:

                    client_ai = Groq(
                        api_key=st.secrets["GROQ_API_KEY"]
                    )

                    gecmis = ""

                    for msg in st.session_state.ai_messages[:-1][-4:]:

                        rol = (
                            "Kullanıcı"
                            if msg["role"] == "user"
                            else "Asistan"
                        )

                        gecmis += (
                            f"{rol}: {msg['content']}\n"
                        )

                    full_prompt = f"""
{context}

{gecmis}

Kullanıcı: {prompt}

Asistan:
"""

                    completion = (
                        client_ai.chat.completions.create(
                            model="llama-3.1-8b-instant",

                            messages=[
                                {
                                    "role": "system",
                                    "content": context
                                },
                                {
                                    "role": "user",
                                    "content": full_prompt
                                }
                            ],

                            temperature=0.7,
                            max_tokens=700
                        )
                    )

                    cevap = (
                        completion
                        .choices[0]
                        .message
                        .content
                    )

                    st.markdown(cevap)

                    st.session_state.ai_messages.append({
                        "role": "assistant",
                        "content": cevap
                    })

                except Exception as e:

                    if "429" in str(e):

                        st.warning(
                            "⏳ API limiti doldu, biraz bekleyin."
                        )

                    else:

                        st.error(f"API hatası: {e}")

    if st.session_state.ai_messages:

        if st.button(
            "🗑️ Sohbeti Temizle",
            key="ai_clear"
        ):

            st.session_state.ai_messages = []

            st.rerun()


st.divider()
st.caption("AGRI-2050 | GEE Sentinel-2/Landsat + Open-Meteo ERA5 + TÜİK + DSİ | 2004-2024")