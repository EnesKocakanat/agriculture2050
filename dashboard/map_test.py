import streamlit as st
import pandas as pd
import plotly.express as px
import json

st.set_page_config(layout="wide")

st.title("🗺️ Turkey Map - TEST VERSION")

# -----------------------
# SIMPLE DATA
# -----------------------
df = pd.DataFrame({
    "il": ["Konya", "Ankara", "İzmir", "Antalya", "İstanbul", "Bursa", "Adana"],
    "ndvi": [0.62, 0.58, 0.71, 0.66, 0.75, 0.69, 0.60],
    "rainfall": [320, 410, 680, 900, 800, 750, 600],
    "agri_score": [78, 72, 85, 88, 90, 83, 77]
})

# -----------------------
# GEOJSON LOAD (SIMPLE PATH)
# -----------------------
with open("geo/tr.json", "r", encoding="utf-8") as f:
    turkey_geo = json.load(f)

# -----------------------
# MAP
# -----------------------
fig = px.choropleth(
    df,
    geojson=turkey_geo,
    locations="il",
    featureidkey="properties.name",
    color="agri_score",
    hover_name="il",
    hover_data=["ndvi", "rainfall", "agri_score"],
    color_continuous_scale="YlGn"
)

fig.update_geos(fitbounds="locations", visible=False)

st.plotly_chart(fig, use_container_width=True)

# -----------------------
# SIMPLE PANEL
# -----------------------
st.divider()

city = st.selectbox("İl seç", df["il"], key="city")

data = df[df["il"] == city].iloc[0]

st.subheader(f"📍 {city}")
st.write("NDVI:", data["ndvi"])
st.write("Yağış:", data["rainfall"])
st.write("Tarım Skoru:", data["agri_score"])
st.write(turkey_geo["features"][0]["properties"])