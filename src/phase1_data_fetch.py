"""
=============================================================
AGRI-2050: Faz 1 - Uydu Veri Seti Çekme Modülü
=============================================================
Kaynak: Google Earth Engine API + Kaggle + Sentetik veri üretimi
Bölge : Konya Ovası (37.87°N, 32.48°E)
=============================================================
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# ─── Konya Ovası koordinatları ───────────────────────────────────────────────
KONYA_BBOX = {
    "north": 38.20, "south": 37.50,
    "east":  33.00, "west":  31.90,
    "name":  "Konya Ovası, Türkiye"
}

# ─── GEE API entegrasyonu (mock + gerçek örnek) ──────────────────────────────
class GoogleEarthEngineClient:
    """
    Google Earth Engine Python API istemcisi.
    Gerçek kullanım için: pip install earthengine-api && ee.Authenticate()

    Bu modül:
    - NDVI (Normalized Difference Vegetation Index)
    - NDWI (Normalized Difference Water Index)
    - LST  (Land Surface Temperature)
    - Yağış verisini çeker.
    """

    def __init__(self, authenticated=False):
        self.authenticated = authenticated
        self._init_mock_params()

    def _init_mock_params(self):
        """Konya Ovası'na özgü gerçekçi parametre aralıkları"""
        self.params = {
            "ndvi":       {"mu": 0.62, "sigma": 0.12, "seasonal_amp": 0.25},
            "ndwi":       {"mu": 0.18, "sigma": 0.08, "seasonal_amp": 0.10},
            "lst_summer": {"mu": 34.5, "sigma": 4.2},
            "lst_winter": {"mu": 2.1,  "sigma": 6.8},
            "precip_ann": {"mu": 312,  "sigma": 55},   # mm/yıl
        }

    def get_ndvi_timeseries(self, start_year=2010, end_year=2024):
        """
        Sentinel-2 / Landsat 8 NDVI zaman serisi simülasyonu.
        
        Gerçek GEE kodu örneği:
        ─────────────────────────────────────────────────────
        import ee
        ee.Initialize()
        
        geometry = ee.Geometry.Rectangle([31.9, 37.5, 33.0, 38.2])
        s2 = ee.ImageCollection('COPERNICUS/S2_SR') \\
               .filterBounds(geometry) \\
               .filterDate('2020-01-01', '2024-12-31') \\
               .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10))
        
        def add_ndvi(image):
            ndvi = image.normalizedDifference(['B8','B4']).rename('NDVI')
            return image.addBands(ndvi)
        
        ndvi_collection = s2.map(add_ndvi)
        monthly_ndvi = ndvi_collection.select('NDVI').mean()
        ─────────────────────────────────────────────────────
        """
        records = []
        p = self.params

        for year in range(start_year, end_year + 1):
            for month in range(1, 13):
                # Mevsimsel NDVI (buğday takvimi: Ekim ekim, Temmuz hasat)
                season_phase = np.sin(2 * np.pi * (month - 3) / 12)
                ndvi_base    = p["ndvi"]["mu"] + p["ndvi"]["seasonal_amp"] * season_phase

                # İklim değişikliği trendi (hafif azalma: -0.003/yıl)
                climate_trend = -0.003 * (year - 2010)

                # ENSO etkisi (3 yıllık döngü)
                enso_effect = 0.02 * np.sin(2 * np.pi * year / 3)

                ndvi_val = (ndvi_base + climate_trend + enso_effect +
                            np.random.normal(0, p["ndvi"]["sigma"] * 0.3))

                records.append({
                    "date":       f"{year}-{month:02d}-15",
                    "year":       year,
                    "month":      month,
                    "ndvi":       round(np.clip(ndvi_val, 0.1, 0.95), 4),
                    "ndwi":       round(np.clip(
                                      p["ndwi"]["mu"] + p["ndwi"]["seasonal_amp"] * season_phase
                                      + np.random.normal(0, 0.03), -0.5, 0.5), 4),
                    "lst_celsius": round(
                                      p["lst_summer"]["mu"] * max(0, season_phase) +
                                      p["lst_winter"]["mu"] * max(0, -season_phase) +
                                      np.random.normal(0, 2.5), 2),
                    "cloud_cover_pct": round(np.random.uniform(5, 35), 1),
                    "source":     "Sentinel-2/Landsat8 (simulated)"
                })

        df = pd.DataFrame(records)
        df["date"] = pd.to_datetime(df["date"])
        return df

    def get_precipitation_data(self, start_year=2010, end_year=2024):
        """CHIRPS / ERA5 yağış verisi (aylık mm)"""
        records = []
        for year in range(start_year, end_year + 1):
            # Konya yağış dağılımı: kış/ilkbahar ağırlıklı
            monthly_pattern = [35, 32, 38, 42, 30, 18, 8, 6, 12, 25, 38, 40]
            drought_year = np.random.choice([0, 1], p=[0.75, 0.25])  # %25 kuraklık yılı

            for month in range(1, 13):
                base_precip = monthly_pattern[month - 1]
                variation   = np.random.normal(0, base_precip * 0.35)
                drought_adj = -base_precip * 0.40 if drought_year else 0

                records.append({
                    "year":          year,
                    "month":         month,
                    "precip_mm":     round(max(0, base_precip + variation + drought_adj), 1),
                    "drought_year":  bool(drought_year),
                    "source":        "CHIRPS v2.0 (simulated)"
                })

        return pd.DataFrame(records)


# ─── Kaggle veri seti entegrasyonu ──────────────────────────────────────────
class KaggleDataClient:
    """
    Kaggle API ile tarımsal veri seti indirme.
    
    Kullanım:
        pip install kaggle
        kaggle datasets download -d USERNAME/DATASET_NAME
    
    Önerilen veri setleri:
        - crop-yield-prediction-dataset
        - world-food-production
        - global-food-agriculture-statistics
    """

    RECOMMENDED_DATASETS = [
        {"slug": "patelris/crop-yield-prediction-dataset",
         "desc": "Küresel tarla ürünleri rekolte tahmini"},
        {"slug": "unitednations/food-agriculture-organization-of-the-united-nations",
         "desc": "FAO dünya gıda üretim istatistikleri"},
        {"slug": "akshaychavan9977/global-food-agriculture-statistics",
         "desc": "1961-2023 küresel tarım verileri"},
    ]

    def generate_konya_yield_data(self, start_year=2005, end_year=2024):
        """
        Konya Ovası gerçekçi rekolte verisi üretimi.
        
        Ürünler (Konya'nın ana tarım ürünleri):
        - Buğday (dominant)
        - Arpa
        - Şeker pancarı
        - Kuru fasulye
        - Elma
        """
        np.random.seed(42)
        records = []

        # Konya Ovası gerçek rekolte referans değerleri (ton/hektar)
        crops = {
            "bugday":        {"base_yield": 3.2,  "area_ha": 580000, "trend": 0.025},
            "arpa":          {"base_yield": 2.8,  "area_ha": 210000, "trend": 0.018},
            "seker_pancari": {"base_yield": 62.0, "area_ha": 45000,  "trend": 0.035},
            "kuru_fasulye":  {"base_yield": 1.4,  "area_ha": 38000,  "trend": 0.010},
            "elma":          {"base_yield": 18.5, "area_ha": 12000,  "trend": 0.015},
        }

        for year in range(start_year, end_year + 1):
            yrs_elapsed   = year - start_year
            drought_year  = year in [2007, 2014, 2018, 2021]
            flood_year    = year in [2009, 2016]
            frost_year    = year in [2011, 2020]

            for crop, specs in crops.items():
                trend_mult   = 1 + specs["trend"] * yrs_elapsed / 10
                drought_mult = 0.72 if drought_year  else 1.0
                flood_mult   = 0.88 if flood_year    else 1.0
                frost_mult   = 0.91 if frost_year    else 1.0
                noise_mult   = np.random.normal(1.0, 0.06)

                yield_t_ha = (specs["base_yield"] * trend_mult *
                              drought_mult * flood_mult * frost_mult * noise_mult)
                total_prod = yield_t_ha * specs["area_ha"]

                records.append({
                    "year":          year,
                    "crop":          crop,
                    "yield_t_ha":    round(yield_t_ha, 3),
                    "area_ha":       specs["area_ha"],
                    "total_prod_t":  round(total_prod),
                    "drought_year":  drought_year,
                    "flood_year":    flood_year,
                    "frost_year":    frost_year,
                    "region":        "Konya",
                    "country":       "Turkey",
                })

        return pd.DataFrame(records)


# ─── Ana veri pipeline ───────────────────────────────────────────────────────
def run_data_pipeline(output_dir="data"):
    os.makedirs(output_dir, exist_ok=True)
    print("=" * 60)
    print("AGRI-2050 | FAZ 1: Veri Toplama Pipeline")
    print("=" * 60)

    # GEE Uydu Verileri
    print("\n[1/3] Google Earth Engine uydu verileri çekiliyor...")
    gee = GoogleEarthEngineClient()
    ndvi_df   = gee.get_ndvi_timeseries(2010, 2024)
    precip_df = gee.get_precipitation_data(2010, 2024)
    print(f"    ✓ NDVI zaman serisi: {len(ndvi_df)} kayıt")
    print(f"    ✓ Yağış verisi     : {len(precip_df)} kayıt")

    # Kaggle Rekolte Verileri
    print("\n[2/3] Konya Ovası rekolte verileri yükleniyor...")
    kaggle  = KaggleDataClient()
    yield_df = kaggle.generate_konya_yield_data(2005, 2024)
    print(f"    ✓ Rekolte verisi   : {len(yield_df)} kayıt | {yield_df['crop'].nunique()} ürün")

    # Verileri birleştir
    print("\n[3/3] Veri setleri birleştiriliyor...")
    ndvi_yearly = (ndvi_df.groupby("year")
                   .agg(ndvi_mean=("ndvi","mean"),
                        ndvi_peak=("ndvi","max"),
                        lst_mean =("lst_celsius","mean"))
                   .reset_index())

    precip_yearly = (precip_df.groupby("year")
                     .agg(precip_annual_mm=("precip_mm","sum"),
                          drought_year=("drought_year","first"))
                     .reset_index())

    bugday_df = yield_df[yield_df["crop"] == "bugday"].copy()
    merged_df = (bugday_df
                 .merge(ndvi_yearly,   on="year")
                 .merge(precip_yearly, on="year"))

    # Kaydet
    ndvi_df.to_csv(f"{output_dir}/satellite_ndvi.csv",     index=False)
    precip_df.to_csv(f"{output_dir}/precipitation.csv",    index=False)
    yield_df.to_csv(f"{output_dir}/crop_yields.csv",       index=False)
    merged_df.to_csv(f"{output_dir}/merged_dataset.csv",   index=False)

    print(f"\n    ✓ satellite_ndvi.csv   → {len(ndvi_df)} satır")
    print(f"    ✓ precipitation.csv    → {len(precip_df)} satır")
    print(f"    ✓ crop_yields.csv      → {len(yield_df)} satır")
    print(f"    ✓ merged_dataset.csv   → {len(merged_df)} satır")
    print("\n[✓] Faz 1 tamamlandı.\n")

    return {"ndvi": ndvi_df, "precip": precip_df,
            "yields": yield_df, "merged": merged_df}


if __name__ == "__main__":
    data = run_data_pipeline()
