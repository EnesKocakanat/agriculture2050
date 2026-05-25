"""
AGRI-2050: Faz 1c (Tam) — GEE NDVI 2004-2024
=============================================
2004-2016: Landsat 7 ETM+
2017-2024: Sentinel-2 SR
Çalıştırma: python src/phase1c_gee_ndvi_full.py
=============================================
"""

import os, time
import ee
import pandas as pd

ee.Initialize(project='agri2050')

ILLER = {
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


def fetch_ndvi_month(point, year, month):
    start = f"{year}-{month:02d}-01"
    end   = f"{year}-{month:02d}-28"

    if year >= 2017:
        # Sentinel-2
        col = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
               .filterBounds(point).filterDate(start, end)
               .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
               .map(lambda img: img.normalizedDifference(['B8','B4']).rename('NDVI'))
               .mean())
        source = 'Sentinel-2'
    else:
        # Landsat 7 ETM+
        col = (ee.ImageCollection('LANDSAT/LE07/C02/T1_L2')
               .filterBounds(point).filterDate(start, end)
               .filter(ee.Filter.lt('CLOUD_COVER', 30))
               .map(lambda img: img.normalizedDifference(['SR_B4','SR_B3']).rename('NDVI'))
               .mean())
        source = 'Landsat7'

    try:
        val = col.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=point,
            scale=100,
            maxPixels=1e9
        ).get('NDVI').getInfo()
        return round(float(val), 4) if val is not None else None, source
    except Exception:
        return None, source


def fetch_ndvi_il(il, lat, lon, start_year=2004, end_year=2024):
    point = ee.Geometry.Point([lon, lat]).buffer(25000)
    records = []
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            val, source = fetch_ndvi_month(point, year, month)
            records.append({
                "il": il, "year": year, "month": month,
                "ndvi": val, "source": source
            })
    return pd.DataFrame(records)


def run_ndvi_pipeline(raw_dir="data/raw", start_year=2004, end_year=2024):
    print("=" * 60)
    print("AGRI-2050 | GEE NDVI 2004-2024 (Landsat7 + Sentinel-2)")
    print("=" * 60)
    print(f"  2004-2016: Landsat 7 | 2017-2024: Sentinel-2\n")

    out_path = os.path.join(raw_dir, "satellite_ndvi_iller.csv")

    # Kaldığı yerden devam
    done_iller = set()
    all_records = []
    if os.path.exists(out_path):
        existing = pd.read_csv(out_path)
        # Sadece tam yıl aralığı olan iller tamamlanmış sayılır
        il_yil = existing.groupby('il')['year'].agg(['min','max'])
        done_iller = set(il_yil[(il_yil['min'] <= start_year) & (il_yil['max'] >= end_year)].index)
        all_records = [existing]
        print(f"  Devam ediliyor — {len(done_iller)} il tamamlandı\n")

    iller_list = [(il, c) for il, c in ILLER.items() if il not in done_iller]
    print(f"  Kalan: {len(iller_list)} il\n")

    for i, (il, (lat, lon)) in enumerate(iller_list, 1):
        try:
            df = fetch_ndvi_il(il, lat, lon, start_year, end_year)
            all_records.append(df)
            ndvi_ort = df["ndvi"].mean()
            print(f"  [{len(done_iller)+i:2d}/81] ✓ {il:<20} ort NDVI: {ndvi_ort:.3f}")
            time.sleep(1)
        except Exception as e:
            print(f"  [{len(done_iller)+i:2d}/81] ✗ {il:<20} Hata: {e}")

        if i % 5 == 0:
            pd.concat(all_records, ignore_index=True).to_csv(out_path, index=False)
            print(f"  💾 Ara kayıt ({len(done_iller)+i} il)")

    full_df = pd.concat(all_records, ignore_index=True)
    full_df.to_csv(out_path, index=False)

    print(f"\n  ✓ {full_df['il'].nunique()}/81 il tamamlandı")
    print(f"  ✓ {len(full_df):,} kayıt → {out_path}")
    print("\n[✓] NDVI tamamlandı.\n")
    return full_df


if __name__ == "__main__":
    os.chdir(os.path.join(os.path.dirname(__file__), ".."))
    run_ndvi_pipeline()
