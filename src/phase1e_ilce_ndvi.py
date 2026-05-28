import os, time, json, warnings
import ee
import pandas as pd
warnings.filterwarnings("ignore")

ee.Initialize(project='agri2050')

START_YEAR = 2019
END_YEAR   = 2024

def fetch_ndvi_ilce(ilce, lat, lon):
    point = ee.Geometry.Point([lon, lat]).buffer(5000)
    records = []
    for year in range(START_YEAR, END_YEAR + 1):
        for month in range(1, 13):
            start = f"{year}-{month:02d}-01"
            end   = f"{year}-{month:02d}-28"
            try:
                col = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                       .filterBounds(point).filterDate(start, end)
                       .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
                       .map(lambda img: img.normalizedDifference(['B8','B4']).rename('NDVI'))
                       .mean())
                val = col.reduceRegion(
                    reducer=ee.Reducer.mean(),
                    geometry=point, scale=100, maxPixels=1e9
                ).get('NDVI').getInfo()
                records.append({"ilce": ilce, "year": year, "month": month,
                                "ndvi": round(float(val), 4) if val is not None else None})
            except:
                records.append({"ilce": ilce, "year": year, "month": month, "ndvi": None})
    return pd.DataFrame(records)

os.chdir(os.path.join(os.path.dirname(__file__), ".."))
cities = json.load(open("geo/turkey-ilçeler/cities.json", encoding="utf-8"))
ilceler = [{"il": c["name"], "ilce": t["name"], "lat": t["latitude"], "lon": t["longitude"]}
           for c in cities for t in c["towns"]]

out_path = "data/raw/ndvi_ilceler.csv"
done = set()
all_records = []
if os.path.exists(out_path):
    ex = pd.read_csv(out_path)
    done = set((ex['il'] + '_' + ex['ilce']).unique())
    all_records = [ex]
    print(f"Devam: {len(done)} ilçe tamamlandı")

kalan = [r for r in ilceler if f"{r['il']}_{r['ilce']}" not in done]
print(f"Kalan: {len(kalan)} ilçe\n")

for i, r in enumerate(kalan, 1):
    try:
        df = fetch_ndvi_ilce(r['ilce'], r['lat'], r['lon'])
        df['il'] = r['il']
        all_records.append(df[['il','ilce','year','month','ndvi']])
        print(f"[{len(done)+i:4d}/973] ✓ {r['il']:<15} {r['ilce']:<20} {df['ndvi'].mean():.3f}")
        time.sleep(1)
    except Exception as e:
        print(f"[{len(done)+i:4d}/973] ✗ {r['ilce']:<20} HATA: {e}")
        time.sleep(2)
    if i % 20 == 0:
        pd.concat(all_records, ignore_index=True).to_csv(out_path, index=False)
        print(f"\n💾 Ara kayıt ({len(done)+i} ilçe)\n")

pd.concat(all_records, ignore_index=True).to_csv(out_path, index=False)
print(f"\n✓ Tamamlandı → {out_path}")