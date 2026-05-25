"""
AGRI-2050: Faz 1b — Türkiye 81 İl Gerçek Yağış Verisi
=======================================================
Kaldığı yerden devam eder, rate limit hatalarında bekler.
Çalıştırma: python src/phase1b_chirps.py
=======================================================
"""

import os, time
import requests
import pandas as pd

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

def fetch_il(il, lat, lon, start_year=2004, end_year=2024):
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat, "longitude": lon,
        "start_date": f"{start_year}-01-01",
        "end_date":   f"{end_year}-12-31",
        "daily": "precipitation_sum",
        "timezone": "Europe/Istanbul",
    }
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    data = r.json()
    df = pd.DataFrame({"date": data["daily"]["time"],
                       "precip_mm": data["daily"]["precipitation_sum"]})
    df["date"]  = pd.to_datetime(df["date"])
    df["year"]  = df["date"].dt.year
    df["month"] = df["date"].dt.month
    monthly = df.groupby(["year","month"])["precip_mm"].sum().round(2).reset_index()
    monthly["il"] = il
    monthly["source"] = "Open-Meteo (ERA5)"
    return monthly


def run_precipitation_pipeline(raw_dir="data/raw", start_year=2004, end_year=2024):
    print("=" * 60)
    print("AGRI-2050 | Türkiye 81 İl Yağış Verisi")
    print("=" * 60)

    out_path = os.path.join(raw_dir, "precipitation_iller.csv")

    # Kaldığı yerden devam et
    done_iller = set()
    if os.path.exists(out_path):
        existing = pd.read_csv(out_path)
        done_iller = set(existing["il"].unique())
        print(f"  Devam ediliyor — {len(done_iller)} il zaten tamamlandı\n")
        all_records = [existing]
    else:
        all_records = []

    iller_list = [(il, c) for il, c in ILLER.items() if il not in done_iller]
    print(f"  Kalan: {len(iller_list)} il\n")

    for i, (il, (lat, lon)) in enumerate(iller_list, 1):
        retries = 3
        for attempt in range(retries):
            try:
                df = fetch_il(il, lat, lon, start_year, end_year)
                all_records.append(df)
                yillik = df.groupby("year")["precip_mm"].sum().mean()
                print(f"  [{len(done_iller)+i:2d}/81] ✓ {il:<20} ort: {yillik:.0f} mm/yıl")
                time.sleep(2)  # rate limit için 2sn bekle
                break
            except requests.exceptions.HTTPError as e:
                if "429" in str(e):
                    wait = 30 * (attempt + 1)
                    print(f"  [{len(done_iller)+i:2d}/81] ⏳ {il:<20} Rate limit — {wait}sn bekleniyor...")
                    time.sleep(wait)
                else:
                    print(f"  [{len(done_iller)+i:2d}/81] ✗ {il:<20} Hata: {e}")
                    break
            except Exception as e:
                print(f"  [{len(done_iller)+i:2d}/81] ✗ {il:<20} Hata: {e}")
                break

        # Her 10 ilde bir kaydet (ara kayıt)
        if i % 10 == 0:
            pd.concat(all_records, ignore_index=True).to_csv(out_path, index=False)
            print(f"  💾 Ara kayıt yapıldı ({len(done_iller)+i} il)")

    full_df = pd.concat(all_records, ignore_index=True)

    # Kuraklık yılı flag
    yearly = full_df.groupby(["il","year"])["precip_mm"].sum().reset_index()
    yearly["drought_year"] = yearly["precip_mm"] < 250
    full_df = full_df.merge(yearly[["il","year","drought_year"]], on=["il","year"], how="left")
    full_df.to_csv(out_path, index=False)

    tamamlanan = full_df["il"].nunique()
    print(f"\n  ✓ {tamamlanan}/81 il tamamlandı")
    print(f"  ✓ {len(full_df):,} kayıt → {out_path}")

    if tamamlanan > 0:
        il_ort = yearly.groupby("il")["precip_mm"].mean().sort_values()
        print(f"\n  En kurak 3 il : {', '.join(il_ort.head(3).index.tolist())}")
        print(f"  En yağışlı 3  : {', '.join(il_ort.tail(3).index.tolist())}")

    print("\n[✓] Tamamlandı.\n")
    return full_df


if __name__ == "__main__":
    os.chdir(os.path.join(os.path.dirname(__file__), ".."))
    run_precipitation_pipeline()
