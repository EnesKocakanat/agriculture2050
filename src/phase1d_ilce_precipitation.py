import os
import time
import json
import requests
import warnings
import pandas as pd

warnings.filterwarnings("ignore")

START_YEAR = 2019
END_YEAR = 2024

failed = []

def fetch_ilce_precip(ilce, lat, lon):

    url = "https://archive-api.open-meteo.com/v1/archive"

    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": f"{START_YEAR}-01-01",
        "end_date": f"{END_YEAR}-12-31",
        "daily": "precipitation_sum",
        "timezone": "Europe/Istanbul"
    }

    resp = requests.get(url, params=params, timeout=30)

    if resp.status_code != 200:
        raise Exception(f"HTTP {resp.status_code}")

    data = resp.json()

    if "daily" not in data:
        raise Exception("daily verisi yok")

    if "time" not in data["daily"]:
        raise Exception("time verisi yok")

    if "precipitation_sum" not in data["daily"]:
        raise Exception("precipitation_sum yok")

    records = []

    for date, p in zip(
        data["daily"]["time"],
        data["daily"]["precipitation_sum"]
    ):

        y, m, _ = date.split("-")

        records.append({
            "ilce": ilce,
            "year": int(y),
            "month": int(m),
            "precip_mm": round(float(p), 2) if p else 0.0
        })

    return pd.DataFrame(records)


# ROOT
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

# JSON
with open("geo/turkey-ilçeler/cities.json", "r", encoding="utf-8") as f:
    cities = json.load(f)

# İLÇELER
ilceler = []

for city in cities:

    for town in city["towns"]:

        ilceler.append({
            "il": city["name"],
            "ilce": town["name"],
            "lat": town["latitude"],
            "lon": town["longitude"]
        })

print(f"Toplam {len(ilceler)} ilçe")

# OUTPUT
out_path = "data/raw/precipitation_ilceler.csv"

done = set()
all_records = []

# DEVAM ET
if os.path.exists(out_path):

    ex = pd.read_csv(out_path)

    if "il" in ex.columns:
        done = set(ex["il"] + "_" + ex["ilce"])

    all_records = [ex]

    print(f"Devam: {len(done)} ilçe tamamlandı")

# KALAN
kalan = [
    r for r in ilceler
    if f"{r['il']}_{r['ilce']}" not in done
]

print(f"Kalan: {len(kalan)} ilçe\n")

# LOOP
for i, r in enumerate(kalan, 1):

    try:

        df = fetch_ilce_precip(
            r["ilce"],
            r["lat"],
            r["lon"]
        )

        df["il"] = r["il"]

        all_records.append(
            df[["il", "ilce", "year", "month", "precip_mm"]]
        )

        yillik = df.groupby("year")["precip_mm"].sum().mean()

        print(
            f"[{len(done)+i:4d}/973] ✓ "
            f"{r['il']:<15} "
            f"{r['ilce']:<20} "
            f"{yillik:.0f} mm"
        )

        # RATE LIMIT KORUMA
        time.sleep(15)

    except Exception as e:

        print(
            f"[{len(done)+i:4d}/973] ✗ "
            f"{r['ilce']:<20} "
            f"HATA: {e}"
        )

        # 429 ise uzun bekle
        if "429" in str(e):

            bekleme = 600

            print(f"⏳ Rate limit! {bekleme}sn bekleniyor...")

            time.sleep(bekleme)

        else:

            time.sleep(30)

        # RETRY
        try:

            df = fetch_ilce_precip(
                r["ilce"],
                r["lat"],
                r["lon"]
            )

            df["il"] = r["il"]

            all_records.append(
                df[["il", "ilce", "year", "month", "precip_mm"]]
            )

            pd.concat(
                all_records,
                ignore_index=True
            ).to_csv(
                out_path,
                index=False
            )

            print(f"  → Retry başarılı: {r['ilce']}")

        except Exception as e2:

            print(f"  → Retry başarısız: {e2}")

            failed.append(r)

    # HER 50 İLÇEDE KAYDET
    if i % 50 == 0:

        pd.concat(
            all_records,
            ignore_index=True
        ).to_csv(
            out_path,
            index=False
        )

        print(f"\n💾 Ara kayıt ({len(done)+i} ilçe)\n")


# FINAL SAVE
pd.concat(
    all_records,
    ignore_index=True
).to_csv(
    out_path,
    index=False
)

print(f"\n✓ Tamamlandı → {out_path}")

# FAIL LIST
if failed:

    pd.DataFrame(failed).to_csv(
        "failed_ilceler.csv",
        index=False
    )

    print(f"\n⚠ Başarısız ilçe sayısı: {len(failed)}")