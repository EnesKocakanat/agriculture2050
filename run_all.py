"""
AGRI-2050 - Ana Çalıştırma Scripti (Güncellenmiş)
==================================================
Tüm fazları sırayla çalıştırır.
Kullanım: python run_all.py
         python run_all.py --skip-gee   (NDVI çekmeyi atla)
         python run_all.py --only 4     (sadece faz 4)
==================================================
"""
import sys, os, time, argparse
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
DATA_DIR      = os.path.join(BASE_DIR, "data")
RAW_DIR       = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
EXTERNAL_DIR  = os.path.join(DATA_DIR, "external")
MODEL_DIR     = os.path.join(BASE_DIR, "models")

def baslik(metin):
    print(f"\n{'═'*60}")
    print(f"  {metin}")
    print(f"{'═'*60}")

def adim(no, metin):
    print(f"\n  [Faz {no}] {metin}...")

def tamam(metin=""):
    print(f"  ✓ {metin}" if metin else "  ✓ Tamamlandı")

def atla(metin):
    print(f"  ⏭ Atlandı: {metin}")


def faz0_tuik():
    adim(0, "TÜİK Veri Entegrasyonu")
    tuik_path = os.path.join(PROCESSED_DIR, "tuik_iller_clean.csv")
    if os.path.exists(tuik_path):
        atla("tuik_iller_clean.csv zaten mevcut")
        return
    from phase0_tuik_integration import run_tuik_pipeline
    run_tuik_pipeline(EXTERNAL_DIR, PROCESSED_DIR)
    tamam("TÜİK verisi işlendi")


def faz1_konya():
    adim(1, "Konya Ovası Simüle Veri")
    ndvi_path = os.path.join(RAW_DIR, "satellite_ndvi.csv")
    if os.path.exists(ndvi_path):
        atla("Konya simüle verisi zaten mevcut")
        return
    from phase1_data_fetch import run_data_pipeline
    run_data_pipeline(DATA_DIR)
    tamam("Konya verisi oluşturuldu")


def faz1b_yagis():
    adim("1b", "81 İl Yağış Verisi (Open-Meteo)")
    precip_path = os.path.join(RAW_DIR, "precipitation_iller.csv")
    if os.path.exists(precip_path):
        import pandas as pd
        df = pd.read_csv(precip_path)
        il_sayisi = df['il'].nunique()
        if il_sayisi >= 81:
            atla(f"precipitation_iller.csv mevcut ({il_sayisi} il)")
            return
        print(f"  → Kaldığı yerden devam ({il_sayisi} il tamamlandı)...")
    from phase1b_chirps import run_chirps_pipeline
    run_chirps_pipeline(raw_dir=RAW_DIR)
    tamam("81 il yağış verisi tamamlandı")


def faz1c_ndvi(skip_gee=False):
    adim("1c", "81 İl NDVI Verisi (GEE Sentinel-2)")
    ndvi_path = os.path.join(RAW_DIR, "satellite_ndvi_iller.csv")
    if skip_gee:
        atla("--skip-gee bayrağı ile atlandı")
        return
    if os.path.exists(ndvi_path):
        import pandas as pd
        df = pd.read_csv(ndvi_path)
        il_sayisi = df['il'].nunique()
        if il_sayisi >= 81:
            atla(f"satellite_ndvi_iller.csv mevcut ({il_sayisi} il)")
            return
        print(f"  → Kaldığı yerden devam ({il_sayisi} il tamamlandı)...")
    from phase1c_gee_ndvi import run_ndvi_pipeline
    run_ndvi_pipeline(raw_dir=RAW_DIR)
    tamam("81 il NDVI verisi tamamlandı")


def faz2_konya():
    adim(2, "Konya ML Model Eğitimi")
    model_path = os.path.join(MODEL_DIR, "ensemble_model.pkl")
    if os.path.exists(model_path):
        atla("Konya modeli zaten mevcut")
        return
    from phase2_ml_model import run_training_pipeline
    run_training_pipeline(DATA_DIR, MODEL_DIR)
    tamam("Konya modeli eğitildi")


def faz2b_turkey():
    adim("2b", "Türkiye Geneli ML Model")
    scores_path = os.path.join(PROCESSED_DIR, "turkey_il_scores.csv")
    if os.path.exists(scores_path):
        import pandas as pd
        df = pd.read_csv(scores_path)
        if len(df) >= 80:
            atla("turkey_il_scores.csv mevcut")
            return
    from phase2_ml_turkey import run_turkey_pipeline
    run_turkey_pipeline(
        raw_dir=RAW_DIR,
        processed_dir=PROCESSED_DIR,
        external_dir=EXTERNAL_DIR,
        model_dir=MODEL_DIR
    )
    tamam("Türkiye modeli eğitildi")


def faz4_forecast():
    adim(4, "2025-2050 İklim Senaryoları")
    fc_path = os.path.join(PROCESSED_DIR, "forecast_turkey_2025_2050.csv")
    if os.path.exists(fc_path):
        import pandas as pd
        df = pd.read_csv(fc_path)
        if len(df) >= 6000:
            atla("forecast_turkey_2025_2050.csv mevcut")
            return
    from phase4_forecast_turkey import run_forecast
    run_forecast(processed_dir=PROCESSED_DIR, out_dir=PROCESSED_DIR)
    tamam("2025-2050 projeksiyonlar tamamlandı")


def ozet_yazdir():
    import pandas as pd, json

    baslik("AGRI-2050 | ÖZET RAPOR")

    # Konya metrikleri
    metrics_path = os.path.join(MODEL_DIR, "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            m = json.load(f)
        print(f"\n  Konya Modeli:")
        print(f"    Test MAE : {m.get('test_mae','N/A')} t/ha")
        print(f"    Test R²  : {m.get('test_r2','N/A')}")

    # Türkiye metrikleri
    tr_metrics = os.path.join(MODEL_DIR, "turkey_metrics.json")
    if os.path.exists(tr_metrics):
        with open(tr_metrics) as f:
            tm = json.load(f)
        print(f"\n  Türkiye Modeli:")
        print(f"    Test MAE : {tm.get('test_mae','N/A')}")
        print(f"    Test R²  : {tm.get('test_r2','N/A')}")
        print(f"    İl Sayısı: {tm.get('il_count','N/A')}")

    # 2050 özet
    fc_path = os.path.join(PROCESSED_DIR, "forecast_turkey_2025_2050.csv")
    if os.path.exists(fc_path):
        fc = pd.read_csv(fc_path)
        baz_path = os.path.join(PROCESSED_DIR, "turkey_il_scores.csv")
        baz = pd.read_csv(baz_path)[['il','tarim_skoru']].rename(columns={'tarim_skoru':'baz'})
        df_2050 = fc[(fc['year']==2050) & (fc['scenario']=='kotumser_ssp585')].merge(baz, on='il')
        df_2050['degisim'] = (df_2050['tarim_skoru'] - df_2050['baz']) / df_2050['baz'] * 100
        print(f"\n  2050 Kötümser Senaryo:")
        print(f"    Ort. Skor Değişimi: {df_2050['degisim'].mean():.1f}%")
        en_cok = df_2050.nsmallest(3,'degisim')['il'].tolist()
        print(f"    En Riskli İller  : {', '.join(en_cok)}")

    print(f"\n  Dashboard başlatmak için:")
    print(f"  → python -m streamlit run dashboard/phase3_dashboard.py")
    print(f"  → python -m streamlit run dashboard/phase3_turkey_dashboard.py")
    print()


def main():
    parser = argparse.ArgumentParser(description="AGRI-2050 Pipeline")
    parser.add_argument('--skip-gee', action='store_true', help='GEE NDVI adımını atla')
    parser.add_argument('--only', type=int, help='Sadece belirtilen fazı çalıştır')
    args = parser.parse_args()

    baslik("AGRI-2050 | Türkiye Tarım Tahmin Sistemi")
    print("  2050'de %70 artan küresel gıda ihtiyacına çözüm")

    t0 = time.time()

    if args.only:
        fazlar = {
            0: faz0_tuik,
            1: faz1_konya,
            2: faz2_konya,
            4: faz4_forecast,
        }
        if args.only in fazlar:
            fazlar[args.only]()
        else:
            print(f"  Hata: Faz {args.only} bulunamadı")
    else:
        faz0_tuik()
        faz1_konya()
        faz1b_yagis()
        faz1c_ndvi(skip_gee=args.skip_gee)
        faz2_konya()
        faz2b_turkey()
        faz4_forecast()

    sure = time.time() - t0
    ozet_yazdir()
    print(f"  Toplam süre: {sure:.1f} saniye\n")


if __name__ == "__main__":
    os.chdir(BASE_DIR)
    main()
