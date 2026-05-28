"""
AGRI-2050: Faz 2d — İlçe Bazlı Tarım Skoru
============================================
973 ilçe × yağış + NDVI → tarım skoru
Çalıştırma: python src/phase2d_ilce_scores.py
============================================
"""

import os, warnings
import pandas as pd
import numpy as np
warnings.filterwarnings("ignore")


def run_ilce_scores(raw_dir="data/raw", processed_dir="data/processed"):
    print("=" * 60)
    print("AGRI-2050 | İlçe Bazlı Tarım Skoru")
    print("=" * 60 + "\n")

    # Yağış yükle
    print("[1/4] Yağış verisi yükleniyor...")
    precip = pd.read_csv(os.path.join(raw_dir, "precipitation_ilceler.csv"))
    precip_y = precip.groupby(['il','ilce','year'])['precip_mm'].sum().reset_index()
    precip_ort = precip_y.groupby(['il','ilce'])['precip_mm'].agg(
        yagis_ort='mean',
        yagis_std='std',
        yagis_min='min',
        yagis_max='max'
    ).reset_index()
    print(f"    ✓ {precip_ort['ilce'].nunique()} ilçe | yağış verisi")

    # NDVI yükle
    print("[2/4] NDVI verisi yükleniyor...")
    ndvi = pd.read_csv(os.path.join(raw_dir, "ndvi_ilceler.csv"))
    ndvi_y = ndvi.groupby(['il','ilce','year'])['ndvi'].mean().reset_index()
    ndvi_ort = ndvi_y.groupby(['il','ilce'])['ndvi'].agg(
        ndvi_ort='mean',
        ndvi_peak='max',
        ndvi_std='std'
    ).reset_index()
    print(f"    ✓ {ndvi_ort['ilce'].nunique()} ilçe | NDVI verisi")

    # Birleştir
    print("[3/4] Birleştiriliyor ve skor hesaplanıyor...")
    df = precip_ort.merge(ndvi_ort, on=['il','ilce'], how='outer')

    # Eksik değerleri il ortalamasıyla doldur
    for col in ['yagis_ort','ndvi_ort','ndvi_peak']:
        il_ort = df.groupby('il')[col].transform('mean')
        df[col] = df[col].fillna(il_ort)

    # Normalize et (0-1)
    def normalize(s):
        mn, mx = s.min(), s.max()
        if mx == mn:
            return pd.Series([0.5]*len(s), index=s.index)
        return (s - mn) / (mx - mn)

    df['yagis_norm'] = normalize(df['yagis_ort'])
    df['ndvi_norm']  = normalize(df['ndvi_ort'])
    df['peak_norm']  = normalize(df['ndvi_peak'])

    # Kuraklık riski (düşük yağış = yüksek risk)
    df['kuraklik_riski'] = 1 - df['yagis_norm']

    # Tarım skoru (ağırlıklı)
    df['tarim_skoru'] = (
        df['ndvi_norm']  * 0.45 +
        df['yagis_norm'] * 0.35 +
        df['peak_norm']  * 0.20
    ) * 10  # 0-10 arası

    df['tarim_skoru'] = df['tarim_skoru'].round(4)

    # İl skorlarıyla karşılaştır
    il_scores = pd.read_csv(os.path.join(processed_dir, "turkey_il_scores.csv"))
    il_ort = il_scores[['il','tarim_skoru']].rename(columns={'tarim_skoru':'il_tarim_skoru'})
    df = df.merge(il_ort, on='il', how='left')

    # Kaydet
    out_path = os.path.join(processed_dir, "ilce_scores.csv")
    df.to_csv(out_path, index=False)

    print(f"    ✓ {len(df)} ilçe skoru hesaplandı")
    print(f"    ✓ Kaydedildi: {out_path}")

    # Özet
    print("\n[4/4] Özet:")
    print(f"\n  En yüksek tarım skoru (ilçe):")
    for _, r in df.nlargest(5, 'tarim_skoru').iterrows():
        print(f"    {r['il']:<15} {r['ilce']:<20} {r['tarim_skoru']:.3f}")

    print(f"\n  En düşük tarım skoru (ilçe):")
    for _, r in df.nsmallest(5, 'tarim_skoru').iterrows():
        print(f"    {r['il']:<15} {r['ilce']:<20} {r['tarim_skoru']:.3f}")

    print(f"\n  İl bazlı ilçe ortalamaları (ilk 5):")
    il_ozet = df.groupby('il')['tarim_skoru'].mean().reset_index().nlargest(5, 'tarim_skoru')
    for _, r in il_ozet.iterrows():
        print(f"    {r['il']:<20} {r['tarim_skoru']:.3f}")

    print(f"\n[✓] İlçe tarım skorları tamamlandı.\n")
    return df


if __name__ == "__main__":
    os.chdir(os.path.join(os.path.dirname(__file__), ".."))
    run_ilce_scores()