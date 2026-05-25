"""
AGRI-2050: Faz 0b — TÜİK Gerçek Rekolte Verisi Entegrasyonu
============================================================
İl bazlı buğday üretim ve verim verisi (2004-2024).
Çalıştırma: python src/phase0b_tuik_rekolt.py
============================================================
"""

import os, warnings
import pandas as pd
import numpy as np
warnings.filterwarnings("ignore")


def parse_tuik_rekolt(xls_path):
    df = pd.read_excel(xls_path, header=None)

    # İl isimlerini çek (satır 1, sütun 3'ten itibaren)
    il_row = df.iloc[1, 3:].tolist()
    iller = [str(x).split('-')[0].strip() for x in il_row if str(x) != 'nan']

    # Gösterge satırlarını bul
    gostergeler = {}
    for i, row in df.iterrows():
        val = str(row[1])
        if 'Üretim Miktarı' in val:
            gostergeler['uretim_ton'] = i
        elif 'Verim' in val:
            gostergeler['verim_kg_dekar'] = i
        elif 'Ekilen Alan' in val and 'uretim_ton' not in gostergeler:
            gostergeler['ekilen_alan'] = i

    # Veriyi uzun formata çevir
    records = []
    for gosterge_adi, baslangic_satir in gostergeler.items():
        for offset in range(21):
            satir = df.iloc[baslangic_satir + offset]
            yil = satir[2]
            if pd.isna(yil):
                continue
            yil = int(yil)
            if yil < 2004:
                continue
            degerler = satir[3:3+len(iller)].tolist()
            for il, deger in zip(iller, degerler):
                records.append({
                    'il': il, 'year': yil,
                    'gosterge': gosterge_adi, 'deger': deger
                })

    df_long = pd.DataFrame(records)
    df_pivot = df_long.pivot_table(
        index=['il','year'], columns='gosterge', values='deger'
    ).reset_index()
    df_pivot.columns.name = None

    # Verim hesapla
    if 'uretim_ton' in df_pivot.columns and 'ekilen_alan' in df_pivot.columns:
        df_pivot['ekilen_ha'] = (df_pivot['ekilen_alan'] / 10).round(0)
        df_pivot['verim_t_ha'] = (df_pivot['uretim_ton'] / df_pivot['ekilen_ha']).round(4)
        df_pivot['verim_kg_dekar'] = df_pivot['verim_kg_dekar'].round(1)

    return df_pivot


def run_rekolt_pipeline(external_dir="data/external", processed_dir="data/processed"):
    print("=" * 60)
    print("AGRI-2050 | TÜİK Gerçek Rekolte Verisi")
    print("=" * 60 + "\n")

    # XLS dosyasını bul
    xls_files = [f for f in os.listdir(external_dir) if f.endswith('.xls') and 'pivot' in f.lower()]
    if not xls_files:
        xls_files = [f for f in os.listdir(external_dir) if f.endswith('.xls')]
    
    if not xls_files:
        print(f"  HATA: {external_dir} klasöründe .xls dosyası bulunamadı!")
        print(f"  TÜİK'ten indirilen pivot.xls dosyasını {external_dir} klasörüne koyun.")
        return None

    xls_path = os.path.join(external_dir, xls_files[0])
    print(f"  [1/3] Dosya okunuyor: {xls_files[0]}")
    df = parse_tuik_rekolt(xls_path)
    print(f"    ✓ {df['il'].nunique()} il | {len(df)} kayıt | {df['year'].min()}-{df['year'].max()}")

    # Özet istatistikler
    print("\n  [2/3] Özet istatistikler:")
    ozet = df.groupby('year').agg(
        toplam_uretim=('uretim_ton','sum'),
        ort_verim_kg=('verim_kg_dekar','mean'),
        il_sayisi=('il','count')
    ).reset_index()
    for _, r in ozet.iterrows():
        print(f"    {int(r['year'])}: {r['toplam_uretim']/1e6:.2f}M ton | "
              f"Ort verim: {r['ort_verim_kg']:.0f} kg/dekar | {int(r['il_sayisi'])} il")

    # En yüksek/düşük verimli iller (2024)
    df_2024 = df[df['year'] == df['year'].max()].dropna(subset=['verim_t_ha'])
    print(f"\n  En yüksek verimli iller ({int(df['year'].max())}):")
    for _, r in df_2024.nlargest(5, 'verim_t_ha').iterrows():
        print(f"    {r['il']:<20} {r['verim_t_ha']:.2f} t/ha")

    print(f"\n  En düşük verimli iller ({int(df['year'].max())}):")
    for _, r in df_2024.nsmallest(5, 'verim_t_ha').iterrows():
        print(f"    {r['il']:<20} {r['verim_t_ha']:.2f} t/ha")

    # Kaydet
    out_path = os.path.join(processed_dir, "tuik_rekolt_clean.csv")
    df.to_csv(out_path, index=False)

    print(f"\n  [3/3] Kaydedildi: {out_path}")
    print("\n[✓] TÜİK rekolte verisi tamamlandı.\n")
    return df


if __name__ == "__main__":
    os.chdir(os.path.join(os.path.dirname(__file__), ".."))
    run_rekolt_pipeline()
