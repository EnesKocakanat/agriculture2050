"""
AGRI-2050: TÜİK Veri Entegrasyonu
- data/external/tuik_tarim_2024.xls dosyasını okur
- 81 il için tarım alanı verisini çıkarır
- data/processed/tuik_iller_clean.csv olarak kaydeder
"""

import xlrd, re, os
import pandas as pd

def load_tuik_data(filepath: str) -> pd.DataFrame:
    wb = xlrd.open_workbook(filepath)
    ws = wb.sheet_by_index(0)

    iller = []
    for i in range(13, ws.nrows):
        row = ws.cell_value(i, 0)
        if isinstance(row, str) and row.strip():
            if re.match(r'\s{5,}TR\w{3}\s+', row):
                il_adi = re.sub(r'^\s+TR\w+\s+', '', row).strip()
                vals = [ws.cell_value(i, j) for j in range(1, 7)]
                iller.append([il_adi] + vals)

    df = pd.DataFrame(iller, columns=[
        'il', 'toplam_alan_dekar', 'ekilen_alan_dekar',
        'nadas_dekar', 'sebze_bahce_dekar',
        'meyve_ickibaharat_dekar', 'sus_bitkileri_dekar'
    ])

    # Hektara çevir
    for col in ['toplam_alan_dekar','ekilen_alan_dekar','nadas_dekar',
                'sebze_bahce_dekar','meyve_ickibaharat_dekar']:
        df[col.replace('dekar','ha').replace('alan_','').replace('bahce_','')] = \
            (df[col] / 10).round(1)

    df['yil'] = 2024
    df = df.drop_duplicates(subset='il', keep='first')
    df = df[df['ekilen_alan_dekar'] > 0].reset_index(drop=True)
    return df


def enrich_yield_data(yield_df: pd.DataFrame, tuik_df: pd.DataFrame) -> pd.DataFrame:
    """
    Rekolte verisine TÜİK alan bilgisini ekle.
    Konya için gerçek alan verisini kullan.
    """
    konya = tuik_df[tuik_df['il'] == 'Konya'].iloc[0]
    real_area_ha = int(konya['ekilen_alan_dekar'] / 10)

    enriched = yield_df.copy()
    # Konya buğday alanını güncelle (simüle 580K → gerçek 1.5M'dan ~%40'ı buğday)
    bugday_share = 0.40
    enriched.loc[enriched['crop'] == 'bugday', 'area_ha'] = int(real_area_ha * bugday_share)
    enriched.loc[enriched['crop'] == 'bugday', 'total_prod_t'] = \
        (enriched.loc[enriched['crop'] == 'bugday', 'yield_t_ha'] *
         int(real_area_ha * bugday_share)).astype(int)

    return enriched


def run_tuik_integration(external_dir='data/external',
                         processed_dir='data/processed',
                         raw_dir='data/raw'):
    print("=" * 60)
    print("AGRI-2050 | TÜİK Veri Entegrasyonu")
    print("=" * 60)

    filepath = os.path.join(external_dir, 'tuik_tarim_2024.xls')
    if not os.path.exists(filepath):
        print(f"[!] Dosya bulunamadı: {filepath}")
        return None

    print("\n[1/3] TÜİK verisi okunuyor...")
    tuik_df = load_tuik_data(filepath)
    print(f"    ✓ {len(tuik_df)} il yüklendi")

    konya = tuik_df[tuik_df['il'] == 'Konya'].iloc[0]
    print(f"    ✓ Konya ekilen alan: {int(konya['ekilen_alan_dekar']/10):,} ha (gerçek)")

    print("\n[2/3] Rekolte verisi güncelleniyor...")
    yield_path = os.path.join(raw_dir, 'crop_yields.csv')
    if os.path.exists(yield_path):
        yield_df = pd.read_csv(yield_path)
        enriched_df = enrich_yield_data(yield_df, tuik_df)
        enriched_df.to_csv(yield_path, index=False)
        print(f"    ✓ crop_yields.csv güncellendi (gerçek alan verisiyle)")

    print("\n[3/3] Temiz TÜİK verisi kaydediliyor...")
    out_path = os.path.join(processed_dir, 'tuik_iller_clean.csv')
    tuik_df.to_csv(out_path, index=False)
    print(f"    ✓ tuik_iller_clean.csv → {len(tuik_df)} il")

    # Özet istatistikler
    print(f"\n  Türkiye Toplam Ekilen Alan : {int(tuik_df['ekilen_alan_dekar'].sum()/10):,} ha")
    print(f"  En büyük 3 il:")
    top3 = tuik_df.nlargest(3, 'ekilen_alan_dekar')
    for _, r in top3.iterrows():
        print(f"    - {r['il']}: {int(r['ekilen_alan_dekar']/10):,} ha")

    print("\n[✓] TÜİK entegrasyonu tamamlandı.\n")
    return tuik_df


if __name__ == "__main__":
    os.chdir(os.path.join(os.path.dirname(__file__), '..'))
    run_tuik_integration()
