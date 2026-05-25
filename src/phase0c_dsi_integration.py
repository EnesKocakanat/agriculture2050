"""
AGRI-2050: Faz 0c — DSİ Sulama Verisi Entegrasyonu
===================================================
2.5  İl Bazlı Sulama Alanları (2019-2024)
2.13 İl Bazlı Sulama Oranı    (2019-2024)
2.4  Hububat Sulama Verim Artışı (2013-2024)
Çalıştırma: python src/phase0c_dsi_integration.py
===================================================
"""

import os, warnings
import pandas as pd
warnings.filterwarnings("ignore")

YIL_SUTUN_2_5  = {2019:8,  2020:13, 2021:18, 2022:23, 2023:28, 2024:33}
YIL_SUTUN_2_13 = {2019:3,  2020:4,  2021:5,  2022:6,  2023:7,  2024:8}
ONCE_SUTUN     = [5,17,29,42,55,68,79,90,101,112,123,134]
SONRA_SUTUN    = [7,19,31,44,57,70,81,92,103,114,125,136]
ARTIS_SUTUN    = [9,21,32,45,58,69,80,91,102,113,124,135]
YILLAR_2_4     = list(range(2013, 2025))


def parse_sulama_alanlari(xls_path):
    df = pd.read_excel(xls_path, header=None)
    records = []
    for i in range(6, 88):
        row = df.iloc[i]
        il = str(row[2]).strip()
        if pd.isna(row[2]) or il in ['nan', 'Türkiye']:
            continue
        for yil, sut in YIL_SUTUN_2_5.items():
            try:
                records.append({'il': il, 'year': yil,
                                 'sulanan_ha': round(float(row[sut]), 1)})
            except Exception:
                pass
    return pd.DataFrame(records)


def parse_sulama_orani(xls_path):
    df = pd.read_excel(xls_path, header=None)
    records = []
    for i in range(4, 87):
        row = df.iloc[i]
        il = str(row[2]).strip()
        if pd.isna(row[2]) or il == 'nan':
            continue
        for yil, sut in YIL_SUTUN_2_13.items():
            try:
                records.append({'il': il, 'year': yil,
                                 'sulama_orani': round(float(row[sut]), 4)})
            except Exception:
                pass
    return pd.DataFrame(records)


def parse_verim_artisi(xls_path):
    df = pd.read_excel(xls_path, header=None)
    hububat = df.iloc[9]
    records = []
    for yil, os_, ss_, as_ in zip(YILLAR_2_4, ONCE_SUTUN, SONRA_SUTUN, ARTIS_SUTUN):
        try:
            records.append({
                'year': yil,
                'susuz_kg_da':  float(hububat[os_]),
                'sululu_kg_da': float(hububat[ss_]),
                'artis_pct':    float(hububat[as_])
            })
        except Exception:
            pass
    return pd.DataFrame(records)


def run_dsi_pipeline(external_dir="data/external", processed_dir="data/processed"):
    print("=" * 60)
    print("AGRI-2050 | DSİ Gerçek Sulama Verisi Entegrasyonu")
    print("=" * 60 + "\n")

    # Dosyaları bul
    def bul(pattern):
        for f in os.listdir(external_dir):
            if pattern in f.lower():
                return os.path.join(external_dir, f)
        return None

    f25  = bul("2_5")
    f213 = bul("2_13")
    f24  = bul("2_4")

    if not all([f25, f213, f24]):
        print("  HATA: DSİ Excel dosyaları bulunamadı!")
        print(f"  Eksik dosyalar: { [n for n,f in [('2_5',f25),('2_13',f213),('2_4',f24)] if not f] }")
        print(f"  Lütfen DSİ Excel dosyalarını {external_dir}/ klasörüne koyun.")
        return None

    print("[1/4] Sulama alanları (2.5) okunuyor...")
    df_alan = parse_sulama_alanlari(f25)
    print(f"    ✓ {df_alan['il'].nunique()} il | {len(df_alan)} kayıt")

    print("[2/4] Sulama oranları (2.13) okunuyor...")
    df_oran = parse_sulama_orani(f213)
    print(f"    ✓ {df_oran['il'].nunique()} il | {len(df_oran)} kayıt")

    print("[3/4] Hububat verim artışları (2.4) okunuyor...")
    df_artis = parse_verim_artisi(f24)
    ort_artis = df_artis['artis_pct'].mean()
    print(f"    ✓ {len(df_artis)} yıl | Ort. verim artışı: %{ort_artis:.0f}")
    print(f"    Susuz ort: {df_artis['susuz_kg_da'].mean():.0f} kg/da → "
          f"Sululu ort: {df_artis['sululu_kg_da'].mean():.0f} kg/da")

    print("[4/4] Birleştiriliyor ve kaydediliyor...")
    df_dsi = df_alan.merge(df_oran, on=['il', 'year'], how='outer')
    df_dsi['hububat_sulama_verim_artis_pct'] = round(ort_artis, 1)

    # En çok/az sulanan iller (2024)
    df24 = df_dsi[df_dsi['year'] == 2024].dropna(subset=['sulanan_ha'])
    print(f"\n  En fazla sulanan iller (2024):")
    for _, r in df24.nlargest(5, 'sulanan_ha').iterrows():
        print(f"    {r['il']:<20} {r['sulanan_ha']:>12,.0f} ha  "
              f"(oran: {r['sulama_orani']:.2f})" if pd.notna(r.get('sulama_orani')) else
              f"    {r['il']:<20} {r['sulanan_ha']:>12,.0f} ha")

    print(f"\n  En az sulanan iller (2024):")
    for _, r in df24[df24['sulanan_ha'] > 0].nsmallest(5, 'sulanan_ha').iterrows():
        print(f"    {r['il']:<20} {r['sulanan_ha']:>12,.0f} ha")

    # Kaydet
    out1 = f"{processed_dir}/dsi_sulama_clean.csv"
    out2 = f"{processed_dir}/dsi_verim_artisi.csv"
    df_dsi.to_csv(out1, index=False)
    df_artis.to_csv(out2, index=False)

    print(f"\n  ✓ {out1}")
    print(f"  ✓ {out2}")
    print(f"\n[✓] DSİ entegrasyonu tamamlandı.\n")
    return df_dsi, df_artis


if __name__ == "__main__":
    os.chdir(os.path.join(os.path.dirname(__file__), ".."))
    run_dsi_pipeline()
