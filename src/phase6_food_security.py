"""
AGRI-2050: Faz 6 — Gıda Güvenliği Modülü
=========================================
2024-2050 Türkiye buğday arz-talep dengesi.
Çalıştırma: python src/phase6_food_security.py
=========================================
"""

import os, json, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

# ── Parametreler ──────────────────────────────────────────────────────
# TÜİK / FAO / UN verileri
NUFUS_2024        = 85_372_000
NUFUS_BUYUME_YILI = 0.0065       # yıllık %0.65 (TÜİK projeksiyon)
KISI_BASI_TUKETIM = 215          # kg/yıl (buğday + un eşdeğeri, FAO)
TUKETIM_ARTIS     = 0.002        # yıllık %0.2 artış (gelir artışı etkisi)
IHRACAT_PAZAR_PAY = 0.18         # üretimin %18'i ihraç ediliyor (2024 ort.)
ITHALAT_ESIK      = 0.05         # üretim açığının %5'i ithalatla karşılanabilir

# Üretim baz değerleri (2024 gerçek TÜİK)
URETIM_2024_TON   = 16_400_000
VERIM_2024        = 2.73         # t/ha ortalama

# İklim senaryoları — üretim değişim katsayıları
SENARYO_URETIM_ETKISI = {
    "optimist_ssp126": {
        "label": "İyimser (SSP1-2.6)",
        "yillik_degisim": -0.003,   # yıllık %0.3 düşüş
        "sulama_telafi":  +0.002,   # sulama iyileştirmesi
        "teknoloji_artis": +0.004,  # teknoloji verimi artışı
        "color": "#22c55e"
    },
    "orta_ssp245": {
        "label": "Orta (SSP2-4.5)",
        "yillik_degisim": -0.006,
        "sulama_telafi":  +0.002,
        "teknoloji_artis": +0.003,
        "color": "#f59e0b"
    },
    "kotumser_ssp585": {
        "label": "Kötümser (SSP5-8.5)",
        "yillik_degisim": -0.012,
        "sulama_telafi":  +0.001,
        "teknoloji_artis": +0.002,
        "color": "#ef4444"
    },
}

# Politika senaryoları
POLITIKA_SENARYOLARI = {
    "baz": {
        "label": "Baz Senaryo (mevcut trend)",
        "sulama_genisletme": 0.0,
        "verim_artis_hedef": 0.0,
    },
    "ileri_sulama": {
        "label": "İleri Sulama Politikası",
        "sulama_genisletme": 0.15,   # ekilen alanın %15 daha sulanması
        "verim_artis_hedef": 0.008,  # yıllık %0.8 verim artışı
    },
    "yuksek_teknoloji": {
        "label": "Yüksek Teknoloji + Sulama",
        "sulama_genisletme": 0.25,
        "verim_artis_hedef": 0.015,  # yıllık %1.5 verim artışı
    },
}


def hesapla_talep(yil):
    """Yıllık buğday talebi (ton)"""
    dt = yil - 2024
    nufus = NUFUS_2024 * (1 + NUFUS_BUYUME_YILI) ** dt
    kisi_basi = KISI_BASI_TUKETIM * (1 + TUKETIM_ARTIS) ** dt
    ic_talep = nufus * kisi_basi / 1000  # kg → ton
    return round(ic_talep), round(nufus)


def hesapla_uretim(yil, senaryo_key, politika_key):
    """Yıllık buğday üretimi (ton)"""
    dt = yil - 2024
    sc = SENARYO_URETIM_ETKISI[senaryo_key]
    pol = POLITIKA_SENARYOLARI[politika_key]

    # Net yıllık değişim
    net_degisim = (sc['yillik_degisim'] +
                   sc['sulama_telafi'] +
                   sc['teknoloji_artis'] +
                   pol['verim_artis_hedef'] +
                   pol['sulama_genisletme'] * 0.01)

    uretim = URETIM_2024_TON * (1 + net_degisim) ** dt
    return round(uretim)


def run_food_security(processed_dir="data/processed"):
    print("=" * 60)
    print("AGRI-2050 | Gıda Güvenliği Modülü")
    print("=" * 60 + "\n")

    yillar = list(range(2024, 2051))
    records = []

    print("[1/3] Arz-talep projeksiyonları hesaplanıyor...")
    for yil in yillar:
        talep, nufus = hesapla_talep(yil)
        for sc_key in SENARYO_URETIM_ETKISI:
            for pol_key in POLITIKA_SENARYOLARI:
                uretim = hesapla_uretim(yil, sc_key, pol_key)
                ihracat = round(uretim * IHRACAT_PAZAR_PAY)
                kullanilabilir = uretim - ihracat
                denge = kullanilabilir - talep
                kendi_kendine_yeterlilik = kullanilabilir / talep * 100

                records.append({
                    "year": yil,
                    "nufus": nufus,
                    "senaryo": sc_key,
                    "senaryo_label": SENARYO_URETIM_ETKISI[sc_key]['label'],
                    "politika": pol_key,
                    "politika_label": POLITIKA_SENARYOLARI[pol_key]['label'],
                    "uretim_ton": uretim,
                    "ihracat_ton": ihracat,
                    "kullanilabilir_ton": kullanilabilir,
                    "ic_talep_ton": talep,
                    "denge_ton": denge,
                    "kky_pct": round(kendi_kendine_yeterlilik, 1),
                    "acik_var": denge < 0,
                })

    df = pd.DataFrame(records)
    out_path = f"{processed_dir}/food_security_2024_2050.csv"
    df.to_csv(out_path, index=False)
    print(f"    ✓ {len(df):,} projeksiyon kaydı")

    print("\n[2/3] 2050 Özet:")
    print(f"\n  {'Senaryo':<25} {'Politika':<30} {'Üretim':>12} {'Talep':>12} {'Denge':>12} {'KKY%':>7}")
    print("  " + "-" * 100)

    df_2050 = df[df['year'] == 2050]
    for _, r in df_2050.iterrows():
        denge_str = f"{r['denge_ton']/1e6:+.2f}M"
        durum = "✓" if r['denge_ton'] >= 0 else "✗"
        print(f"  {durum} {r['senaryo_label']:<23} {r['politika_label']:<30} "
              f"{r['uretim_ton']/1e6:>10.2f}M  "
              f"{r['ic_talep_ton']/1e6:>10.2f}M  "
              f"{denge_str:>12}  {r['kky_pct']:>6.1f}%")

    print("\n[3/3] Kritik Bulgular:")
    # Kötümser + baz senaryoda ne zaman açık başlıyor?
    kotumser_baz = df[(df['senaryo']=='kotumser_ssp585') & (df['politika']=='baz')]
    acik_yil = kotumser_baz[kotumser_baz['acik_var']]['year'].min()
    if pd.notna(acik_yil):
        print(f"    ⚠ Kötümser senaryoda gıda açığı başlangıcı: {int(acik_yil)}")
    else:
        print(f"    ✓ Kötümser senaryoda bile 2050'ye kadar açık yok")

    # Yüksek teknoloji politikasının katkısı
    iy_2050 = df[(df['year']==2050) & (df['senaryo']=='orta_ssp245') & (df['politika']=='yuksek_teknoloji')]['uretim_ton'].values[0]
    baz_2050 = df[(df['year']==2050) & (df['senaryo']=='orta_ssp245') & (df['politika']=='baz')]['uretim_ton'].values[0]
    fark = (iy_2050 - baz_2050) / 1e6
    print(f"    ✓ Yüksek teknoloji politikası 2050'de +{fark:.1f}M ton ek üretim sağlar")

    # 2050 nüfus ve talep
    talep_2050, nufus_2050 = hesapla_talep(2050)
    print(f"    → 2050 tahmini nüfus : {nufus_2050/1e6:.1f}M")
    print(f"    → 2050 iç talep      : {talep_2050/1e6:.2f}M ton")

    # Özet JSON
    ozet = {
        "acik_baslangic_yil": int(acik_yil) if pd.notna(acik_yil) else None,
        "nufus_2050": nufus_2050,
        "talep_2050_ton": talep_2050,
        "uretim_2050_orta_baz": int(baz_2050),
        "uretim_2050_orta_ileri": int(iy_2050),
        "teknoloji_katki_ton": int(iy_2050 - baz_2050),
    }
    with open(f"{processed_dir}/food_security_summary.json", "w", encoding='utf-8') as f:
        json.dump(ozet, f, indent=2, ensure_ascii=False)

    print(f"\n  ✓ Kaydedildi: {out_path}")
    print("\n[✓] Gıda güvenliği modülü tamamlandı.\n")
    return df


if __name__ == "__main__":
    os.chdir(os.path.join(os.path.dirname(__file__), ".."))
    run_food_security()
