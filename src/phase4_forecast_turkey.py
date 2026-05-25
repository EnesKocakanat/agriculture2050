"""
AGRI-2050: Faz 4 — Türkiye 81 İl 2025-2050 İklim Senaryoları
=============================================================
IPCC SSP1-2.6 / SSP2-4.5 / SSP5-8.5 senaryolarıyla
81 il için yıllık tarım skoru tahmini.
Çalıştırma: python src/phase4_forecast_turkey.py
=============================================================
"""

import os, pickle, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

# ── IPCC Değişim Katsayıları (2025-2050, Türkiye) ──────────────────────────
# Kaynak: IPCC AR6 / MED bölgesi projeksiyon ortalamaları
SCENARIOS = {
    "optimist_ssp126": {
        "label": "İyimser (SSP1-2.6)",
        "precip_trend": -0.003,   # yıllık % değişim (yağış azalması)
        "temp_trend":   +0.025,   # yıllık °C artış
        "ndvi_trend":   -0.001,   # yıllık NDVI değişimi
        "color": "#22c55e"
    },
    "orta_ssp245": {
        "label": "Orta (SSP2-4.5)",
        "precip_trend": -0.006,
        "temp_trend":   +0.045,
        "ndvi_trend":   -0.002,
        "color": "#f59e0b"
    },
    "kotumser_ssp585": {
        "label": "Kötümser (SSP5-8.5)",
        "precip_trend": -0.012,
        "temp_trend":   +0.080,
        "ndvi_trend":   -0.004,
        "color": "#ef4444"
    },
}

# Bölgesel hassasiyet katsayıları (kuraklığa duyarlılık)
BOLGE_HASSASIYET = {
    "İç Anadolu": 1.3,
    "Güneydoğu Anadolu": 1.4,
    "Doğu Anadolu": 1.2,
    "Akdeniz": 1.1,
    "Ege": 1.0,
    "Marmara": 0.8,
    "Karadeniz": 0.7,
}

IL_BOLGE = {
    "Ankara":"İç Anadolu","Konya":"İç Anadolu","Eskişehir":"İç Anadolu",
    "Kırıkkale":"İç Anadolu","Aksaray":"İç Anadolu","Niğde":"İç Anadolu",
    "Nevşehir":"İç Anadolu","Kırşehir":"İç Anadolu","Kayseri":"İç Anadolu",
    "Sivas":"İç Anadolu","Yozgat":"İç Anadolu","Çankırı":"İç Anadolu",
    "Karaman":"İç Anadolu",
    "Şanlıurfa":"Güneydoğu Anadolu","Diyarbakır":"Güneydoğu Anadolu",
    "Mardin":"Güneydoğu Anadolu","Batman":"Güneydoğu Anadolu",
    "Şırnak":"Güneydoğu Anadolu","Siirt":"Güneydoğu Anadolu",
    "Gaziantep":"Güneydoğu Anadolu","Adıyaman":"Güneydoğu Anadolu",
    "Kilis":"Güneydoğu Anadolu","Osmaniye":"Güneydoğu Anadolu",
    "Erzurum":"Doğu Anadolu","Erzincan":"Doğu Anadolu","Bayburt":"Doğu Anadolu",
    "Ağrı":"Doğu Anadolu","Kars":"Doğu Anadolu","Iğdır":"Doğu Anadolu",
    "Ardahan":"Doğu Anadolu","Malatya":"Doğu Anadolu","Elazığ":"Doğu Anadolu",
    "Bingöl":"Doğu Anadolu","Tunceli":"Doğu Anadolu","Van":"Doğu Anadolu",
    "Muş":"Doğu Anadolu","Bitlis":"Doğu Anadolu","Hakkari":"Doğu Anadolu",
    "Antalya":"Akdeniz","Mersin":"Akdeniz","Adana":"Akdeniz",
    "Hatay":"Akdeniz","Kahramanmaraş":"Akdeniz","Isparta":"Akdeniz","Burdur":"Akdeniz",
    "İzmir":"Ege","Aydın":"Ege","Denizli":"Ege","Muğla":"Ege",
    "Manisa":"Ege","Afyonkarahisar":"Ege","Kütahya":"Ege","Uşak":"Ege",
    "İstanbul":"Marmara","Tekirdağ":"Marmara","Edirne":"Marmara",
    "Kırklareli":"Marmara","Balıkesir":"Marmara","Çanakkale":"Marmara",
    "Bursa":"Marmara","Bilecik":"Marmara","Kocaeli":"Marmara",
    "Sakarya":"Marmara","Yalova":"Marmara",
    "Trabzon":"Karadeniz","Ordu":"Karadeniz","Giresun":"Karadeniz",
    "Rize":"Karadeniz","Artvin":"Karadeniz","Gümüşhane":"Karadeniz",
    "Samsun":"Karadeniz","Tokat":"Karadeniz","Çorum":"Karadeniz",
    "Amasya":"Karadeniz","Sinop":"Karadeniz","Kastamonu":"Karadeniz",
    "Zonguldak":"Karadeniz","Karabük":"Karadeniz","Bartın":"Karadeniz",
    "Bolu":"Karadeniz","Düzce":"Karadeniz",
}


def run_forecast(processed_dir="data/processed", out_dir="data/processed"):
    print("=" * 60)
    print("AGRI-2050 | Türkiye 2025-2050 İklim Senaryoları")
    print("=" * 60 + "\n")

    scores = pd.read_csv(f"{processed_dir}/turkey_il_scores.csv")
    print(f"  ✓ {len(scores)} il yüklendi\n")

    records = []
    yillar = list(range(2025, 2051))

    for _, row in scores.iterrows():
        il = row['il']
        baz_skor   = row['tarim_skoru']
        baz_ndvi   = row['ndvi_ort']
        baz_yagis  = row['yagis_ort']
        bolge      = IL_BOLGE.get(il, "İç Anadolu")
        hassasiyet = BOLGE_HASSASIYET.get(bolge, 1.0)

        for sc_key, sc in SCENARIOS.items():
            for yil in yillar:
                dt = yil - 2024  # 2024 baz yıl

                # Projeksiyon
                yagis_proj = baz_yagis * (1 + sc['precip_trend'] * dt)
                ndvi_proj  = max(0.05, baz_ndvi + sc['ndvi_trend'] * dt)

                # Kuraklık çarpanı
                yagis_oran = yagis_proj / baz_yagis
                kuraklık_carpan = 1.0
                if yagis_oran < 0.85:
                    kuraklık_carpan = 0.85 + (yagis_oran - 0.85) * hassasiyet
                elif yagis_oran < 0.95:
                    kuraklık_carpan = 0.95 + (yagis_oran - 0.95) * (hassasiyet * 0.5)

                # Sıcaklık stresi (>35°C gün etkisi proxy)
                temp_stress = 1.0 - max(0, sc['temp_trend'] * dt - 0.5) * 0.02 * hassasiyet

                # Final skor
                skor = baz_skor * kuraklık_carpan * temp_stress
                skor = max(0.05, skor)

                # Güven aralığı
                belirsizlik = 0.05 + dt * 0.008
                skor_ust = skor * (1 + belirsizlik)
                skor_alt = skor * (1 - belirsizlik)

                records.append({
                    "il": il,
                    "bolge": bolge,
                    "year": yil,
                    "scenario": sc_key,
                    "scenario_label": sc['label'],
                    "tarim_skoru": round(skor, 4),
                    "skor_ust": round(skor_ust, 4),
                    "skor_alt": round(skor_alt, 4),
                    "yagis_proj": round(yagis_proj, 1),
                    "ndvi_proj": round(ndvi_proj, 4),
                    "hassasiyet": hassasiyet,
                })

    df = pd.DataFrame(records)
    out_path = f"{out_dir}/forecast_turkey_2025_2050.csv"
    df.to_csv(out_path, index=False)

    toplam = len(df)
    print(f"  ✓ {toplam:,} projeksiyon kaydı oluşturuldu")
    print(f"  ✓ {df['il'].nunique()} il × {len(yillar)} yıl × {len(SCENARIOS)} senaryo\n")

    # Özet: 2050'de en çok etkilenen iller
    df_2050 = df[df['year'] == 2050]
    baz = scores[['il','tarim_skoru']].rename(columns={'tarim_skoru':'baz_skor'})
    df_2050 = df_2050.merge(baz, on='il')
    df_2050['degisim_pct'] = (df_2050['tarim_skoru'] - df_2050['baz_skor']) / df_2050['baz_skor'] * 100

    print("  2050 Kötümser Senaryo — En Fazla Etkilenen 5 İl:")
    kotumser_2050 = df_2050[df_2050['scenario'] == 'kotumser_ssp585'].nsmallest(5, 'degisim_pct')
    for _, r in kotumser_2050.iterrows():
        print(f"    {r['il']:<20} {r['degisim_pct']:+.1f}%")

    print("\n  2050 İyimser Senaryo — En Az Etkilenen 5 İl:")
    iyimser_2050 = df_2050[df_2050['scenario'] == 'optimist_ssp126'].nlargest(5, 'degisim_pct')
    for _, r in iyimser_2050.iterrows():
        print(f"    {r['il']:<20} {r['degisim_pct']:+.1f}%")

    print(f"\n  ✓ Kaydedildi: {out_path}")
    print("\n[✓] Faz 4 tamamlandı.\n")
    return df


if __name__ == "__main__":
    os.chdir(os.path.join(os.path.dirname(__file__), ".."))
    run_forecast()
