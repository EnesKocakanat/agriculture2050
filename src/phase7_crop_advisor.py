"""
AGRI-2050: Faz 7 — Ürün Öneri Motoru + Sulama Planı
=====================================================
İlçe bazlı iklim verisiyle en uygun ürün ve sulama önerisi.
Çalıştırma: python src/phase7_crop_advisor.py
=====================================================
"""

import os, warnings
import pandas as pd
import numpy as np
import json
warnings.filterwarnings("ignore")

# ── Ürün-İklim Veritabanı (FAO bazlı) ────────────────────────────────
# Her ürün için: min/max yağış (mm/yıl), min/max sıcaklık (°C),
# NDVI eşiği, sulama ihtiyacı (mm/yıl), sulama yöntemi, hasat ayı
URUN_VERITABANI = {
    "Buğday": {
        "kategori": "Tahıl",
        "yagis_min": 250, "yagis_max": 1000,
        "ndvi_min": 0.2, "ndvi_max": 0.7,
        "sulama_ihtiyac_mm": 450,
        "sulama_yontem": "Yağmurlama",
        "ekim_ay": [10, 11],
        "hasat_ay": [6, 7],
        "bakim": "Sonbaharda ekilir, ilkbaharda gübrelenir. Kuru tarıma uygun.",
        "kar_potansiyel": "Orta",
        "emoji": "🌾"
    },
    "Arpa": {
        "kategori": "Tahıl",
        "yagis_min": 200, "yagis_max": 900,
        "ndvi_min": 0.2, "ndvi_max": 0.65,
        "sulama_ihtiyac_mm": 350,
        "sulama_yontem": "Yağmurlama",
        "ekim_ay": [10, 11],
        "hasat_ay": [6, 7],
        "bakim": "Kuraklığa dayanıklı. Az sulama ile yetişir.",
        "kar_potansiyel": "Orta",
        "emoji": "🌾"
    },
    "Mısır": {
        "kategori": "Tahıl",
        "yagis_min": 500, "yagis_max": 1200,
        "ndvi_min": 0.4, "ndvi_max": 0.9,
        "sulama_ihtiyac_mm": 600,
        "sulama_yontem": "Damla veya Yağmurlama",
        "ekim_ay": [4, 5],
        "hasat_ay": [9, 10],
        "bakim": "Yüksek su ihtiyacı. Sıcak iklim sever. Düzenli sulama şart.",
        "kar_potansiyel": "Yüksek",
        "emoji": "🌽"
    },
    "Ayçiçeği": {
        "kategori": "Yağlı Tohum",
        "yagis_min": 350, "yagis_max": 900,
        "ndvi_min": 0.3, "ndvi_max": 0.8,
        "sulama_ihtiyac_mm": 400,
        "sulama_yontem": "Yağmurlama",
        "ekim_ay": [4, 5],
        "hasat_ay": [8, 9],
        "bakim": "Kuraklığa orta dayanıklı. Derin köklü, az sulama yeterli.",
        "kar_potansiyel": "Orta-Yüksek",
        "emoji": "🌻"
    },
    "Şeker Pancarı": {
        "kategori": "Endüstriyel",
        "yagis_min": 400, "yagis_max": 900,
        "ndvi_min": 0.35, "ndvi_max": 0.8,
        "sulama_ihtiyac_mm": 550,
        "sulama_yontem": "Damla",
        "ekim_ay": [3, 4],
        "hasat_ay": [10, 11],
        "bakim": "Derin toprak sever. Düzenli sulama gerekli. Fabrika sözleşmeli.",
        "kar_potansiyel": "Yüksek",
        "emoji": "🟫"
    },
    "Domates": {
        "kategori": "Sebze",
        "yagis_min": 400, "yagis_max": 1200,
        "ndvi_min": 0.4, "ndvi_max": 0.9,
        "sulama_ihtiyac_mm": 700,
        "sulama_yontem": "Damla",
        "ekim_ay": [3, 4, 5],
        "hasat_ay": [7, 8, 9],
        "bakim": "Sıcak iklim sever. Damla sulama ile %40 su tasarrufu. Hastalığa dikkat.",
        "kar_potansiyel": "Yüksek",
        "emoji": "🍅"
    },
    "Biber": {
        "kategori": "Sebze",
        "yagis_min": 400, "yagis_max": 1100,
        "ndvi_min": 0.4, "ndvi_max": 0.85,
        "sulama_ihtiyac_mm": 650,
        "sulama_yontem": "Damla",
        "ekim_ay": [3, 4],
        "hasat_ay": [7, 8, 9, 10],
        "bakim": "Sıcak ve güneşli sever. Düzenli sulama. Sera ile yıl uzatılabilir.",
        "kar_potansiyel": "Yüksek",
        "emoji": "🌶️"
    },
    "Elma": {
        "kategori": "Meyve",
        "yagis_min": 500, "yagis_max": 1200,
        "ndvi_min": 0.45, "ndvi_max": 0.85,
        "sulama_ihtiyac_mm": 600,
        "sulama_yontem": "Damla",
        "ekim_ay": [2, 3],
        "hasat_ay": [8, 9, 10],
        "bakim": "Soğuk kış gerektirir. Yüksek rakımda iyi gelişir. Budama önemli.",
        "kar_potansiyel": "Yüksek",
        "emoji": "🍎"
    },
    "Üzüm (Bağ)": {
        "kategori": "Meyve",
        "yagis_min": 300, "yagis_max": 900,
        "ndvi_min": 0.3, "ndvi_max": 0.8,
        "sulama_ihtiyac_mm": 450,
        "sulama_yontem": "Damla",
        "ekim_ay": [2, 3],
        "hasat_ay": [8, 9, 10],
        "bakim": "Kuraklığa dayanıklı. Az sulama ile kaliteli ürün. Budama kritik.",
        "kar_potansiyel": "Yüksek",
        "emoji": "🍇"
    },
    "Zeytin": {
        "kategori": "Meyve",
        "yagis_min": 350, "yagis_max": 800,
        "ndvi_min": 0.3, "ndvi_max": 0.7,
        "sulama_ihtiyac_mm": 350,
        "sulama_yontem": "Damla",
        "ekim_ay": [10, 11],
        "hasat_ay": [10, 11, 12],
        "bakim": "Akdeniz iklimi sever. Az bakım. Dondan korunmalı (-7°C altı tehlikeli).",
        "kar_potansiyel": "Yüksek",
        "emoji": "🫒"
    },
    "Çay": {
        "kategori": "Endüstriyel",
        "yagis_min": 1200, "yagis_max": 3000,
        "ndvi_min": 0.6, "ndvi_max": 0.95,
        "sulama_ihtiyac_mm": 0,  # yağışla yeterli
        "sulama_yontem": "Yağışa bağlı (sulama gerekmez)",
        "ekim_ay": [3, 4],
        "hasat_ay": [5, 6, 7, 8, 9],
        "bakim": "Asidik toprak sever. Çok yağış şart. Karadeniz kıyısına özgü.",
        "kar_potansiyel": "Yüksek",
        "emoji": "🍵"
    },
    "Fındık": {
        "kategori": "Sert Kabuklu",
        "yagis_min": 700, "yagis_max": 2000,
        "ndvi_min": 0.5, "ndvi_max": 0.9,
        "sulama_ihtiyac_mm": 200,
        "sulama_yontem": "Yağışa bağlı / Damla",
        "ekim_ay": [11, 12],
        "hasat_ay": [8, 9],
        "bakim": "Karadeniz iklimine uygun. Eğimli arazide yetişir. Az bakım.",
        "kar_potansiyel": "Çok Yüksek",
        "emoji": "🌰"
    },
    "Pamuk": {
        "kategori": "Endüstriyel",
        "yagis_min": 500, "yagis_max": 1500,
        "ndvi_min": 0.4, "ndvi_max": 0.85,
        "sulama_ihtiyac_mm": 800,
        "sulama_yontem": "Damla veya Yüzey",
        "ekim_ay": [4, 5],
        "hasat_ay": [9, 10, 11],
        "bakim": "Sıcak iklim şart. Çukurova ve GAP bölgesine uygun. Çok sulama.",
        "kar_potansiyel": "Yüksek",
        "emoji": "🌿"
    },
    "Soğan": {
        "kategori": "Sebze",
        "yagis_min": 300, "yagis_max": 900,
        "ndvi_min": 0.25, "ndvi_max": 0.7,
        "sulama_ihtiyac_mm": 400,
        "sulama_yontem": "Yağmurlama veya Damla",
        "ekim_ay": [3, 4, 9, 10],
        "hasat_ay": [6, 7, 8],
        "bakim": "Kuru ve sıcak iklimde iyi verim. İç Anadolu'ya uygun.",
        "kar_potansiyel": "Orta",
        "emoji": "🧅"
    },
    "Patates": {
        "kategori": "Sebze",
        "yagis_min": 400, "yagis_max": 1000,
        "ndvi_min": 0.35, "ndvi_max": 0.8,
        "sulama_ihtiyac_mm": 500,
        "sulama_yontem": "Yağmurlama",
        "ekim_ay": [3, 4],
        "hasat_ay": [7, 8, 9],
        "bakim": "Serin iklim sever. Yüksek rakımda kaliteli. Düzenli sulama.",
        "kar_potansiyel": "Orta-Yüksek",
        "emoji": "🥔"
    },
}

AY_ISIMLERI = {
    1:"Ocak", 2:"Şubat", 3:"Mart", 4:"Nisan", 5:"Mayıs", 6:"Haziran",
    7:"Temmuz", 8:"Ağustos", 9:"Eylül", 10:"Ekim", 11:"Kasım", 12:"Aralık"
}

SULAMA_DETAY = {
    "Damla": {
        "verimlilik": "%90-95",
        "su_tasarrufu": "%40-50 (yüzey sulamaya göre)",
        "maliyet_ha": "15,000-25,000 ₺",
        "uygun_urunler": "Sebze, meyve, bağ-bahçe",
        "avantaj": "En verimli yöntem. Kök bölgesine direkt su. Hastalık riski düşük.",
    },
    "Yağmurlama": {
        "verimlilik": "%70-80",
        "su_tasarrufu": "%20-30",
        "maliyet_ha": "8,000-15,000 ₺",
        "uygun_urunler": "Tahıl, çayır, sebze",
        "avantaj": "Geniş alanlara uygun. Kurulum kolay. Don koruması sağlar.",
    },
    "Yüzey": {
        "verimlilik": "%40-60",
        "su_tasarrufu": "—",
        "maliyet_ha": "2,000-5,000 ₺",
        "uygun_urunler": "Pirinç, pamuk, şeker pancarı",
        "avantaj": "Düz arazilerde ucuz. Ama çok su israf eder.",
    },
}


def urun_puanla(urun_adi, urun, yagis_ort, ndvi_ort):
    """Bir ürünün ilçeye uygunluk puanını hesapla (0-100)"""
    puan = 0

    # Yağış uyumu
    if urun["yagis_min"] <= yagis_ort <= urun["yagis_max"]:
        # Optimum aralıkta
        opt_merkez = (urun["yagis_min"] + urun["yagis_max"]) / 2
        uzaklik = abs(yagis_ort - opt_merkez) / (urun["yagis_max"] - urun["yagis_min"])
        puan += 50 * (1 - uzaklik)
    elif yagis_ort < urun["yagis_min"]:
        eksik = (urun["yagis_min"] - yagis_ort) / urun["yagis_min"]
        puan += max(0, 30 * (1 - eksik * 2))
    else:
        fazla = (yagis_ort - urun["yagis_max"]) / urun["yagis_max"]
        puan += max(0, 20 * (1 - fazla))

    # NDVI uyumu
    if urun["ndvi_min"] <= ndvi_ort <= urun["ndvi_max"]:
        puan += 50
    elif ndvi_ort < urun["ndvi_min"]:
        eksik = (urun["ndvi_min"] - ndvi_ort) / urun["ndvi_min"]
        puan += max(0, 30 * (1 - eksik * 2))
    else:
        puan += 30

    return round(min(100, puan), 1)


def ilce_urun_onerileri(il, ilce, yagis_ort, ndvi_ort, top_n=5):
    """İlçe için en uygun ürünleri öner"""
    puanlar = []
    for urun_adi, urun in URUN_VERITABANI.items():
        puan = urun_puanla(urun_adi, urun, yagis_ort, ndvi_ort)
        puanlar.append({
            "urun": urun_adi,
            "kategori": urun["kategori"],
            "uygunluk_pct": puan,
            "sulama_ihtiyac_mm": urun["sulama_ihtiyac_mm"],
            "sulama_yontem": urun["sulama_yontem"],
            "ekim_aylari": ", ".join([AY_ISIMLERI[m] for m in urun["ekim_ay"]]),
            "hasat_aylari": ", ".join([AY_ISIMLERI[m] for m in urun["hasat_ay"]]),
            "bakim": urun["bakim"],
            "kar_potansiyel": urun["kar_potansiyel"],
            "emoji": urun["emoji"],
            "il": il,
            "ilce": ilce,
        })

    df = pd.DataFrame(puanlar).sort_values("uygunluk_pct", ascending=False)
    return df.head(top_n).reset_index(drop=True)


def run_crop_advisor(processed_dir="data/processed", raw_dir="data/raw"):
    print("=" * 60)
    print("AGRI-2050 | Ürün Öneri Motoru")
    print("=" * 60 + "\n")

    # İlçe skorları yükle
    ilce_scores = pd.read_csv(os.path.join(processed_dir, "ilce_scores.csv"))

    print("[1/3] Tüm ilçeler için ürün önerileri hesaplanıyor...")
    all_recs = []
    for _, row in ilce_scores.iterrows():
        recs = ilce_urun_onerileri(
            il=row['il'],
            ilce=row['ilce'],
            yagis_ort=row.get('yagis_ort', 400),
            ndvi_ort=row.get('ndvi_ort', 0.35),
            top_n=5
        )
        all_recs.append(recs)

    df_all = pd.concat(all_recs, ignore_index=True)

    out_path = os.path.join(processed_dir, "ilce_urun_onerileri.csv")
    df_all.to_csv(out_path, index=False)
    print(f"    ✓ {len(df_all)} öneri kaydedildi → {out_path}")

    # Sulama detayları kaydet
    sulama_df = pd.DataFrame([
        {"yontem": k, **v} for k, v in SULAMA_DETAY.items()
    ])
    sulama_path = os.path.join(processed_dir, "sulama_detay.csv")
    sulama_df.to_csv(sulama_path, index=False)

    # Ürün veritabanını JSON olarak kaydet
    urun_path = os.path.join(processed_dir, "urun_veritabani.json")
    with open(urun_path, "w", encoding="utf-8") as f:
        json.dump(URUN_VERITABANI, f, ensure_ascii=False, indent=2)

    print(f"    ✓ Sulama detayları → {sulama_path}")
    print(f"    ✓ Ürün veritabanı → {urun_path}")

    print("\n[2/3] Örnek öneriler:")
    for test_il, test_ilce in [("Konya","Merkez"),("Rize","Merkez"),("Şanlıurfa","Merkez"),("Antalya","Merkez")]:
        row = ilce_scores[(ilce_scores['il']==test_il) & (ilce_scores['ilce'].str.contains('Merkez', na=False))]
        if len(row) == 0:
            row = ilce_scores[ilce_scores['il']==test_il].head(1)
        if len(row) == 0:
            continue
        row = row.iloc[0]
        recs = ilce_urun_onerileri(test_il, row['ilce'], row.get('yagis_ort',400), row.get('ndvi_ort',0.35), top_n=3)
        print(f"\n  {test_il} — {row['ilce']} (yağış:{row.get('yagis_ort',0):.0f}mm, NDVI:{row.get('ndvi_ort',0):.3f}):")
        for _, r in recs.iterrows():
            print(f"    {r['emoji']} {r['urun']:<20} Uygunluk: %{r['uygunluk_pct']:.0f}  Sulama: {r['sulama_yontem']}")

    print(f"\n[✓] Ürün öneri motoru tamamlandı.\n")
    return df_all


if __name__ == "__main__":
    os.chdir(os.path.join(os.path.dirname(__file__), ".."))
    run_crop_advisor()
