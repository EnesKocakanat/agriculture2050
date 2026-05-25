"""
AGRI-2050: Faz 5 — Akıllı Sulama Öneri Sistemi
===============================================
Kuraklık riskli iller için sulama ihtiyacı hesaplama.
Çalıştırma: python src/phase5_irrigation.py
===============================================
"""

import os, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

# ── Ürün su ihtiyaçları (mm/yıl) ─────────────────────────────────────
URUN_SU_IHTIYACI = {
    "bugday":          450,
    "arpa":            350,
    "misir":           600,
    "seker_pancari":   700,
    "pamuk":           900,
    "domates":         500,
    "elma":            600,
    "uzum":            400,
}

# ── Sulama yöntemi verimlilikleri ─────────────────────────────────────
SULAMA_YONTEMLERI = {
    "Yüzey Sulama":     {"verimlilik": 0.50, "maliyet_ha": 800},
    "Yağmurlama":       {"verimlilik": 0.75, "maliyet_ha": 1800},
    "Damla Sulama":     {"verimlilik": 0.90, "maliyet_ha": 3500},
    "Akıllı Damla":     {"verimlilik": 0.95, "maliyet_ha": 5500},
}

# ── Bölge bazlı öncelikli ürünler ─────────────────────────────────────
BOLGE_URUN = {
    "İç Anadolu":            "bugday",
    "Güneydoğu Anadolu":     "bugday",
    "Doğu Anadolu":          "arpa",
    "Akdeniz":               "pamuk",
    "Ege":                   "uzum",
    "Marmara":               "domates",
    "Karadeniz":             "misir",
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


def sulama_ihtiyaci_hesapla(yagis_mm, urun, bolge):
    """Net sulama ihtiyacını hesapla (mm/yıl)"""
    et0 = URUN_SU_IHTIYACI.get(urun, 500)  # referans ET
    
    # Yağış etkinliği (Doorenbos & Pruitt yöntemi basitleştirilmiş)
    if yagis_mm < 200:
        etkin_yagis = yagis_mm * 0.60
    elif yagis_mm < 400:
        etkin_yagis = yagis_mm * 0.70
    elif yagis_mm < 600:
        etkin_yagis = yagis_mm * 0.80
    else:
        etkin_yagis = yagis_mm * 0.85

    net_sulama = max(0, et0 - etkin_yagis)
    return round(net_sulama, 1)


def oneri_olustur(il, yagis_mm, sulama_mm, skor, bolge):
    """İl bazlı sulama önerisi"""
    urun = BOLGE_URUN.get(bolge, "bugday")
    
    # Risk seviyesi
    if yagis_mm < 300:
        risk = "KRİTİK"
        oncelik = 1
    elif yagis_mm < 450:
        risk = "YÜKSEK"
        oncelik = 2
    elif yagis_mm < 600:
        risk = "ORTA"
        oncelik = 3
    else:
        risk = "DÜŞÜK"
        oncelik = 4

    # Sulama yöntemi önerisi
    if sulama_mm > 300:
        yontem = "Akıllı Damla"
    elif sulama_mm > 150:
        yontem = "Damla Sulama"
    elif sulama_mm > 50:
        yontem = "Yağmurlama"
    else:
        yontem = "Yüzey Sulama"

    yontem_bilgi = SULAMA_YONTEMLERI[yontem]
    
    # Su tasarrufu potansiyeli (yüzey sulama baz alınarak)
    baz_su = sulama_mm / SULAMA_YONTEMLERI["Yüzey Sulama"]["verimlilik"]
    oneri_su = sulama_mm / yontem_bilgi["verimlilik"]
    tasarruf_pct = (1 - oneri_su / baz_su) * 100 if baz_su > 0 else 0

    return {
        "il": il,
        "bolge": bolge,
        "risk_seviyesi": risk,
        "oncelik": oncelik,
        "yagis_mm": yagis_mm,
        "sulama_ihtiyaci_mm": sulama_mm,
        "oncelikli_urun": urun,
        "onerilen_yontem": yontem,
        "yontem_verimliligi_pct": int(yontem_bilgi["verimlilik"] * 100),
        "yatirim_maliyet_ha_tl": yontem_bilgi["maliyet_ha"],
        "su_tasarrufu_pct": round(tasarruf_pct, 1),
        "tarim_skoru": round(skor, 3),
    }


def run_irrigation_plan(processed_dir="data/processed", raw_dir="data/raw"):
    print("=" * 60)
    print("AGRI-2050 | Akıllı Sulama Öneri Sistemi")
    print("=" * 60 + "\n")

    scores = pd.read_csv(f"{processed_dir}/turkey_il_scores.csv")
    precip = pd.read_csv(f"{raw_dir}/precipitation_iller.csv")

    # Yıllık ortalama yağış
    yagis_ort = (precip.groupby(['il','year'])['precip_mm']
                 .sum().reset_index()
                 .groupby('il')['precip_mm'].mean()
                 .reset_index())
    yagis_ort.columns = ['il','yagis_ort']

    df = scores.merge(yagis_ort, on='il', how='left', suffixes=('','_precip'))
    df['yagis_ort'] = df['yagis_ort'].fillna(df.get('yagis_ort_precip', df['yagis_ort']))
    df['bolge'] = df['il'].map(IL_BOLGE).fillna('İç Anadolu')

    print("[1/3] Sulama ihtiyaçları hesaplanıyor...")
    oneriler = []
    for _, row in df.iterrows():
        bolge = row['bolge']
        urun  = BOLGE_URUN.get(bolge, "bugday")
        sulama_mm = sulama_ihtiyaci_hesapla(row['yagis_ort'], urun, bolge)
        oneri = oneri_olustur(row['il'], row['yagis_ort'], sulama_mm,
                               row['tarim_skoru'], bolge)
        oneriler.append(oneri)

    df_oneri = pd.DataFrame(oneriler).sort_values(['oncelik','sulama_ihtiyaci_mm'],
                                                    ascending=[True, False])

    out_path = f"{processed_dir}/irrigation_plan.csv"
    df_oneri.to_csv(out_path, index=False)

    print(f"    ✓ {len(df_oneri)} il için sulama planı oluşturuldu\n")

    print("[2/3] Risk Dağılımı:")
    for risk in ["KRİTİK","YÜKSEK","ORTA","DÜŞÜK"]:
        n = (df_oneri['risk_seviyesi'] == risk).sum()
        bar = "█" * n
        print(f"    {risk:<10} {bar} ({n} il)")

    print("\n[3/3] Kritik İller (İlk 10):")
    kritik = df_oneri[df_oneri['risk_seviyesi'].isin(['KRİTİK','YÜKSEK'])].head(10)
    for _, r in kritik.iterrows():
        print(f"    {r['il']:<20} {r['risk_seviyesi']:<10} "
              f"Sulama: {r['sulama_ihtiyaci_mm']:>4.0f}mm  "
              f"Öneri: {r['onerilen_yontem']}")

    print(f"\n  ✓ Kaydedildi: {out_path}")
    print("\n[✓] Sulama planı tamamlandı.\n")
    return df_oneri


if __name__ == "__main__":
    os.chdir(os.path.join(os.path.dirname(__file__), ".."))
    run_irrigation_plan()
