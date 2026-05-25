# AGRI-2050 | Konya Ovası Rekolte Tahmin Sistemi

## 📁 Proje Yapısı

```
agri2050/
│
├── data/
│   ├── raw/              ← Ham veriler (uydu, yağış, rekolte CSV)
│   ├── processed/        ← Temizlenmiş, birleştirilmiş veriler
│   └── external/         ← Dış kaynak veriler (FAO, TÜİK vb.)
│
├── notebooks/            ← Jupyter Notebook'lar (keşif & analiz)
│
├── src/
│   ├── phase1_data_fetch.py   ← Uydu veri çekme (GEE + Kaggle)
│   └── phase2_ml_model.py     ← ML model eğitimi (Ensemble)
│
├── dashboard/
│   └── phase3_dashboard.py    ← Streamlit arayüzü
│
├── models/               ← Eğitilmiş model dosyaları (.pkl)
├── reports/              ← Çıktı grafik ve raporlar
│
├── run_all.py            ← Tüm pipeline'ı çalıştırır
└── requirements.txt      ← Bağımlılıklar
```

## 🚀 Kurulum & Çalıştırma

```bash
# 1. Sanal ortam oluştur
python -m venv venv
source venv/bin/activate      # Mac/Linux
venv\Scripts\activate         # Windows

# 2. Bağımlılıkları yükle
pip install -r requirements.txt

# 3. Pipeline'ı çalıştır (Faz 1 + 2)
python run_all.py

# 4. Dashboard'u başlat
streamlit run dashboard/phase3_dashboard.py
```

## 📊 Veri Akışı

```
data/raw/          →   src/phase1_data_fetch.py
                              ↓
                   data/processed/
                              ↓
                   src/phase2_ml_model.py
                              ↓
                          models/
                              ↓
                   dashboard/phase3_dashboard.py
```
