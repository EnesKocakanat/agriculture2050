"""
AGRI-2050: Faz 2b — Türkiye Geneli ML Model Eğitimi
====================================================
81 il için gerçek veri ile rekolte tahmini.
Çalıştırma: python src/phase2_ml_turkey.py
====================================================
"""

import os, json, pickle, warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, r2_score
warnings.filterwarnings("ignore")

def load_and_merge(raw_dir="data/raw", processed_dir="data/processed", external_dir="data/external"):
    print("[1/4] Veriler yükleniyor...")

    # TÜİK tarım alanları
    tuik = pd.read_csv(f"{processed_dir}/tuik_iller_clean.csv")
    tuik = tuik[['il','ekilen_alan_dekar']].copy()
    tuik['ekilen_ha'] = (tuik['ekilen_alan_dekar'] / 10).round(0).astype(int)

    # Yağış verisi
    precip = pd.read_csv(f"{raw_dir}/precipitation_iller.csv")
    # drought_year kolonunu bul
    drought_col = 'drought_year' if 'drought_year' in precip.columns else 'drought_year_x'
    precip_yearly = (precip.groupby(['il','year'])
                     .agg(precip_mm=('precip_mm','sum'),
                          drought_year=(drought_col,'first'))
                     .reset_index())

    # NDVI verisi
    ndvi = pd.read_csv(f"{raw_dir}/satellite_ndvi_iller.csv")
    ndvi_yearly = (ndvi.groupby(['il','year'])
                   .agg(ndvi_mean=('ndvi','mean'),
                        ndvi_peak=('ndvi','max'))
                   .reset_index())

    # Birleştir
    merged = precip_yearly.merge(ndvi_yearly, on=['il','year'], how='inner')
    merged = merged.merge(tuik[['il','ekilen_ha']], on='il', how='left')

    print(f"    ✓ {merged['il'].nunique()} il | {len(merged)} kayıt")
    print(f"    ✓ Yıllar: {merged['year'].min()} - {merged['year'].max()}")
    return merged


def create_features(df):
    df = df.copy()
    df['drought_index']   = df['precip_mm'] / df.groupby('il')['precip_mm'].transform('mean')
    df['ndvi_stress']     = 1 - df['ndvi_mean']
    df['water_deficit']   = (df['precip_mm'] < 270).astype(int)
    df['heat_proxy']      = df['ndvi_mean'] * df['drought_index']
    df['year_trend']      = df['year'] - df['year'].min()
    df['drought_year']    = df['drought_year'].fillna(False).astype(int)
    df['ndvi_lag1']       = df.groupby('il')['ndvi_mean'].shift(1)
    df['precip_lag1']     = df.groupby('il')['precip_mm'].shift(1)
    df['ekilen_ha_log']   = np.log1p(df['ekilen_ha'])
    return df.dropna()

FEATURE_COLS = [
    'ndvi_mean','ndvi_peak','ndvi_stress','ndvi_lag1',
    'precip_mm','drought_index','water_deficit','precip_lag1',
    'heat_proxy','drought_year','year_trend','ekilen_ha_log'
]

def train_model(X, y):
    models = {
        'rf':  RandomForestRegressor(n_estimators=300, max_depth=8, random_state=42),
        'gbm': GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, random_state=42),
        'ridge': Ridge(alpha=1.0),
    }
    weights = {'rf': 0.35, 'gbm': 0.40, 'ridge': 0.25}
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)
    tscv = TimeSeriesSplit(n_splits=4)

    print("\n  Model Eğitimi:")
    for name, m in models.items():
        m.fit(X_sc, y)
        cv = cross_val_score(m, X_sc, y, cv=tscv, scoring='neg_mean_absolute_error')
        print(f"    [{name:6s}] CV-MAE: {-cv.mean():.4f} ± {cv.std():.4f}")

    return models, weights, scaler


def predict_ensemble(models, weights, scaler, X):
    X_sc = scaler.transform(X)
    return sum(m.predict(X_sc) * weights[n] for n, m in models.items())


def run_turkey_pipeline(raw_dir="data/raw", processed_dir="data/processed",
                        external_dir="data/external", model_dir="models"):
    os.makedirs(model_dir, exist_ok=True)
    print("=" * 60)
    print("AGRI-2050 | Türkiye Geneli ML Model Eğitimi")
    print("=" * 60 + "\n")

    # Veri yükle ve birleştir
    df = load_and_merge(raw_dir, processed_dir, external_dir)

    # Feature engineering
    print("[2/4] Özellikler hazırlanıyor...")
    df = create_features(df)
    print(f"    ✓ {len(FEATURE_COLS)} özellik | {len(df)} örnek")

    X = df[FEATURE_COLS].values
    # Hedef: normalize edilmiş NDVI skoru (rekolte proxy)
    # Gerçek rekolte verisi olmadığından NDVI × yağış × alan skoru kullanıyoruz
    y = (df['ndvi_mean'] * df['drought_index'] * np.log1p(df['ekilen_ha'])).values

    # Train/test split
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    print("\n[3/4] Ensemble model eğitiliyor...")
    models, weights, scaler = train_model(X_train, y_train)

    # Test sonuçları
    y_pred = predict_ensemble(models, weights, scaler, X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2  = r2_score(y_test, y_pred)
    print(f"\n  Test MAE: {mae:.4f} | R²: {r2:.4f}")

    # İl bazlı skorlar hesapla
    print("\n[4/4] İl bazlı tarım skorları hesaplanıyor...")
    df['tarim_skoru'] = predict_ensemble(models, weights, scaler, X)

    il_skorlar = (df.groupby('il')
                  .agg(tarim_skoru=('tarim_skoru','mean'),
                       ndvi_ort=('ndvi_mean','mean'),
                       yagis_ort=('precip_mm','mean'),
                       kuraklık_yil=('drought_year','sum'))
                  .reset_index()
                  .sort_values('tarim_skoru', ascending=False))

    out_path = f"{processed_dir}/turkey_il_scores.csv"
    il_skorlar.to_csv(out_path, index=False)
    print(f"    ✓ turkey_il_scores.csv → {len(il_skorlar)} il")

    print("\n  En yüksek tarım skoru:")
    for _, r in il_skorlar.head(5).iterrows():
        print(f"    {r['il']:<20} skor: {r['tarim_skoru']:.3f}")

    print("\n  En düşük tarım skoru:")
    for _, r in il_skorlar.tail(5).iterrows():
        print(f"    {r['il']:<20} skor: {r['tarim_skoru']:.3f}")

    # Modeli kaydet
    with open(f"{model_dir}/turkey_model.pkl", "wb") as f:
        pickle.dump({'models': models, 'weights': weights,
                     'scaler': scaler, 'features': FEATURE_COLS}, f)

    metrics = {'test_mae': round(mae,4), 'test_r2': round(r2,4),
               'il_count': len(il_skorlar), 'feature_count': len(FEATURE_COLS)}
    with open(f"{model_dir}/turkey_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n[✓] Türkiye modeli tamamlandı.\n")
    return il_skorlar, metrics


if __name__ == "__main__":
    os.chdir(os.path.join(os.path.dirname(__file__), ".."))
    run_turkey_pipeline()
