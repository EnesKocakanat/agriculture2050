"""
AGRI-2050: Faz 2c — Gerçek Rekolte + DSİ Sulama Verisiyle ML Model
===================================================================
TÜİK ton/ha + GEE NDVI + Open-Meteo yağış + DSİ sulama verisi.
Çalıştırma: python src/phase2c_ml_real_yield.py
===================================================================
"""

import os, json, pickle, warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, LeaveOneGroupOut
from sklearn.metrics import mean_absolute_error, r2_score
warnings.filterwarnings("ignore")

FEATURE_COLS = [
    'ndvi_mean', 'ndvi_peak', 'ndvi_stress', 'ndvi_lag1',
    'precip_mm', 'drought_index', 'water_deficit', 'precip_lag1',
    'heat_proxy', 'year_trend',
    'sulanan_ha', 'sulama_orani', 'sulama_yogunluk',
]


def load_and_merge(raw_dir, processed_dir):
    print("[1/4] Veriler yükleniyor...")

    # Gerçek rekolte verisi
    rekolt = pd.read_csv(f"{processed_dir}/tuik_rekolt_clean.csv")
    rekolt = rekolt[['il', 'year', 'verim_t_ha', 'uretim_ton', 'ekilen_ha']].dropna(subset=['verim_t_ha'])

    # Yağış
    precip = pd.read_csv(f"{raw_dir}/precipitation_iller.csv")
    drought_col = 'drought_year_x' if 'drought_year_x' in precip.columns else 'drought_year'
    precip_y = precip.groupby(['il', 'year']).agg(
        precip_mm=('precip_mm', 'sum'),
        drought_year=(drought_col, 'first')
    ).reset_index()

    # NDVI
    ndvi = pd.read_csv(f"{raw_dir}/satellite_ndvi_iller.csv")
    ndvi_y = ndvi.groupby(['il', 'year']).agg(
        ndvi_mean=('ndvi', 'mean'),
        ndvi_peak=('ndvi', 'max')
    ).reset_index()

    # DSİ sulama verisi
    dsi_path = f"{processed_dir}/dsi_sulama_clean.csv"
    if os.path.exists(dsi_path):
        dsi = pd.read_csv(dsi_path)[['il', 'year', 'sulanan_ha', 'sulama_orani']]
        print(f"    ✓ DSİ sulama verisi yüklendi ({dsi['il'].nunique()} il, {dsi['year'].min()}-{dsi['year'].max()})")
    else:
        print("    ⚠ DSİ verisi bulunamadı, sıfır olarak doldurulacak")
        dsi = None

    # Birleştir
    df = rekolt.merge(precip_y, on=['il', 'year'], how='inner')
    df = df.merge(ndvi_y, on=['il', 'year'], how='inner')

    if dsi is not None:
        df = df.merge(dsi, on=['il', 'year'], how='left')
    else:
        df['sulanan_ha'] = 0.0
        df['sulama_orani'] = 0.0

    # DSİ sadece 2019-2024 — önceki yıllar için il ortalamasıyla doldur
    df['sulanan_ha']   = df.groupby('il')['sulanan_ha'].transform(lambda x: x.fillna(x.mean()))
    df['sulama_orani'] = df.groupby('il')['sulama_orani'].transform(lambda x: x.fillna(x.mean()))
    df['sulanan_ha']   = df['sulanan_ha'].fillna(0)
    df['sulama_orani'] = df['sulama_orani'].fillna(0)

    print(f"    ✓ {df['il'].nunique()} il | {len(df)} kayıt | {df['year'].min()}-{df['year'].max()}")
    return df


def feature_engineering(df):
    df = df.copy()
    df['drought_index']    = df['precip_mm'] / df.groupby('il')['precip_mm'].transform('mean')
    df['ndvi_stress']      = 1 - df['ndvi_mean']
    df['water_deficit']    = (df['precip_mm'] < 270).astype(int)
    df['heat_proxy']       = df['ndvi_mean'] * df['drought_index']
    df['year_trend']       = df['year'] - df['year'].min()
    df['ndvi_lag1']        = df.groupby('il')['ndvi_mean'].shift(1)
    df['precip_lag1']      = df.groupby('il')['precip_mm'].shift(1)
    df['drought_year']     = df['drought_year'].fillna(False).astype(int)
    # Sulama yoğunluğu: sulanan ha / ekilen ha
    df['sulama_yogunluk']  = (df['sulanan_ha'] / df['ekilen_ha'].replace(0, np.nan)).fillna(0)
    return df.dropna(subset=FEATURE_COLS + ['verim_t_ha'])


def train_ensemble(X, y, groups):
    models = {
        'rf':    RandomForestRegressor(n_estimators=300, max_depth=6, min_samples_leaf=3, random_state=42),
        'gbm':   GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42),
        'ridge': Ridge(alpha=10.0),
    }
    weights = {'rf': 0.35, 'gbm': 0.40, 'ridge': 0.25}
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)

    logo = LeaveOneGroupOut()
    print("\n  Model Eğitimi (Leave-One-Year-Out CV):")
    for name, m in models.items():
        m.fit(X_sc, y)
        cv = cross_val_score(m, X_sc, y, cv=logo, groups=groups,
                             scoring='neg_mean_absolute_error')
        print(f"    [{name:6s}] CV-MAE: {-cv.mean():.4f} ± {cv.std():.4f} t/ha")

    return models, weights, scaler


def predict_ensemble(models, weights, scaler, X):
    X_sc = scaler.transform(X)
    return sum(m.predict(X_sc) * weights[n] for n, m in models.items())


def feature_importance(models, feature_cols):
    rf_imp  = models['rf'].feature_importances_
    gbm_imp = models['gbm'].feature_importances_
    imp = (rf_imp * 0.35 + gbm_imp * 0.40) / 0.75
    df_imp = pd.DataFrame({'ozellik': feature_cols, 'onem': imp})
    return df_imp.sort_values('onem', ascending=False)


def run_real_yield_pipeline(raw_dir="data/raw", processed_dir="data/processed", model_dir="models"):
    os.makedirs(model_dir, exist_ok=True)
    print("=" * 60)
    print("AGRI-2050 | Gerçek Rekolte + DSİ ML Modeli")
    print("=" * 60 + "\n")

    df = load_and_merge(raw_dir, processed_dir)

    print("\n[2/4] Feature engineering...")
    df = feature_engineering(df)
    print(f"    ✓ {len(FEATURE_COLS)} özellik | {len(df)} örnek | {df['il'].nunique()} il")

    X = df[FEATURE_COLS].values
    y = df['verim_t_ha'].values
    groups = df['year'].values

    print("\n[3/4] Ensemble model eğitiliyor...")
    models, weights, scaler = train_ensemble(X, y, groups)

    # Test (son yıl = 2024)
    mask_test  = df['year'] == df['year'].max()
    mask_train = ~mask_test
    models_final, _, scaler_final = train_ensemble(
        X[mask_train], y[mask_train], groups[mask_train]
    )
    y_pred = predict_ensemble(models_final, weights, scaler_final, X[mask_test])
    mae = mean_absolute_error(y[mask_test], y_pred)
    r2  = r2_score(y[mask_test], y_pred)
    print(f"\n  Test (2024): MAE={mae:.4f} t/ha | R²={r2:.4f}")

    # Özellik önem sırası
    print("\n  Özellik Önem Sırası:")
    df_imp = feature_importance(models, FEATURE_COLS)
    for _, r in df_imp.head(8).iterrows():
        bar = '█' * int(r['onem'] * 100)
        print(f"    {r['ozellik']:<22} {r['onem']:.3f}  {bar}")

    # 2024 tahmin vs gerçek
    df_test = df[mask_test].copy()
    df_test['verim_tahmin'] = y_pred
    df_test['hata_pct'] = ((df_test['verim_tahmin'] - df_test['verim_t_ha'])
                           / df_test['verim_t_ha'] * 100).abs()

    print(f"\n  2024 En İyi Tahminler:")
    print(f"  {'İl':<20} {'Gerçek':>8} {'Tahmin':>8} {'Hata%':>7} {'Sulama':>10}")
    print("  " + "-" * 57)
    for _, r in df_test.nsmallest(10, 'hata_pct').iterrows():
        print(f"  {r['il']:<20} {r['verim_t_ha']:>7.3f}  {r['verim_tahmin']:>7.3f}"
              f"  {r['hata_pct']:>6.1f}%  {r['sulama_orani']:>8.2f}")

    # İl skorlarını kaydet
    print("\n[4/4] İl bazlı tahminler kaydediliyor...")
    df['verim_tahmin'] = predict_ensemble(models, weights, scaler, X)
    il_ozet = df.groupby('il').agg(
        verim_gercek_ort=('verim_t_ha',    'mean'),
        verim_tahmin_ort=('verim_tahmin',  'mean'),
        uretim_ton_ort  =('uretim_ton',    'mean'),
        precip_ort      =('precip_mm',     'mean'),
        ndvi_ort        =('ndvi_mean',     'mean'),
        sulanan_ha_ort  =('sulanan_ha',    'mean'),
        sulama_orani_ort=('sulama_orani',  'mean'),
    ).reset_index().round(4)

    out_path = f"{processed_dir}/turkey_real_yield_scores.csv"
    il_ozet.to_csv(out_path, index=False)
    print(f"    ✓ {len(il_ozet)} il → {out_path}")

    # Model kaydet
    with open(f"{model_dir}/real_yield_model.pkl", "wb") as f:
        pickle.dump({'models': models, 'weights': weights,
                     'scaler': scaler, 'features': FEATURE_COLS}, f)

    metrics = {
        'test_mae_t_ha': round(mae, 4),
        'test_r2':       round(r2, 4),
        'il_count':      int(df['il'].nunique()),
        'sample_count':  len(df),
        'feature_count': len(FEATURE_COLS),
        'target':        'verim_t_ha (gerçek TÜİK + DSİ verisi)',
    }
    with open(f"{model_dir}/real_yield_metrics.json", "w", encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"\n  En verimli iller (ortalama gerçek verim):")
    for _, r in il_ozet.nlargest(5, 'verim_gercek_ort').iterrows():
        print(f"    {r['il']:<20} {r['verim_gercek_ort']:.3f} t/ha  "
              f"sulama: {r['sulama_orani_ort']:.2f}")

    print(f"\n  En düşük verimli iller:")
    for _, r in il_ozet.nsmallest(5, 'verim_gercek_ort').iterrows():
        print(f"    {r['il']:<20} {r['verim_gercek_ort']:.3f} t/ha  "
              f"sulama: {r['sulama_orani_ort']:.2f}")

    print(f"\n[✓] Gerçek rekolte + DSİ modeli tamamlandı.\n")
    return il_ozet, metrics


if __name__ == "__main__":
    os.chdir(os.path.join(os.path.dirname(__file__), ".."))
    run_real_yield_pipeline()
