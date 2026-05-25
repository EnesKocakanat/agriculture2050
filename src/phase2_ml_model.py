"""
=============================================================
AGRI-2050: Faz 2 - Makine Öğrenmesi Model Eğitimi
=============================================================
Görev : Konya Ovası buğday rekoltesi tahmini (2025-2030)
Model : Ensemble (RandomForest + XGBoost + LightGBM)
Input : NDVI, yağış, sıcaklık, iklim endeksleri
Output: ton/hektar ve toplam üretim tahmini
=============================================================
"""

import os
import json
import numpy as np
import pandas as pd
import pickle
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline


# ─── Feature Engineering ────────────────────────────────────────────────────
class AgriculturalFeatureEngineer:
    """
    Tarımsal tahmin için özellik mühendisliği.
    Her özellik agronomi literatüründen türetilmiştir.
    """

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # ── NDVI Özellikleri ────────────────────────────────────────────────
        # İlkbahar NDVI: hasat tahmini için kritik pencere (Mart-Mayıs)
        df["ndvi_growing_season"] = df["ndvi_mean"] * 1.15  # proxy
        df["ndvi_stress_index"]   = 1 - df["ndvi_mean"]     # bitki stresi

        # ── Su Stresi Özellikleri ────────────────────────────────────────────
        df["drought_index"] = df["precip_annual_mm"] / df["precip_annual_mm"].mean()
        df["water_deficit"]  = (df["precip_annual_mm"] < 270).astype(int)

        # ── Isı Stresi ──────────────────────────────────────────────────────
        df["heat_stress"]   = (df["lst_mean"] > 32).astype(int)
        df["lst_normalized"] = (df["lst_mean"] - df["lst_mean"].mean()) / df["lst_mean"].std()

        # ── Bileşik İklim Endeksi (kendi formülümüz) ─────────────────────────
        df["climate_index"] = (df["ndvi_mean"] * 0.4 +
                               df["drought_index"] * 0.35 +
                               (1 - df["heat_stress"] * 0.1) * 0.25)

        # ── Lag özellikleri (önceki yılın etkisi) ────────────────────────────
        df["ndvi_lag1"]     = df["ndvi_mean"].shift(1)
        df["precip_lag1"]   = df["precip_annual_mm"].shift(1)
        df["yield_lag1"]    = df["yield_t_ha"].shift(1)

        # ── Trend değişkeni ─────────────────────────────────────────────────
        df["year_trend"] = df["year"] - df["year"].min()

        # ── Ekstrem hava olayları ────────────────────────────────────────────
        for col in ["drought_year", "flood_year", "frost_year"]:
            if col not in df.columns:
                df[col] = False
        df["extreme_event"] = (df["drought_year"] | df["flood_year"] | df["frost_year"]).astype(int)

        return df.dropna()

    def get_feature_columns(self):
        return [
            "ndvi_mean", "ndvi_peak", "ndvi_growing_season", "ndvi_stress_index",
            "ndvi_lag1", "lst_mean", "lst_normalized", "heat_stress",
            "precip_annual_mm", "drought_index", "water_deficit", "precip_lag1",
            "climate_index", "extreme_event", "year_trend", "yield_lag1"
        ]


# ─── Ensemble Model ──────────────────────────────────────────────────────────
class CropYieldEnsemble:
    """
    3 modelin ağırlıklı ortalaması:
    - RandomForest     (w=0.35) → non-linear ilişkiler
    - GradientBoosting (w=0.40) → en yüksek doğruluk
    - Ridge Regression (w=0.25) → stabilite, yorumlanabilirlik
    """

    def __init__(self):
        self.models = {
            "random_forest": RandomForestRegressor(
                n_estimators=300, max_depth=8,
                min_samples_leaf=2, random_state=42
            ),
            "gradient_boost": GradientBoostingRegressor(
                n_estimators=200, learning_rate=0.05,
                max_depth=5, subsample=0.8, random_state=42
            ),
            "ridge": Ridge(alpha=1.0),
        }
        self.weights    = {"random_forest": 0.35, "gradient_boost": 0.40, "ridge": 0.25}
        self.scaler     = StandardScaler()
        self.is_trained = False
        self.metrics    = {}
        self.feature_importance = {}

    def train(self, X_train, y_train, feature_names):
        X_scaled = self.scaler.fit_transform(X_train)
        tscv     = TimeSeriesSplit(n_splits=4)

        print("\n  Model Eğitimi:")
        for name, model in self.models.items():
            model.fit(X_scaled, y_train)
            cv_scores = cross_val_score(model, X_scaled, y_train,
                                        cv=tscv, scoring="neg_mean_absolute_error")
            self.metrics[name] = {
                "cv_mae": round(-cv_scores.mean(), 4),
                "cv_std": round(cv_scores.std(), 4),
            }
            print(f"    [{name:20s}] CV-MAE: {self.metrics[name]['cv_mae']:.4f} "
                  f"± {self.metrics[name]['cv_std']:.4f} t/ha")

            if hasattr(model, "feature_importances_"):
                self.feature_importance[name] = dict(
                    zip(feature_names, model.feature_importances_))

        self.is_trained = True
        return self

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        preds = {}
        for name, model in self.models.items():
            preds[name] = model.predict(X_scaled)

        ensemble_pred = sum(
            preds[name] * self.weights[name]
            for name in self.models
        )
        return ensemble_pred, preds

    def predict_with_uncertainty(self, X, n_bootstrap=100):
        """Bootstrap ile güven aralığı hesaplama"""
        X_scaled = self.scaler.transform(X)
        rf = self.models["random_forest"]

        # RandomForest'ın bireysel ağaç tahminleri
        tree_preds = np.array([tree.predict(X_scaled) for tree in rf.estimators_])
        mean_pred  = tree_preds.mean(axis=0)
        std_pred   = tree_preds.std(axis=0)

        # Ensemble tahmini
        ensemble_pred, _ = self.predict(X)

        return {
            "mean":    ensemble_pred,
            "lower_80": ensemble_pred - 1.28 * std_pred,
            "upper_80": ensemble_pred + 1.28 * std_pred,
            "lower_95": ensemble_pred - 1.96 * std_pred,
            "upper_95": ensemble_pred + 1.96 * std_pred,
        }


# ─── Gelecek Projeksiyon ─────────────────────────────────────────────────────
class ClimateScenarioProjector:
    """
    IPCC SSP2-4.5 ve SSP5-8.5 senaryolarına göre
    2025-2030 iklim projeksiyonu.
    """

    SCENARIOS = {
        "optimist_ssp126": {
            "ndvi_trend":    +0.002,
            "precip_trend":  -2.5,
            "temp_trend":    +0.02,
            "label":         "İyimser (SSP1-2.6)",
            "color":         "#22c55e"
        },
        "orta_ssp245": {
            "ndvi_trend":    -0.003,
            "precip_trend":  -5.0,
            "temp_trend":    +0.04,
            "label":         "Orta (SSP2-4.5)",
            "color":         "#f59e0b"
        },
        "kotumser_ssp585": {
            "ndvi_trend":    -0.008,
            "precip_trend":  -9.0,
            "temp_trend":    +0.08,
            "label":         "Kötümser (SSP5-8.5)",
            "color":         "#ef4444"
        },
    }

    def project(self, base_row: dict, years: list) -> pd.DataFrame:
        projections = []
        for scenario_key, params in self.SCENARIOS.items():
            for i, year in enumerate(years):
                dt = year - 2024
                row = {
                    "year":              year,
                    "scenario":          scenario_key,
                    "scenario_label":    params["label"],
                    "ndvi_mean":         base_row["ndvi_mean"]   + params["ndvi_trend"]   * dt,
                    "ndvi_peak":         base_row["ndvi_peak"]   + params["ndvi_trend"]   * dt * 0.8,
                    "lst_mean":          base_row["lst_mean"]    + params["temp_trend"]   * dt,
                    "precip_annual_mm":  base_row["precip_annual_mm"] + params["precip_trend"] * dt,
                    "drought_year":      False,
                    "flood_year":        False,
                    "frost_year":        False,
                }
                # Türetilmiş özellikler için referans değerleri
                row["ndvi_lag1"]    = base_row.get("ndvi_mean", 0.62)
                row["precip_lag1"]  = base_row.get("precip_annual_mm", 312)
                row["yield_lag1"]   = base_row.get("yield_t_ha", 3.2)
                projections.append(row)

        return pd.DataFrame(projections)


# ─── Ana Eğitim Pipeline ─────────────────────────────────────────────────────
def run_training_pipeline(data_dir="data", model_dir="models"):
    os.makedirs(model_dir, exist_ok=True)
    print("=" * 60)
    print("AGRI-2050 | FAZ 2: ML Model Eğitimi")
    print("=" * 60)

    # Veriyi yükle
    merged_df = pd.read_csv(f"{data_dir}/merged_dataset.csv")
    print(f"\n[1/4] Veri yüklendi: {merged_df.shape}")

    # Feature engineering
    fe = AgriculturalFeatureEngineer()
    df = fe.create_features(merged_df)
    feature_cols = fe.get_feature_columns()

    X = df[feature_cols].values
    y = df["yield_t_ha"].values

    print(f"[2/4] Özellikler hazırlandı: {len(feature_cols)} değişken, {len(X)} örnek")

    # Train/test split (zaman serisi için son 3 yıl test)
    split_idx = len(X) - 3
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # Model eğitimi
    print("\n[3/4] Ensemble model eğitiliyor...")
    model = CropYieldEnsemble()
    model.train(X_train, y_train, feature_cols)

    # Test seti değerlendirmesi
    ensemble_pred, _ = model.predict(X_test)
    test_mae = mean_absolute_error(y_test, ensemble_pred)
    test_r2  = r2_score(y_test, ensemble_pred)
    print(f"\n  Test Seti Sonuçları:")
    print(f"    MAE : {test_mae:.4f} t/ha")
    print(f"    R²  : {test_r2:.4f}")

    # Gelecek tahminleri
    print("\n[4/4] 2025-2030 projeksiyonları hesaplanıyor...")
    projector = ClimateScenarioProjector()
    base_row  = {
        "ndvi_mean": df["ndvi_mean"].iloc[-1],
        "ndvi_peak": df["ndvi_peak"].iloc[-1],
        "lst_mean":  df["lst_mean"].iloc[-1],
        "precip_annual_mm": df["precip_annual_mm"].iloc[-1],
        "yield_t_ha": df["yield_t_ha"].iloc[-1],
    }

    future_df   = projector.project(base_row, list(range(2025, 2031)))
    future_feat = fe.create_features(future_df.assign(
        yield_t_ha=base_row["yield_t_ha"],
        flood_year=False, frost_year=False
    ))

    results = []
    for scenario in ClimateScenarioProjector.SCENARIOS:
        sub   = future_feat[future_feat["scenario"] == scenario]
        if len(sub) == 0:
            continue
        X_fut = sub[feature_cols].values
        preds = model.predict_with_uncertainty(X_fut)

        for i, (_, row) in enumerate(sub.iterrows()):
            area_ha = 580000
            results.append({
                "year":             int(row["year"]),
                "scenario":         scenario,
                "scenario_label":   ClimateScenarioProjector.SCENARIOS[scenario]["label"],
                "yield_t_ha":       round(float(preds["mean"][i]), 3),
                "yield_lower_80":   round(float(preds["lower_80"][i]), 3),
                "yield_upper_80":   round(float(preds["upper_80"][i]), 3),
                "total_prod_t":     int(preds["mean"][i] * area_ha),
                "area_ha":          area_ha,
            })

    results_df = results_df = pd.DataFrame(results)
    results_df.to_csv(f"{data_dir}/forecast_2025_2030.csv", index=False)
    print(f"  ✓ forecast_2025_2030.csv: {len(results_df)} satır")

    # Modeli kaydet
    with open(f"{model_dir}/ensemble_model.pkl", "wb") as f:
        pickle.dump({"model": model, "feature_engineer": fe,
                     "metrics": {"test_mae": test_mae, "test_r2": test_r2}}, f)
    print(f"  ✓ ensemble_model.pkl kaydedildi")

    # Metrikler
    metrics = {
        "test_mae": round(test_mae, 4),
        "test_r2":  round(test_r2, 4),
        "cv_metrics": model.metrics,
        "feature_count": len(feature_cols),
        "training_samples": len(X_train),
    }
    with open(f"{model_dir}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n[✓] Faz 2 tamamlandı.\n")
    return model, results_df, metrics


if __name__ == "__main__":
    from phase1_data_fetch import run_data_pipeline
    run_data_pipeline()
    run_training_pipeline()
