
# -*- coding: utf-8 -*-
"""
03_stage2_conditional_severity.py
=====================================
Severity Model — GLM BUG FIX + LEAKAGE-FREE VALIDATION

FIXES:
  [BUG-1] GLM prediction error: The const column added by
          sm.add_constant() was being lost during reindexing.
          Solution: The feature column list is stored without
          the const column, all matrices are aligned consistently,
          and the constant is added only once after alignment.

  [BUG-2] Early-stopping leakage: Validation year = 2016;
          test year = 2017 (held out and isolated).
"""

import os
import pandas as pd
import numpy as np
import lightgbm as lgb
import shap
import statsmodels.api as sm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import roc_auc_score, classification_report, average_precision_score

FILE_PATH  = r"C:\Users\m-a-o\Desktop\TarsimData\V2\model_ready_data_spatial_final_v1.csv"
OUTPUT_DIR = r"C:\Users\m-a-o\Desktop\TarsimData\V2\EDA_Plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# [1] DATA PREPARATION
# =============================================================================
print("--- [1] Loading data ---")
df = pd.read_csv(FILE_PATH, low_memory=False)

df_sev = df[df["Claim_Is"] == 1].copy()
df_sev["Loss_Ratio"] = df_sev["Claim"] / df_sev["Exposure"]
df_sev = df_sev[df_sev["Loss_Ratio"] > 0].copy()

df_sev["Urun_Clean"] = df_sev["Urun Adi Genel"].astype(str).str.upper().str.strip()
df_sev["Urun_Clean"] = df_sev["Urun_Clean"].replace("BUGDAY", "BUĞDAY")
df_sev = df_sev[df_sev["Urun_Clean"].isin(["PAMUK", "MISIR", "BUĞDAY", "ARPA"])].copy()
df_sev["Urun_Kategori"] = pd.Categorical(
    df_sev["Urun_Clean"],
    categories=["ARPA", "BUĞDAY", "MISIR", "PAMUK"], ordered=False
)
df_sev["Is_Catastrophic"] = (df_sev["Loss_Ratio"] > 0.50).astype(int)

FEATURES_LGB = [
    "WMAXSHR", "CAPE_dur", "WMAXSHR_burst", "h_agl",
    "LATITUDE", "LONGITUDE", "RAKIM", "Urun_Kategori"
]
GLM_COLS = ["WMAXSHR", "CAPE_dur", "WMAXSHR_burst", "h_agl", "Urun_Kategori"]

df_sev = df_sev.dropna(
    subset=FEATURES_LGB + ["Loss_Ratio", "Is_Catastrophic", "yil", "Claim", "Exposure"]
).copy()

# =============================================================================
# [2] DATA SPLIT: Train 2008-2015 | Validation 2016 | Test 2017
# =============================================================================
print("\n--- [2] Data split: Train≤2015 | Validation=2016 | Test=2017 ---")
train_mask = df_sev["yil"] <= 2015
val_mask   = df_sev["yil"] == 2016
test_mask  = df_sev["yil"] == 2017

X_train_lp = df_sev.loc[train_mask, FEATURES_LGB].copy()
X_val_lp   = df_sev.loc[val_mask,   FEATURES_LGB].copy()
X_test_lp  = df_sev.loc[test_mask,  FEATURES_LGB].copy()

y_train_reg = np.log(df_sev.loc[train_mask, "Loss_Ratio"].values)
y_val_reg   = np.log(df_sev.loc[val_mask,   "Loss_Ratio"].values)
y_test_reg  = np.log(df_sev.loc[test_mask,  "Loss_Ratio"].values)

y_train_clf = df_sev.loc[train_mask, "Is_Catastrophic"].values
y_val_clf   = df_sev.loc[val_mask,   "Is_Catastrophic"].values
y_test_clf  = df_sev.loc[test_mask,  "Is_Catastrophic"].values

print(f"    Train: {len(X_train_lp):,} | Validation: {len(X_val_lp):,} | Test: {len(X_test_lp):,}")
print(f"    Catastrophic cases — Train: {y_train_clf.sum()} | Validation: {y_val_clf.sum()} | Test: {y_test_clf.sum()}")

# =============================================================================
# [3] MODEL A: GAMMA GLM — BUG FIX APPLIED
# =============================================================================
print("\n" + "="*60)
print("--- MODEL A: GAMMA GLM (GLM bug fixed) ---")
print("="*60)

def make_glm_matrix(df_subset, glm_cols, ref_cols=None):
    """
    Apply one-hot encoding, align columns with the reference feature set,
    and add a constant term.

    ref_cols: Feature column list from the training matrix (excluding const).

    Returns:
        (X_with_const, dummy_cols_no_const)
    """
    X = pd.get_dummies(df_subset[glm_cols], drop_first=True).astype(float)
    if ref_cols is not None:
        X = X.reindex(columns=ref_cols, fill_value=0.0)
    X_const = sm.add_constant(X, has_constant="add")
    return X_const, list(X.columns)

# Training matrix: store the feature columns as the reference (excluding const)
X_train_glm, glm_feature_cols = make_glm_matrix(
    df_sev.loc[train_mask], GLM_COLS, ref_cols=None
)
# Align validation and test matrices with the training columns, then add const
X_val_glm,  _ = make_glm_matrix(df_sev.loc[val_mask],  GLM_COLS, ref_cols=glm_feature_cols)
X_test_glm, _ = make_glm_matrix(df_sev.loc[test_mask], GLM_COLS, ref_cols=glm_feature_cols)

print(f"    GLM matrix shape — Train: {X_train_glm.shape}, Test: {X_test_glm.shape}")
assert X_train_glm.shape[1] == X_test_glm.shape[1], "Column counts do not match!"

offset_train = np.log(df_sev.loc[train_mask, "Exposure"].values.astype(float))
offset_val   = np.log(df_sev.loc[val_mask,   "Exposure"].values.astype(float))
offset_test  = np.log(df_sev.loc[test_mask,  "Exposure"].values.astype(float))

gamma_model = sm.GLM(
    df_sev.loc[train_mask, "Claim"].astype(float).values,
    X_train_glm,
    family=sm.families.Gamma(link=sm.families.links.Log()),
    offset=offset_train
)
gamma_results = gamma_model.fit()
print(gamma_results.summary())

# Generate test-set predictions
gamma_pred_claim  = gamma_results.predict(X_test_glm, offset=offset_test)
gamma_pred_lr     = gamma_pred_claim / df_sev.loc[test_mask, "Exposure"].values
gamma_pred_log_lr = np.log(np.clip(gamma_pred_lr, 1e-6, None))

gamma_rmse = np.sqrt(mean_squared_error(y_test_reg, gamma_pred_log_lr))
gamma_mae  = mean_absolute_error(y_test_reg, gamma_pred_log_lr)
print(f"\n    Gamma GLM — Test RMSE: {gamma_rmse:.4f} | MAE: {gamma_mae:.4f}")

# =============================================================================
# [4] MODEL B: OLS BASELINE
# =============================================================================
ols_model       = sm.OLS(y_train_reg, X_train_glm).fit()
ols_pred        = ols_model.predict(X_test_glm)
ols_rmse        = np.sqrt(mean_squared_error(y_test_reg, ols_pred))
ols_mae         = mean_absolute_error(y_test_reg, ols_pred)
ols_r2          = r2_score(y_test_reg, ols_pred)

# =============================================================================
# [5] MODEL C: LIGHTGBM REGRESSOR — Validation year = 2016 (leakage-free)
# =============================================================================
print("\n--- MODEL C: LightGBM Regressor (validation year = 2016) ---")
lgb_reg = lgb.LGBMRegressor(
    n_estimators=3000, learning_rate=0.01, max_depth=7, num_leaves=63,
    subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1
)
lgb_reg.fit(
    X_train_lp, y_train_reg,
    eval_set=[(X_val_lp, y_val_reg)],
    eval_metric="rmse",
    categorical_feature=["Urun_Kategori"],
    callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(-1)]
)
print(f"    Best iteration (validation): {lgb_reg.best_iteration_}")

lgb_pred = lgb_reg.predict(X_test_lp)
lgb_rmse = np.sqrt(mean_squared_error(y_test_reg, lgb_pred))
lgb_mae  = mean_absolute_error(y_test_reg, lgb_pred)
lgb_r2   = r2_score(y_test_reg, lgb_pred)

# =============================================================================
# [6] TABLE 4
# =============================================================================
print("\n" + "="*65)
print("TABLE 4 — MODEL COMPARISON (Log Scale | Test: 2017)")
print("="*65)
print(f"{'Model':<45} {'RMSE':>8} {'MAE':>8} {'R2':>8}")
print("-"*65)
print(f"{'Gamma GLM (log-link, offset)':<45} {gamma_rmse:>8.4f} {gamma_mae:>8.4f} {'N/A':>8}")
print(f"{'OLS (log-target)':<45} {ols_rmse:>8.4f} {ols_mae:>8.4f} {ols_r2:>8.4f}")
print(f"{'LightGBM + spatial (validation=2016)':<45} {lgb_rmse:>8.4f} {lgb_mae:>8.4f} {lgb_r2:>8.4f}")
print("="*65)

# =============================================================================
# [7] MODEL D: CATASTROPHIC DAMAGE CLASSIFICATION — Validation year = 2016
# =============================================================================
print("\n--- MODEL D: Catastrophic Damage Classification (validation year = 2016) ---")
pos_weight = (len(y_train_clf) - y_train_clf.sum()) / max(1, y_train_clf.sum())
print(f"    scale_pos_weight = {pos_weight:.2f}")

lgb_clf = lgb.LGBMClassifier(
    n_estimators=3000, learning_rate=0.01, max_depth=6, num_leaves=31,
    subsample=0.8, colsample_bytree=0.8,
    scale_pos_weight=pos_weight, random_state=42, n_jobs=-1
)
lgb_clf.fit(
    X_train_lp, y_train_clf,
    eval_set=[(X_val_lp, y_val_clf)],
    eval_metric="auc",
    categorical_feature=["Urun_Kategori"],
    callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(-1)]
)
print(f"    Best iteration (validation): {lgb_clf.best_iteration_}")

y_pred_prob = lgb_clf.predict_proba(X_test_lp)[:, 1]
roc_auc_clf = roc_auc_score(y_test_clf, y_pred_prob)
pr_auc_clf  = average_precision_score(y_test_clf, y_pred_prob)

print(f"\n    Catastrophic damage model — ROC-AUC: {roc_auc_clf:.4f} | PR-AUC: {pr_auc_clf:.4f}")
print("\n" + classification_report(y_test_clf, (y_pred_prob >= 0.5).astype(int), digits=4))

# =============================================================================
# [8] SHAP ANALYSIS
# =============================================================================
print("\n--- SHAP analysis (Catastrophic Damage Model) ---")
explainer = shap.TreeExplainer(lgb_clf)
shap_vals = explainer.shap_values(X_test_lp)
shap_pos  = shap_vals[1] if isinstance(shap_vals, list) else shap_vals

fig, ax = plt.subplots(figsize=(10, 6))
shap.summary_plot(shap_pos, X_test_lp, show=False)
ax.set_title("Catastrophic Damage (>50% Loss) — SHAP\n"
             "Train: 2008-2015 | Validation: 2016 | Test: 2017", fontsize=12)
plt.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "SHAP_Catastrophic_v2.png"), dpi=300, bbox_inches="tight")
plt.close()

print(f"\n✅ Stage 2 v2 completed. Outputs saved to: {OUTPUT_DIR}")
