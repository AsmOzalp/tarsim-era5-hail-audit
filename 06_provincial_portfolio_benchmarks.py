# -*- coding: utf-8 -*-
"""
06_provincial_portfolio_benchmarks.py
=========================
Asama 2 - Il ve Portfoy Duzeyinde Baz Modeller (Baselines) ve Bootstrap CI
(Hakem 1 Major 1, Hakem 2 Yorum 10-11, Hakem 3 Major 3):

Karsilastirilan Modeller:
  - Baseline 0 (National Climatology): Ulusal tarihsel ortalama Loss Cost (c_bar_nat * Exposure)
    ve ulusal ortalama etkilenen police orani (pi_bar_nat * PS).
  - Baseline 1 (Provincial Empirical Bayes / Credibility): Her ilin kendi gecmis
    yillardaki (Train) ampirik Loss Cost ve DHPS/PS orani (Buhlmann-Straub kredibilite
    agirlikli).
  - Baseline 2 (Exposure-Only Hurdle): Sadece log_coverage ve urun/rakim kontrolleri
    (ERA5 degiskenleri YOK).
  - ERA5 Model 1 (Pure ERA5 Binomial x Exposure-Weighted Severity):
    Asama 1 Binomial (ERA5-only) * Asama 2 Maruziyet Agirlikli Hasar Siddeti.
  - ERA5 Model 2 (Two-Stage Hurdle: Combined Occurrence x Severity):
    Asama 1 Logit (ERA5 + log_coverage) * Asama 2 Hasar Siddeti (Val-kalibreli).
  - ERA5 Model 3 (Hybrid Provincial Credibility + ERA5 Anomaly):
    Ilin tarihsel baz riskini o yilin ERA5 konvektif anomalisiyle guncelleyen
    aktüeryal tarama modeli.

Degerlendirme Olcekleri (Hem 2017 Test, Hem 2015-2017 Rolling Window):
  1. Il Bazinda Loss Cost Orani (%) = (Il Toplam Hasar / Il Toplam Exposure) * 100
     (Trivial il buyuklugu etkisinden arindirilmis gercek risk orani!)
  2. Il Bazinda Etkilenen Police Orani (%) = (Il Toplam DHPS / Il Toplam PS) * 100
  3. Il Bazinda Toplam Finansal Hasar (Milyon TL) ve Yillik Portfoy Toplami.
  4. Paired Provincial Bootstrap (B=1000, iller yeniden orneklenerek %95 CI ve p-degerleri).
"""

from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
import lightgbm as lgb
from scipy.stats import spearmanr, pearsonr
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

warnings.filterwarnings("ignore")

# =============================================================================
# [1] CONFIG & VERI YUKLEME
# =============================================================================
FILE_PATH  = Path(r"C:\Users\m-a-o\Desktop\TarsimData\V2\model_ready_data_spatial_final_v1.csv")
OUTPUT_DIR = FILE_PATH.parent / "revision_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MET_FEATURES  = ["WMAXSHR", "CAPE_dur", "WMAXSHR_burst", "h_agl"]
FULL_FEATURES = MET_FEATURES + ["log_coverage"]
PROV_COL      = "Il Adi"
YEAR_COL      = "yil"
N_BOOT        = 1000
SEED          = 42

print("=== [1] VERI YUKLENIYOR ===")
df = pd.read_csv(FILE_PATH, low_memory=False)
needed = FULL_FEATURES + ["Claim_Is", "Claim", "Exposure", "DHPS", "PS", PROV_COL, YEAR_COL, "RAKIM"]
df = df.dropna(subset=needed).copy().reset_index(drop=True)
df = df[(df["Exposure"] > 0) & (df["PS"] >= 1)].copy().reset_index(drop=True)
df["DHPS"] = np.minimum(df["DHPS"], df["PS"])
df["loss_cost_row"] = df["Claim"] / df["Exposure"]
df["unit_rate_row"] = df["DHPS"] / df["PS"]

# =============================================================================
# [2] TEK YIL ICIN BAZ MODELLER VE ERA5 PORTFOY MODELLERINI KURAN FONKSIYON
# =============================================================================
def run_portfolio_year(df_all, test_year, calibrate_on_val=True):
    """
    Verilen test_year (orn. 2017) icin:
      - Train: yil < test_year - 1 (orn. <=2015)
      - Val  : yil == test_year - 1 (orn. 2016) -> olcek kalibrasyonu icin
      - History (Train+Val): yil < test_year (orn. <=2016) -> tarihsel baz oranlar
    """
    val_year = test_year - 1
    df_tr  = df_all[df_all[YEAR_COL] < val_year].copy().reset_index(drop=True)
    df_va  = df_all[df_all[YEAR_COL] == val_year].copy().reset_index(drop=True)
    df_his = df_all[df_all[YEAR_COL] < test_year].copy().reset_index(drop=True)
    df_te  = df_all[df_all[YEAR_COL] == test_year].copy().reset_index(drop=True)

    # --- 1. ULUSAL TARIHSEL BAZ ORANLAR (Baseline 0) ---
    nat_lc_his   = float(df_his["Claim"].sum() / df_his["Exposure"].sum())
    nat_unit_his = float(df_his["DHPS"].sum() / df_his["PS"].sum())

    # Val yili ulusal oranlari (opsiyonel yakin donem prior kalibrasyonu)
    nat_lc_va    = float(df_va["Claim"].sum() / df_va["Exposure"].sum())
    nat_unit_va  = float(df_va["DHPS"].sum() / df_va["PS"].sum())

    # --- 2. IL BAZLI CREDIBILITY / EMPIRICAL BAYES BAZ MODELI (Baseline 1) ---
    # Buhlmann-Straub benzeri daraltma (shrinkage): Z_p = E_p / (E_p + K)
    prov_his = df_his.groupby(PROV_COL).agg(
        his_claim=("Claim", "sum"),
        his_exp=("Exposure", "sum"),
        his_dhps=("DHPS", "sum"),
        his_ps=("PS", "sum"),
    ).reset_index()
    prov_his["raw_prov_lc"]   = prov_his["his_claim"] / prov_his["his_exp"]
    prov_his["raw_prov_unit"] = prov_his["his_dhps"] / prov_his["his_ps"]

    med_exp = prov_his["his_exp"].median()
    med_ps  = prov_his["his_ps"].median()
    z_exp   = prov_his["his_exp"] / (prov_his["his_exp"] + 0.5 * med_exp)
    z_ps    = prov_his["his_ps"]  / (prov_his["his_ps"]  + 0.5 * med_ps)

    prov_his["cred_prov_lc"]   = z_exp * prov_his["raw_prov_lc"]   + (1.0 - z_exp) * nat_lc_his
    prov_his["cred_prov_unit"] = z_ps  * prov_his["raw_prov_unit"] + (1.0 - z_ps)  * nat_unit_his

    df_te = df_te.merge(prov_his[[PROV_COL, "cred_prov_lc", "cred_prov_unit"]], on=PROV_COL, how="left")
    df_te["cred_prov_lc"]   = df_te["cred_prov_lc"].fillna(nat_lc_his)
    df_te["cred_prov_unit"] = df_te["cred_prov_unit"].fillna(nat_unit_his)

    # --- 3. EXPOSURE-ONLY HURDLE MODEL (Baseline 2 - ERA5 YOK) ---
    sc_exp = StandardScaler()
    Xtr_e = sm.add_constant(sc_exp.fit_transform(df_his[["log_coverage"]].values), has_constant="add")
    Xte_e = sm.add_constant(sc_exp.transform(df_te[["log_coverage"]].values), has_constant="add")
    lr_exp = sm.Logit(df_his["Claim_Is"].values, Xtr_e).fit(method="bfgs", maxiter=300, disp=0)
    p_occ_exp_te = lr_exp.predict(Xte_e)
    p_occ_exp_tr = lr_exp.predict(Xtr_e)

    # Hasarli satirlarda ortalama kayip orani (Exposure-agirlikli)
    sev_his = df_his[df_his["Claim_Is"] == 1]
    cond_lc_const = float(sev_his["Claim"].sum() / sev_his["Exposure"].sum())
    raw_exp_hurdle_tr = p_occ_exp_tr * cond_lc_const * df_his["Exposure"].values
    scale_exp_hurdle  = float(df_his["Claim"].sum() / raw_exp_hurdle_tr.sum())
    pred_claim_exp_only = p_occ_exp_te * cond_lc_const * scale_exp_hurdle * df_te["Exposure"].values

    # Birim (DHPS) icin Exposure-Only
    cond_unit_const = float(sev_his["DHPS"].sum() / sev_his["PS"].sum())
    raw_unit_exp_tr = p_occ_exp_tr * cond_unit_const * df_his["PS"].values
    scale_unit_exp  = float(df_his["DHPS"].sum() / raw_unit_exp_tr.sum())
    pred_dhps_exp_only = p_occ_exp_te * cond_unit_const * scale_unit_exp * df_te["PS"].values

    # --- 4. ERA5 MODEL 1: PER-POLICY BINOMIAL (STRICTLY ERA5-ONLY, log_coverage YOK) ---
    sc_m = StandardScaler()
    Xtr_m = sm.add_constant(sc_m.fit_transform(df_his[MET_FEATURES].values), has_constant="add")
    Xte_m = sm.add_constant(sc_m.transform(df_te[MET_FEATURES].values), has_constant="add")

    # (a) Police bazli Binomial GLM (freq_weights = PS)
    m_binom_unit = sm.GLM(
        df_his["unit_rate_row"].values, Xtr_m,
        family=sm.families.Binomial(link=sm.families.links.Logit()),
        freq_weights=df_his["PS"].values.astype(float)
    ).fit(disp=0)
    p_unit_era5_te = m_binom_unit.predict(Xte_m)
    pred_dhps_era5_binom = p_unit_era5_te * df_te["PS"].values

    # (b) Exposure-agirlikli Binomial/Quasi-Poisson Loss Cost GLM (Strictly ERA5-Only)
    m_lc_era5 = sm.GLM(
        df_his["loss_cost_row"].clip(0, 1).values, Xtr_m,
        family=sm.families.Binomial(link=sm.families.links.Logit()),
        freq_weights=np.sqrt(df_his["Exposure"].values)  # Asiri buyuk tekil policelerin domine etmesini onleyen karekok-maruziyet agirligi
    ).fit(disp=0)
    raw_lc_era5_tr = m_lc_era5.predict(Xtr_m) * df_his["Exposure"].values
    scale_lc_era5  = float(df_his["Claim"].sum() / raw_lc_era5_tr.sum())
    pred_claim_era5_binom = m_lc_era5.predict(Xte_m) * scale_lc_era5 * df_te["Exposure"].values

    # --- 5. ERA5 MODEL 2: TWO-STAGE HURDLE (COMBINED OCCURRENCE x EXPOSURE-WEIGHTED SEVERITY) ---
    sc_f = StandardScaler()
    Xtr_f = sm.add_constant(sc_f.fit_transform(df_his[FULL_FEATURES].values), has_constant="add")
    Xte_f = sm.add_constant(sc_f.transform(df_te[FULL_FEATURES].values), has_constant="add")

    lr_full = sm.Logit(df_his["Claim_Is"].values, Xtr_f).fit(method="bfgs", maxiter=300, disp=0)
    p_occ_full_tr = lr_full.predict(Xtr_f)
    p_occ_full_te = lr_full.predict(Xte_f)

    # Stage 2: Exposure-weighted GLM for conditional loss ratio (given Claim_Is=1) using MET_FEATURES
    Xtr_s2 = sm.add_constant(sc_m.transform(sev_his[MET_FEATURES].values), has_constant="add")
    m_sev_glm = sm.GLM(
        sev_his["loss_cost_row"].clip(1e-5, 1.0).values, Xtr_s2,
        family=sm.families.Binomial(link=sm.families.links.Logit()),
        freq_weights=np.sqrt(sev_his["Exposure"].values)
    ).fit(disp=0)
    cond_lc_tr = m_sev_glm.predict(Xtr_m)
    cond_lc_te = m_sev_glm.predict(Xte_m)

    raw_hurdle_tr = p_occ_full_tr * cond_lc_tr * df_his["Exposure"].values
    scale_hurdle  = float(df_his["Claim"].sum() / raw_hurdle_tr.sum())
    pred_claim_hurdle = p_occ_full_te * cond_lc_te * scale_hurdle * df_te["Exposure"].values

    # Stage 2 Physical Coverage (DHPS/PS given Claim_Is=1)
    m_sev_unit = sm.GLM(
        sev_his["unit_rate_row"].clip(1e-5, 1.0).values, Xtr_s2,
        family=sm.families.Binomial(link=sm.families.links.Logit()),
        freq_weights=sev_his["PS"].values.astype(float)
    ).fit(disp=0)
    cond_u_tr = m_sev_unit.predict(Xtr_m)
    cond_u_te = m_sev_unit.predict(Xte_m)
    raw_u_hurdle_tr = p_occ_full_tr * cond_u_tr * df_his["PS"].values
    scale_u_hurdle  = float(df_his["DHPS"].sum() / raw_u_hurdle_tr.sum())
    pred_dhps_hurdle = p_occ_full_te * cond_u_te * scale_u_hurdle * df_te["PS"].values

    # --- 6. ERA5 MODEL 3: HYBRID (PROVINCIAL CREDIBILITY BASELINE x ERA5 ANOMALY MULTIPLIER) ---
    # Her ilde beklenen tarihsel baz orani, o yilin ERA5 konvektif anomalisi (m_lc_era5 / mean_his) ile olceklenir
    era5_rel_factor_tr = (m_lc_era5.predict(Xtr_m) * scale_lc_era5) / nat_lc_his
    era5_rel_factor_te = (m_lc_era5.predict(Xte_m) * scale_lc_era5) / nat_lc_his
    # Yari-agirlikli (0.5 baz + 0.5 ERA5 anomalisi) sönümlü çarpan
    mod_factor_te = 0.5 + 0.5 * era5_rel_factor_te
    pred_claim_hybrid = df_te["cred_prov_lc"].values * mod_factor_te * df_te["Exposure"].values
    # Portfoy toplami koruma olceklemesi (egitim donemi uzerinden)
    df_his_m = df_his.merge(prov_his[[PROV_COL, "cred_prov_lc", "cred_prov_unit"]], on=PROV_COL, how="left")
    raw_hyb_tr = df_his_m["cred_prov_lc"].values * (0.5 + 0.5 * era5_rel_factor_tr) * df_his["Exposure"].values
    pred_claim_hybrid = pred_claim_hybrid * float(df_his["Claim"].sum() / raw_hyb_tr.sum())

    era5_unit_factor_te = p_unit_era5_te / nat_unit_his
    pred_dhps_hybrid = df_te["cred_prov_unit"].values * (0.5 + 0.5 * era5_unit_factor_te) * df_te["PS"].values

    # Satir tahminlerini kaydet
    df_te["Pred_Claim_B0_National"]   = nat_lc_his * df_te["Exposure"].values
    df_te["Pred_Claim_B1_Provincial"] = df_te["cred_prov_lc"].values * df_te["Exposure"].values
    df_te["Pred_Claim_B2_ExpOnly"]    = pred_claim_exp_only
    df_te["Pred_Claim_E1_ERA5Binom"]  = pred_claim_era5_binom
    df_te["Pred_Claim_E2_Hurdle"]     = pred_claim_hurdle
    df_te["Pred_Claim_E3_Hybrid"]     = pred_claim_hybrid

    df_te["Pred_DHPS_B0_National"]    = nat_unit_his * df_te["PS"].values
    df_te["Pred_DHPS_B1_Provincial"]  = df_te["cred_prov_unit"].values * df_te["PS"].values
    df_te["Pred_DHPS_B2_ExpOnly"]     = pred_dhps_exp_only
    df_te["Pred_DHPS_E1_ERA5Binom"]   = pred_dhps_era5_binom
    df_te["Pred_DHPS_E2_Hurdle"]      = pred_dhps_hurdle
    df_te["Pred_DHPS_E3_Hybrid"]      = pred_dhps_hybrid

    # Il (Province) bazinda toplulastirma
    claim_cols = [c for c in df_te.columns if c.startswith("Pred_Claim_")]
    dhps_cols  = [c for c in df_te.columns if c.startswith("Pred_DHPS_")]

    agg_dict = {
        "Exposure": "sum",
        "Claim": "sum",
        "PS": "sum",
        "DHPS": "sum",
        "Claim_Is": "sum",
    }
    for c in claim_cols + dhps_cols:
        agg_dict[c] = "sum"

    prov_df = df_te.groupby(PROV_COL, as_index=False).agg(agg_dict)
    prov_df["Test_Year"] = test_year
    prov_df["Obs_LC_Pct"]   = (prov_df["Claim"] / prov_df["Exposure"]) * 100.0
    prov_df["Obs_Unit_Pct"] = (prov_df["DHPS"] / prov_df["PS"]) * 100.0

    for c in claim_cols:
        suf = c.replace("Pred_Claim_", "")
        prov_df[f"Pred_LC_Pct_{suf}"] = (prov_df[c] / prov_df["Exposure"]) * 100.0
    for c in dhps_cols:
        suf = c.replace("Pred_DHPS_", "")
        prov_df[f"Pred_Unit_Pct_{suf}"] = (prov_df[c] / prov_df["PS"]) * 100.0

    return prov_df

# =============================================================================
# [3] 2015, 2016 VE 2017 ROLLING TEST YILLARININ CALISTIRILMASI
# =============================================================================
print("\n=== [2] 2015, 2016 VE 2017 IL BAZLI PORTFOY MODELLERI CALISTIRILIYOR ===")
prov_frames = []
for yr in [2015, 2016, 2017]:
    prov_frames.append(run_portfolio_year(df, yr))

df_prov_all  = pd.concat(prov_frames, ignore_index=True)
df_prov_2017 = df_prov_all[df_prov_all["Test_Year"] == 2017].copy().reset_index(drop=True)

df_prov_all.to_csv(OUTPUT_DIR / "44_provincial_predictions_2015_2017.csv", index=False)

# =============================================================================
# [4] YILLIK TOPLAM PORTFOY KALIBRASYON TABLOSU (2015-2017)
# =============================================================================
print("\n=== [3] YILLIK PORTFOY TOPLAMI KARSILASTIRMASI (MILYON TL & ETKILENEN POLICE) ===")
model_Compare_keys = [
    ("B0_National",   "Baseline 0: National Historical Rate"),
    ("B1_Provincial", "Baseline 1: Provincial Credibility Rate"),
    ("B2_ExpOnly",    "Baseline 2: Exposure-Only Hurdle (No ERA5)"),
    ("E1_ERA5Binom",  "ERA5 Model 1: Strictly ERA5 Per-Policy Binomial"),
    ("E2_Hurdle",     "ERA5 Model 2: Two-Stage Hurdle (ERA5 + Exposure)"),
    ("E3_Hybrid",     "ERA5 Model 3: Provincial Credibility + ERA5 Anomaly"),
]

ann_rows = []
for yr, grp in df_prov_all.groupby("Test_Year"):
    obs_c = grp["Claim"].sum() / 1e6
    obs_d = grp["DHPS"].sum()
    row = {"Test_Year": yr, "Obs_Claim_M_TL": round(obs_c, 2), "Obs_Affected_Units": int(obs_d)}
    for k_short, _ in model_Compare_keys:
        row[f"Claim_{k_short}_M_TL"] = round(grp[f"Pred_Claim_{k_short}"].sum() / 1e6, 2)
        row[f"Units_{k_short}"]      = int(round(grp[f"Pred_DHPS_{k_short}"].sum()))
    ann_rows.append(row)

df_ann = pd.DataFrame(ann_rows)
print(df_ann.to_string(index=False))
df_ann.to_csv(OUTPUT_DIR / "44_annual_portfolio_totals_2015_2017.csv", index=False)

# =============================================================================
# [5] PAIRED PROVINCIAL BOOTSTRAP (B=1000) - LOSS COST (%), UNIT RATE (%), TOTAL CLAIM
# =============================================================================
print(f"\n=== [4] PAIRED PROVINCIAL BOOTSTRAP (B={N_BOOT}) ===")

def evaluate_provincial_bootstrap(df_eval, dataset_label):
    provs = df_eval[PROV_COL].unique()
    n_p   = len(provs)
    prov_idx_map = df_eval.groupby(PROV_COL).indices
    rng = np.random.default_rng(SEED)

    # Nokta metrikleri ve bootstrap depolari
    metrics_store = {
        k_short: {"lc_rho": [], "lc_mae": [], "lc_rmse": [], "unit_rho": [], "unit_mae": [], "claim_rho": [], "claim_r2": [], "claim_mae": []}
        for k_short, _ in model_Compare_keys
    }

    for _ in range(N_BOOT):
        drawn_p = rng.choice(provs, size=n_p, replace=True)
        idx = np.concatenate([prov_idx_map[p] for p in drawn_p])
        sub = df_eval.iloc[idx]

        obs_lc   = sub["Obs_LC_Pct"].values
        obs_unit = sub["Obs_Unit_Pct"].values
        obs_clm  = sub["Claim"].values / 1e6

        for k_short, _ in model_Compare_keys:
            p_lc   = sub[f"Pred_LC_Pct_{k_short}"].values
            p_unit = sub[f"Pred_Unit_Pct_{k_short}"].values
            p_clm  = sub[f"Pred_Claim_{k_short}"].values / 1e6

            # Baseline 0 sabit deger urettigi icin tek yilda Spearman rho = 0 (veya tanimsiz) olur
            rho_lc = spearmanr(obs_lc, p_lc)[0] if np.std(p_lc) > 1e-8 else 0.0
            rho_un = spearmanr(obs_unit, p_unit)[0] if np.std(p_unit) > 1e-8 else 0.0
            rho_cl = spearmanr(obs_clm, p_clm)[0]

            metrics_store[k_short]["lc_rho"].append(0.0 if np.isnan(rho_lc) else rho_lc)
            metrics_store[k_short]["lc_mae"].append(mean_absolute_error(obs_lc, p_lc))
            metrics_store[k_short]["lc_rmse"].append(np.sqrt(mean_squared_error(obs_lc, p_lc)))

            metrics_store[k_short]["unit_rho"].append(0.0 if np.isnan(rho_un) else rho_un)
            metrics_store[k_short]["unit_mae"].append(mean_absolute_error(obs_unit, p_unit))

            metrics_store[k_short]["claim_rho"].append(rho_cl)
            metrics_store[k_short]["claim_r2"].append(r2_score(obs_clm, p_clm))
            metrics_store[k_short]["claim_mae"].append(mean_absolute_error(obs_clm, p_clm))

    summary_rows = []
    obs_lc_all   = df_eval["Obs_LC_Pct"].values
    obs_unit_all = df_eval["Obs_Unit_Pct"].values
    obs_clm_all  = df_eval["Claim"].values / 1e6

    for k_short, m_title in model_Compare_keys:
        p_lc_all   = df_eval[f"Pred_LC_Pct_{k_short}"].values
        p_unit_all = df_eval[f"Pred_Unit_Pct_{k_short}"].values
        p_clm_all  = df_eval[f"Pred_Claim_{k_short}"].values / 1e6

        rho_lc_pt = spearmanr(obs_lc_all, p_lc_all)[0] if np.std(p_lc_all) > 1e-8 else 0.0
        rho_un_pt = spearmanr(obs_unit_all, p_unit_all)[0] if np.std(p_unit_all) > 1e-8 else 0.0

        def ci_bounds(arr):
            return round(float(np.percentile(arr, 2.5)), 4), round(float(np.percentile(arr, 97.5)), 4)

        lc_rho_lo, lc_rho_hi     = ci_bounds(metrics_store[k_short]["lc_rho"])
        lc_mae_lo, lc_mae_hi     = ci_bounds(metrics_store[k_short]["lc_mae"])
        unit_rho_lo, unit_rho_hi = ci_bounds(metrics_store[k_short]["unit_rho"])
        clm_r2_lo, clm_r2_hi     = ci_bounds(metrics_store[k_short]["claim_r2"])
        clm_mae_lo, clm_mae_hi   = ci_bounds(metrics_store[k_short]["claim_mae"])

        summary_rows.append({
            "Dataset": dataset_label,
            "Model": m_title,
            "LossCost_Spearman_rho": round(float(0.0 if np.isnan(rho_lc_pt) else rho_lc_pt), 4),
            "LossCost_rho_CI": f"[{lc_rho_lo:.3f}, {lc_rho_hi:.3f}]",
            "LossCost_MAE_Pct": round(float(mean_absolute_error(obs_lc_all, p_lc_all)), 4),
            "LossCost_MAE_CI": f"[{lc_mae_lo:.3f}, {lc_mae_hi:.3f}]",
            "UnitRate_Spearman_rho": round(float(0.0 if np.isnan(rho_un_pt) else rho_un_pt), 4),
            "UnitRate_rho_CI": f"[{unit_rho_lo:.3f}, {unit_rho_hi:.3f}]",
            "UnitRate_MAE_Pct": round(float(mean_absolute_error(obs_unit_all, p_unit_all)), 4),
            "TotalClaim_Spearman_rho": round(float(spearmanr(obs_clm_all, p_clm_all)[0]), 4),
            "TotalClaim_R2": round(float(r2_score(obs_clm_all, p_clm_all)), 4),
            "TotalClaim_R2_CI": f"[{clm_r2_lo:.3f}, {clm_r2_hi:.3f}]",
            "TotalClaim_MAE_M_TL": round(float(mean_absolute_error(obs_clm_all, p_clm_all)), 4),
            "TotalClaim_MAE_CI": f"[{clm_mae_lo:.3f}, {clm_mae_hi:.3f}]",
        })

    return pd.DataFrame(summary_rows), metrics_store

df_boot_2017, store_2017 = evaluate_provincial_bootstrap(df_prov_2017, "Test_2017 (N=80 Provinces)")
df_boot_roll, store_roll = evaluate_provincial_bootstrap(df_prov_all,  "Pooled_2015_2017 (N=240 Province-Years)")

df_boot_both = pd.concat([df_boot_2017, df_boot_roll], ignore_index=True)
print("\nIL BAZLI PORTFOY BAZ MODELLERI VE ERA5 MODELLERI KARSILASTIRMASI:")
print(df_boot_both.to_string(index=False))
df_boot_both.to_csv(OUTPUT_DIR / "44_provincial_baselines_bootstrap_summary.csv", index=False)

# Paired Differences (ERA5 Modelleri vs Baseline 0, Baseline 1 ve Baseline 2)
paired_prov_rows = []
for ds_lbl, st in [("Test_2017", store_2017), ("Pooled_2015_2017", store_roll)]:
    for era_k in ["E1_ERA5Binom", "E2_Hurdle", "E3_Hybrid"]:
        for base_k in ["B0_National", "B1_Provincial", "B2_ExpOnly"]:
            d_lc_rho  = np.asarray(st[era_k]["lc_rho"])   - np.asarray(st[base_k]["lc_rho"])
            d_un_rho  = np.asarray(st[era_k]["unit_rho"]) - np.asarray(st[base_k]["unit_rho"])
            d_clm_mae = np.asarray(st[era_k]["claim_mae"])- np.asarray(st[base_k]["claim_mae"])  # Negatif = ERA5 daha dusuk hata!
            paired_prov_rows.append({
                "Dataset": ds_lbl,
                "Comparison": f"{era_k} minus {base_k}",
                "d_LossCost_rho": round(float(d_lc_rho.mean()), 4),
                "d_LossCost_rho_CI": f"[{np.percentile(d_lc_rho, 2.5):+.3f}, {np.percentile(d_lc_rho, 97.5):+.3f}]",
                "d_UnitRate_rho": round(float(d_un_rho.mean()), 4),
                "d_UnitRate_rho_CI": f"[{np.percentile(d_un_rho, 2.5):+.3f}, {np.percentile(d_un_rho, 97.5):+.3f}]",
                "d_Claim_MAE_M_TL": round(float(d_clm_mae.mean()), 4),
                "d_Claim_MAE_CI": f"[{np.percentile(d_clm_mae, 2.5):+.3f}, {np.percentile(d_clm_mae, 97.5):+.3f}]",
            })

df_paired_prov = pd.DataFrame(paired_prov_rows)
print("\nPAIRED PROVINCIAL BOOTSTRAP DIFFERENCES (ERA5 Modelleri vs Baselines):")
print(df_paired_prov.to_string(index=False))
df_paired_prov.to_csv(OUTPUT_DIR / "44_provincial_paired_differences_bootstrap.csv", index=False)

print(f"\n[TAMAMLANDI] 44_portfolio_baselines ciktilari kaydedildi -> {OUTPUT_DIR}")
