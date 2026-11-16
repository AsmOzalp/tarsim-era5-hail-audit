
# -*- coding: utf-8 -*-
"""
05_hazard_maps_and_consistency.py
==========================
Stage 2 - Hazard Mapping and Climatological Consistency Analysis

Main objectives
---------------

1. Generate three different Out-of-Fold (OOF) hazard maps under
   81-fold K-Means Leave-One-Region-Out Cross-Validation (LORO-CV):

   - Model A (Per-Policy Binomial ERA5-Only):
     DHPS ~ Binomial(PS, p_i), using only the four ERA5 meteorological
     predictors (excluding log_coverage).

   - Model B (Exposure-Standardized Binary Logit):
     The model is trained using ERA5 predictors and log_coverage.
     However, when generating OOF predictions, log_coverage is fixed
     at the portfolio-wide median (zero after fold-specific
     standardization) across all test locations. This provides
     exposure-standardized ERA5 predictions.

   - Model C (Unadjusted Binary Logit):
     Two specifications are evaluated: ERA5-only and ERA5 with
     observed exposure. These provide a direct comparison with
     the original manuscript figures.

2. Generate publication-quality multi-panel visualizations
   (combined or separate versions of Figures 3 and 4):

   - (a) Pure Meteorological Hail Hazard Map (LORO-CV OOF)
   - (b) Inter-Annual Climatological Variability (Inter-annual SD)
   - (c) Insured Exposure and Policy Density Map
         (assessment of observational exposure bias)
   - (d) Regional Internal Consistency Assessment across 81 regions,
         including 95% block bootstrap confidence intervals.

3. Produce elevation-band and regional climatological summary tables
   for comparison with Kahraman et al. (2016).
"""

from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
import lightgbm as lgb
from scipy.stats import spearmanr, pearsonr
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# =============================================================================
# [1] CONFIGURATION AND DATA LOADING
# =============================================================================

FILE_PATH  = Path(r"C:\Users\m-a-o\Desktop\TarsimData\V2\model_ready_data_spatial_final_v1.csv")
OUTPUT_DIR = FILE_PATH.parent / "revision_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MET_FEATURES  = ["WMAXSHR", "CAPE_dur", "WMAXSHR_burst", "h_agl"]
FULL_FEATURES = MET_FEATURES + ["log_coverage"]
TARGET_BIN    = "Claim_Is"
N_CLUSTERS    = 81
N_BOOT        = 1000
SEED          = 42

print("=== [1] LOADING DATA AND CONSTRUCTING 81 LORO-CV REGIONS ===")

df = pd.read_csv(FILE_PATH, low_memory=False)

needed = FULL_FEATURES + [
    TARGET_BIN, "DHPS", "PS", "Exposure", "Claim",
    "yil", "LATITUDE", "LONGITUDE", "RAKIM", "Il Adi"
]

df = df.dropna(subset=needed).copy().reset_index(drop=True)
df = df[df["PS"] >= 1].copy().reset_index(drop=True)

df["DHPS"] = np.minimum(df["DHPS"], df["PS"])
df["p_obs_unit"] = df["DHPS"] / df["PS"]

# Construct 81 K-Means spatial clusters using unique village coordinates.
u_coords = df[["LATITUDE", "LONGITUDE"]].drop_duplicates().reset_index(drop=True)

km = KMeans(
    n_clusters=N_CLUSTERS,
    random_state=SEED,
    n_init=10
)

u_coords["Cluster_81"] = km.fit_predict(
    u_coords[["LATITUDE", "LONGITUDE"]]
)

df = df.merge(
    u_coords,
    on=["LATITUDE", "LONGITUDE"],
    how="left"
)

print(
    f"  Total observations: {len(df):,} | "
    f"Unique locations: {len(u_coords):,} | "
    f"Clusters: {N_CLUSTERS}"
)

# =============================================================================
# [2] GENERATION OF 81-FOLD LORO-CV OUT-OF-FOLD (OOF) PREDICTIONS
# =============================================================================

print("\n=== [2] GENERATING 81-FOLD LORO-CV OOF PREDICTIONS ===")

oof_binom_era5      = np.full(len(df), np.nan)  # Model A: Binomial GLM (ERA5-only, freq_weights=PS)
oof_lgb_binom_era5  = np.full(len(df), np.nan)  # Model A2: LightGBM Binomial (ERA5-only, weight=PS)
oof_logit_exp_std   = np.full(len(df), np.nan)  # Model B: Binary Logit (ERA5 + standardized exposure)
oof_logit_era5_raw  = np.full(len(df), np.nan)  # Model C1: Binary Logit (unadjusted ERA5-only)
oof_logit_full_raw  = np.full(len(df), np.nan)  # Model C2: Binary Logit (ERA5 + observed exposure)

for cid in range(N_CLUSTERS):

    te_m = (df["Cluster_81"] == cid).values
    tr_m = ~te_m

    if te_m.sum() == 0:
        continue

    # --- 1. Standardization using ERA5 predictors only ---

    sc_met = StandardScaler()

    Xtr_m = sm.add_constant(
        sc_met.fit_transform(df.loc[tr_m, MET_FEATURES].values),
        has_constant="add"
    )

    Xte_m = sm.add_constant(
        sc_met.transform(df.loc[te_m, MET_FEATURES].values),
        has_constant="add"
    )

    # Model A: Per-policy Binomial GLM
    # Response: DHPS / PS
    # Frequency weights: PS
    # Predictors: ERA5 variables only

    m_binom = sm.GLM(
        df.loc[tr_m, "p_obs_unit"].values,
        Xtr_m,
        family=sm.families.Binomial(
            link=sm.families.links.Logit()
        ),
        freq_weights=df.loc[tr_m, "PS"].values.astype(float)
    ).fit(disp=0)

    oof_binom_era5[te_m] = m_binom.predict(Xte_m)

    # Model A2: LightGBM Per-Policy Binomial Model
    # Strictly ERA5-only, with policy-unit weighting.

    m_lgb_b = lgb.LGBMRegressor(
        objective="cross_entropy",
        n_estimators=250,
        learning_rate=0.03,
        max_depth=5,
        num_leaves=23,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=SEED,
        n_jobs=-1,
        verbose=-1
    )

    m_lgb_b.fit(
        df.loc[tr_m, MET_FEATURES].values,
        df.loc[tr_m, "p_obs_unit"].values,
        sample_weight=df.loc[tr_m, "PS"].values.astype(float)
    )

    oof_lgb_binom_era5[te_m] = np.clip(
        m_lgb_b.predict(df.loc[te_m, MET_FEATURES].values),
        1e-6,
        1.0 - 1e-6
    )

    # Model C1: Unweighted Binary Logit
    # Predictors: ERA5 variables only
    # Response: Binary claim indicator (Claim_Is)

    m_bin_met = sm.Logit(
        df.loc[tr_m, TARGET_BIN].values,
        Xtr_m
    ).fit(
        method="bfgs",
        maxiter=300,
        disp=0
    )

    oof_logit_era5_raw[te_m] = m_bin_met.predict(Xte_m)

    # --- 2. Standardization using the full predictor set
    # (ERA5 + log_coverage) ---

    sc_full = StandardScaler()

    Xtr_f = sm.add_constant(
        sc_full.fit_transform(df.loc[tr_m, FULL_FEATURES].values),
        has_constant="add"
    )

    Xte_f_raw = sm.add_constant(
        sc_full.transform(df.loc[te_m, FULL_FEATURES].values),
        has_constant="add"
    )

    # Exposure-standardized test matrix:
    # Set the standardized log_coverage variable to zero.
    # A z-score of zero corresponds to the training-fold mean exposure.

    Xte_f_std = Xte_f_raw.copy()
    Xte_f_std[:, -1] = 0.0

    # Model C2: Binary Logit using ERA5 and observed exposure

    m_bin_full = sm.Logit(
        df.loc[tr_m, TARGET_BIN].values,
        Xtr_f
    ).fit(
        method="bfgs",
        maxiter=300,
        disp=0
    )

    # Predictions using observed exposure
    oof_logit_full_raw[te_m] = m_bin_full.predict(Xte_f_raw)

    # Predictions after standardizing exposure
    oof_logit_exp_std[te_m] = m_bin_full.predict(Xte_f_std)


# Store all OOF predictions in the main dataset.

df["oof_binom_era5"]     = oof_binom_era5
df["oof_lgb_binom_era5"] = oof_lgb_binom_era5
df["oof_logit_exp_std"]  = oof_logit_exp_std
df["oof_logit_era5_raw"] = oof_logit_era5_raw
df["oof_logit_full_raw"] = oof_logit_full_raw


# =============================================================================
# [3] REGIONAL INTERNAL CONSISTENCY ASSESSMENT
#     (81 REGIONS - R2-C12)
# =============================================================================

print(
    "\n=== [3] REGIONAL INTERNAL CONSISTENCY "
    "AND EXPOSURE-BIAS ASSESSMENT ==="
)

reg_agg = []

for cid, grp in df.groupby("Cluster_81"):

    total_ps   = grp["PS"].sum()
    total_dhps = grp["DHPS"].sum()

    reg_agg.append({

        "Cluster_81": cid,
        "N_Obs": len(grp),
        "Total_PS": total_ps,
        "Mean_log_coverage": grp["log_coverage"].mean(),
        "Mean_RAKIM": grp["RAKIM"].mean(),

        "Empirical_Binary_Freq": grp[TARGET_BIN].mean(),

        "Empirical_Unit_Rate_DHPS_PS": total_dhps / total_ps,

        " empirical_Loss_Cost": (
            grp["Claim"].sum() / grp["Exposure"].sum()
        ),

        # Regional averages of out-of-fold model predictions.

        "Pred_Binom_ERA5_UnitWeighted": np.average(
            grp["oof_binom_era5"],
            weights=grp["PS"]
        ),

        "Pred_Binom_ERA5_Unweighted": (
            grp["oof_binom_era5"].mean()
        ),

        "Pred_LGB_Binom_ERA5": np.average(
            grp["oof_lgb_binom_era5"],
            weights=grp["PS"]
        ),

        "Pred_Logit_ExpStandardized_ERA5": (
            grp["oof_logit_exp_std"].mean()
        ),

        "Pred_Logit_ERA5_Raw": (
            grp["oof_logit_era5_raw"].mean()
        ),

        "Pred_Logit_Full_WithExposure": (
            grp["oof_logit_full_raw"].mean()
        ),
    })


df_reg = pd.DataFrame(reg_agg)

df_reg.to_csv(
    OUTPUT_DIR / "42_regional_81_internal_consistency.csv",
    index=False
)


# Calculate correlations across the 81 regions
# and obtain 95% confidence intervals using 1,000 bootstrap iterations.

rng = np.random.default_rng(SEED)

consistency_pairs = [

    (
        "Original Full Model (ERA5 + log_coverage) vs Empirical Binary Freq",
        "Pred_Logit_Full_WithExposure",
        "Empirical_Binary_Freq"
    ),

    (
        "Exposure-Standardized Logit (Pure ERA5) vs Empirical Binary Freq",
        "Pred_Logit_ExpStandardized_ERA5",
        "Empirical_Binary_Freq"
    ),

    (
        "Exposure-Standardized Logit (Pure ERA5) vs Empirical Unit Rate (DHPS/PS)",
        "Pred_Logit_ExpStandardized_ERA5",
        "Empirical_Unit_Rate_DHPS_PS"
    ),

    (
        "Binomial GLM (Pure ERA5) vs Empirical Unit Rate (DHPS/PS)",
        "Pred_Binom_ERA5_UnitWeighted",
        "Empirical_Unit_Rate_DHPS_PS"
    ),

    (
        "LightGBM Binomial (Pure ERA5) vs Empirical Unit Rate (DHPS/PS)",
        "Pred_LGB_Binom_ERA5",
        "Empirical_Unit_Rate_DHPS_PS"
    ),

    (
        "Confounding Check: Mean log_coverage vs Empirical Binary Freq",
        "Mean_log_coverage",
        "Empirical_Binary_Freq"
    ),

    (
        "Confounding Check: Mean log_coverage vs Empirical Unit Rate (DHPS/PS)",
        "Mean_log_coverage",
        "Empirical_Unit_Rate_DHPS_PS"
    ),
]

cons_rows = []

for label, col_x, col_y in consistency_pairs:

    x_v = df_reg[col_x].values
    y_v = df_reg[col_y].values

    rho_pt, p_val = spearmanr(x_v, y_v)
    r_pt, _ = pearsonr(x_v, y_v)

    boot_rhos = []

    for _ in range(N_BOOT):

        b_idx = rng.integers(
            0,
            len(df_reg),
            size=len(df_reg)
        )

        if len(np.unique(x_v[b_idx])) > 2:

            boot_rhos.append(
                spearmanr(x_v[b_idx], y_v[b_idx])[0]
            )

    lo, hi = np.percentile(
        boot_rhos,
        [2.5, 97.5]
    )

    cons_rows.append({

        "Assessment_Pair": label,
        "Spearman_rho": round(float(rho_pt), 4),
        "Rho_95CI_Low": round(float(lo), 4),
        "Rho_95CI_High": round(float(hi), 4),
        "p_value": float(f"{p_val:.4e}"),
        "Pearson_r": round(float(r_pt), 4),

    })


df_cons = pd.DataFrame(cons_rows)

print(
    "\nRegional Internal Consistency Results "
    "(81 Spatial Regions):"
)

print(df_cons.to_string(index=False))

df_cons.to_csv(
    OUTPUT_DIR / "42_internal_consistency_correlations.csv",
    index=False
)


# =============================================================================
# [4] CLIMATOLOGICAL COMPARISON WITH KAHRAMAN ET AL. (2016):
#     ELEVATION BANDS (R1-M10)
# =============================================================================

print(
    "\n=== [4] CLIMATOLOGICAL HAZARD SUMMARY "
    "BY ELEVATION BAND (R1-M10) ==="
)

df["Elev_Band"] = pd.cut(
    df["RAKIM"],
    bins=[-100, 500, 1000, 1500, 5000],
    labels=[
        "Lowland (<500 m)",
        "Mid-Elevation (500-1000 m)",
        "High Plateau (1000-1500 m)",
        "Highland (>1500 m)"
    ]
)

elev_summary = df.groupby(
    "Elev_Band",
    observed=False
).agg(

    N_Obs=("Claim_Is", "count"),

    N_Villages=(
        "LATITUDE",
        lambda s: len(np.unique(s))
    ),

    Mean_h_agl_m=("h_agl", "mean"),

    Mean_WMAXSHR=("WMAXSHR", "mean"),

    Empirical_Binary_Freq_Pct=(
        "Claim_Is",
        lambda s: s.mean() * 100
    ),

    Empirical_Unit_Rate_Pct=(
        "p_obs_unit",
        lambda s: s.mean() * 100
    ),

    OOF_Binom_ERA5_Pct=(
        "oof_binom_era5",
        lambda s: s.mean() * 100
    ),

    OOF_Logit_ExpStd_ERA5_Pct=(
        "oof_logit_exp_std",
        lambda s: s.mean() * 100
    ),

).reset_index()


print(elev_summary.round(2).to_string(index=False))

elev_summary.round(3).to_csv(
    OUTPUT_DIR / "42_elevation_climatology_kahraman_comparison.csv",
    index=False
)


# =============================================================================
# [5] MULTI-PANEL PUBLICATION-QUALITY HAZARD MAPS
#     (PURE HAZARD + UNCERTAINTY + EXPOSURE + REGIONAL CONSISTENCY)
# =============================================================================

print("\n=== [5] GENERATING PUBLICATION-QUALITY MAPS ===")


# Calculate 10-year village-level averages and inter-annual
# standard deviations of annual out-of-fold predictions.

vil_yr = df.groupby(
    ["LATITUDE", "LONGITUDE", "yil"],
    as_index=False
).agg(

    binom_era5=("oof_binom_era5", "mean"),
    lgb_binom_era5=("oof_lgb_binom_era5", "mean"),
    exp_std_era5=("oof_logit_exp_std", "mean"),

    total_ps=("PS", "sum"),
    total_exp=("Exposure", "sum"),

)


vil_map = vil_yr.groupby(
    ["LATITUDE", "LONGITUDE"],
    as_index=False
).agg(

    mean_binom_era5=("binom_era5", "mean"),
    sd_binom_era5=("binom_era5", "std"),

    mean_lgb_era5=("lgb_binom_era5", "mean"),
    sd_lgb_era5=("lgb_binom_era5", "std"),

    mean_exp_std_era5=("exp_std_era5", "mean"),
    sd_exp_std_era5=("exp_std_era5", "std"),

    mean_ps=("total_ps", "mean"),

    mean_log_exp=(
        "total_exp",
        lambda s: np.log10(np.maximum(s.mean(), 1.0))
    ),

)


# Replace missing standard deviations with zero.

vil_map["sd_lgb_era5"] = vil_map["sd_lgb_era5"].fillna(0.0)
vil_map["sd_binom_era5"] = vil_map["sd_binom_era5"].fillna(0.0)
vil_map["sd_exp_std_era5"] = vil_map["sd_exp_std_era5"].fillna(0.0)


# Convert values to a 0-100 index using 1st and 99th percentile clipping.

def to_index_0_100(arr):

    lo, hi = np.percentile(arr, [1, 99])

    return np.clip(
        (arr - lo) / max(hi - lo, 1e-8) * 100.0,
        0.0,
        100.0
    )


vil_map["Hazard_Index_LGB"] = to_index_0_100(
    vil_map["mean_lgb_era5"].values
)

vil_map["Uncert_Index_LGB"] = to_index_0_100(
    vil_map["sd_lgb_era5"].values
)

vil_map["Hazard_Index_Binom"] = to_index_0_100(
    vil_map["mean_binom_era5"].values
)

vil_map["Hazard_Index_ExpStd"] = to_index_0_100(
    vil_map["mean_exp_std_era5"].values
)


fig, axes = plt.subplots(
    2, 2,
    figsize=(16, 11)
)


# -------------------------------------------------------------------------
# Panel (a): Out-of-Fold Per-Policy Binomial ERA5 Surface
#             (LightGBM Binomial, Pure ERA5)
# -------------------------------------------------------------------------

ax = axes[0, 0]

sc1 = ax.scatter(
    vil_map["LONGITUDE"],
    vil_map["LATITUDE"],
    c=vil_map["Hazard_Index_LGB"],
    cmap="viridis",
    s=2.5,
    alpha=0.8,
    rasterized=True
)

cb1 = plt.colorbar(
    sc1,
    ax=ax,
    pad=0.02
)

cb1.set_label(
    "ERA5 Damage Propensity Index (0-100)",
    fontsize=10,
    fontweight="bold"
)

ax.set_title(
    "(a) Out-of-Fold Per-Policy Binomial ERA5 Surface (LORO-CV)",
    fontsize=11.5,
    fontweight="bold"
)

ax.set_xlabel("Longitude (°E)", fontsize=10)
ax.set_ylabel("Latitude (°N)", fontsize=10)
ax.grid(alpha=0.3, ls="--")


# -------------------------------------------------------------------------
# Panel (b): Inter-Annual Volatility Surface
#             (Standard Deviation of Annual OOF Predictions)
# -------------------------------------------------------------------------

ax = axes[0, 1]

sc2 = ax.scatter(
    vil_map["LONGITUDE"],
    vil_map["LATITUDE"],
    c=vil_map["Uncert_Index_LGB"],
    cmap="magma",
    s=2.5,
    alpha=0.8,
    rasterized=True
)

cb2 = plt.colorbar(
    sc2,
    ax=ax,
    pad=0.02
)

cb2.set_label(
    "Inter-annual Volatility Index (0-100)",
    fontsize=10,
    fontweight="bold"
)

ax.set_title(
    "(b) Inter-annual Volatility Surface "
    "(SD of Annual OOF Predictions)",
    fontsize=11.5,
    fontweight="bold"
)

ax.set_xlabel("Longitude (°E)", fontsize=10)
ax.set_ylabel("Latitude (°N)", fontsize=10)
ax.grid(alpha=0.3, ls="--")


# -------------------------------------------------------------------------
# Panel (c): Insured Exposure Density Map
#             (Assessment of Observational Exposure Bias)
# -------------------------------------------------------------------------

ax = axes[1, 0]

sc3 = ax.scatter(
    vil_map["LONGITUDE"],
    vil_map["LATITUDE"],
    c=np.log10(np.maximum(vil_map["mean_ps"], 1.0)),
    cmap="YlOrBr",
    s=2.5,
    alpha=0.8,
    rasterized=True
)

cb3 = plt.colorbar(
    sc3,
    ax=ax,
    pad=0.02
)

cb3.set_label(
    "log10(Mean Insured Policy Units per Year)",
    fontsize=10,
    fontweight="bold"
)

ax.set_title(
    "(c) Insured Exposure Density "
    "(Mean Policy Units per Site)",
    fontsize=11.5,
    fontweight="bold"
)

ax.set_xlabel("Longitude (°E)", fontsize=10)
ax.set_ylabel("Latitude (°N)", fontsize=10)
ax.grid(alpha=0.3, ls="--")


# -------------------------------------------------------------------------
# Panel (d): Regional Internal Consistency Assessment
#             (81 LORO-CV Regions - LightGBM Binomial, Pure ERA5)
# -------------------------------------------------------------------------

ax = axes[1, 1]

x_reg = df_reg["Pred_LGB_Binom_ERA5"].values * 100.0
y_reg = df_reg["Empirical_Unit_Rate_DHPS_PS"].values * 100.0

sizes = np.clip(
    df_reg["Total_PS"].values / 1200.0,
    15,
    350
)

rho_b, p_b = spearmanr(x_reg, y_reg)

ax.scatter(
    x_reg,
    y_reg,
    s=sizes,
    c="#1f77b4",
    alpha=0.65,
    edgecolors="black",
    linewidth=0.8
)


# Add an ordinary least squares trend line.

m_sl, b_int = np.polyfit(
    x_reg,
    y_reg,
    1
)

x_line = np.linspace(
    x_reg.min(),
    x_reg.max(),
    100
)

ax.plot(
    x_line,
    m_sl * x_line + b_int,
    "r--",
    lw=2,
    label=f"Linear Trend (Spearman ρ = {rho_b:.2f}, p = {p_b:.4f})"
)

ax.set_xlabel(
    "Out-of-Fold Predicted Policy Damage Rate (%) "
    "[Pure ERA5, LightGBM]",
    fontsize=10.5,
    fontweight="bold"
)

ax.set_ylabel(
    "Empirical Affected Policy Rate (DHPS / PS, %)",
    fontsize=10.5,
    fontweight="bold"
)

ax.set_title(
    "(d) Regional Internal Consistency Assessment "
    "(81 LORO-CV Folds)",
    fontsize=11.5,
    fontweight="bold"
)

ax.legend(
    loc="upper left",
    fontsize=10
)

ax.grid(alpha=0.3, ls="--")


# -------------------------------------------------------------------------
# Save the final multi-panel figure.
# -------------------------------------------------------------------------

plt.tight_layout()

out_map = OUTPUT_DIR / "42_Figure3_4_decomposed_hazard_and_exposure.png"

fig.savefig(
    out_map,
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)


print(
    "\n[COMPLETED] All outputs and maps generated successfully."
)

print(
    f"Output directory: {OUTPUT_DIR}"
)
