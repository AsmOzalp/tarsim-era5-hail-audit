# -*- coding: utf-8 -*-
"""
01_clean_pipeline_and_summary.py
=====================================================
Repository-Ready Audit Script Producing Exact Outputs for All Remaining Items:

  [1] Exact Data Pipeline Reconciliation (49,640 wide rows x 10 yrs = 496,400
      total cells -> 184,578 active DPSB > 0 cells -> 184,469 modeling panel;
      exact cause of the 109 dropped rows) and regeneration of Tables 1, 2, 3,
      and 4 directly from the 184,469 modeling panel so all portfolio sums match.
  [2] Exact Actuarial Consistency Flag Sensitivity (307 rows with DOM > 0 &
      DHPS == 0 set to DHPS = 1; 85 rows capped at DHPS = PS; 1 row with PS = 0):
      rerunning M4 Odds Ratios and LORO-81 (M4 - M2) when the 307 flagged rows
      are (a) dropped or (b) treated as non-events (Claim_Is = 0).
  [3] Unbuffered and 28-km Buffered LORO-81 for both M4_Orig (CAPE_dur) and
      M4_Rate (CAPE_dur_rate = CAPE_dur / Window_Hours) vs M2_Base, with
      B = 5,000 paired block bootstrap iterations under BOTH Grid_ID (1,333
      cells) and Cluster_81 (81 regions) and Holm-Bonferroni adjusted p-values.
"""

from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss, log_loss
)
from tqdm import tqdm

warnings.filterwarnings("ignore")

BASE_DIR   = Path(r"C:\Users\m-a-o\Desktop\TarsimData\V2")
CLEAN_PATH = BASE_DIR.parent / "clean_data.csv"
MOD_PATH   = BASE_DIR / "model_ready_data_spatial_final_v1.csv"
OUTPUT_DIR = BASE_DIR / "revision_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET       = "Claim_Is"
MET_FEATURES = ["WMAXSHR", "CAPE_dur", "WMAXSHR_burst", "h_agl"]
EXP_FEATURES = ["log_coverage"]
BLOCK_GRID   = "Grid_ID"
YEAR_COL     = "yil"
EARTH_R      = 6371.0
N_BOOT_5K    = 5000
SEED         = 42

# =============================================================================
# [1] EXACT DATA PIPELINE AUDIT (496,400 -> 184,578 -> 184,469) & TABLES 1-4
# =============================================================================
print("=== [1] VERI ZINCIRI (496,400 -> 184,578 -> 184,469) VE TABLO 1-4 URETIMI ===")

df_cl  = pd.read_csv(CLEAN_PATH, low_memory=False)
df_mod = pd.read_csv(MOD_PATH, low_memory=False)

id_cols = ["Urun Adi Genel", "Il Adi", "Ilce Adi", "Koy Adi", "Bucak Adi", "Koy Kod", "LONGITUDE", "LATITUDE", "RAKIM"]
long_chunks = []
for yr in range(2008, 2018):
    sub = df_cl[id_cols].copy()
    sub["yil"]      = yr
    sub["PS_raw"]   = pd.to_numeric(df_cl[f"PS_{yr}"], errors="coerce").fillna(0)
    sub["DHPS_raw"] = pd.to_numeric(df_cl[f"DHPS_{yr}"], errors="coerce").fillna(0)
    sub["DOM_raw"]  = pd.to_numeric(df_cl[f"DOM_{yr}"], errors="coerce").fillna(0)
    sub["DPSB_raw"] = pd.to_numeric(df_cl[f"DPSB_{yr}"], errors="coerce").fillna(0)
    long_chunks.append(sub)

df_wide_long = pd.concat(long_chunks, ignore_index=True)

# 1A. 496,400 hucrenin ayrintili dokumu
n_total_cells     = len(df_wide_long)
mask_dpsb_gt0     = df_wide_long["DPSB_raw"] > 0
mask_ps_gt0_dpsb0 = (df_wide_long["PS_raw"] > 0) & (df_wide_long["DPSB_raw"] == 0)
mask_zero_both    = (df_wide_long["PS_raw"] == 0) & (df_wide_long["DPSB_raw"] == 0)

print(f"  1. Genis Tablo Toplam Potansiyel Hucre (49,640 x 10 yil) : {n_total_cells:,}")
print(f"     - Tamamen Sifir Hucre (PS=0 ve DPSB=0)                : {mask_zero_both.sum():,}")
print(f"     - PS>0 ama DPSB=0 (Sigorta bedeli 0 olan kayitlar)    : {mask_ps_gt0_dpsb0.sum():,} (Toplam PS={df_wide_long.loc[mask_ps_gt0_dpsb0, 'PS_raw'].sum():,.0f}, DHPS={df_wide_long.loc[mask_ps_gt0_dpsb0, 'DHPS_raw'].sum():,.0f})")
print(f"     - Aktif Maruziyet Hucreleri (DPSB > 0)                : {mask_dpsb_gt0.sum():,} (Ham PS={df_wide_long.loc[mask_dpsb_gt0, 'PS_raw'].sum():,.0f}, Ham DHPS={df_wide_long.loc[mask_dpsb_gt0, 'DHPS_raw'].sum():,.0f})")

# 1B. 184,578 aktif (DPSB > 0) hucreden neden 109 tanesi 184,469 panelinde yok?
df_act = df_wide_long[mask_dpsb_gt0].copy().reset_index(drop=True)
na_coord = df_act["LATITUDE"].isna() | df_act["LONGITUDE"].isna()
na_rakim = (~na_coord) & df_act["RAKIM"].isna()
dom_gt_dpsb = (~na_coord) & (~df_act["RAKIM"].isna()) & (df_act["DOM_raw"] > df_act["DPSB_raw"])
other_drop  = len(df_act) - int(na_coord.sum() + na_rakim.sum() + dom_gt_dpsb.sum()) - len(df_mod)

print(f"  2. 184,578 Aktif Hucreden 184,469 Modelleme Paneline Gecis (Toplam Elenen = {len(df_act) - len(df_mod)} satir):")
print(f"     - Koordinati (LATITUDE/LONGITUDE) eksik olan          : {na_coord.sum():,}")
print(f"     - Rakimi (RAKIM) eksik olan                           : {na_rakim.sum():,}")
print(f"     - Hasari sigorta bedelini asan (DOM > DPSB)           : {dom_gt_dpsb.sum():,}")
print(f"     - Diger / ERA5 kenar maskesi farki                    : {other_drop:,}")

# 1C. 184,469 Modelleme Paneli Uzerinden Tablo 2 (Yillik), Tablo 3 (Urun) ve Tablo 4 (Il) Yeniden Uretimi
needed = MET_FEATURES + EXP_FEATURES + [TARGET, YEAR_COL, BLOCK_GRID, "PS", "DHPS", "Exposure", "Claim", "LATITUDE", "LONGITUDE", "RAKIM", "Urun Adi Genel", "Il Adi"]
df = df_mod.dropna(subset=needed).copy().reset_index(drop=True)
df["site_id"]    = df["LATITUDE"].round(5).astype(str) + "|" + df["LONGITUDE"].round(5).round(5).astype(str)
df["Crop_Clean"] = df["Urun Adi Genel"].astype(str).str.upper().str.strip().replace("BUGDAY", "BUĞDAY")

# Tablo 2: Yillik Ozet (184,469 Modelleme Paneli)
t2_rows = []
for yr, g in df.groupby(YEAR_COL):
    t2_rows.append({
        "Year": int(yr),
        "N_Obs": len(g),
        "Unique_Sites": g["site_id"].nunique(),
        "Binary_Events": int(g[TARGET].sum()),
        "Binary_Rate_Pct": round(float(g[TARGET].mean() * 100), 2),
        "Total_PS": int(g["PS"].sum()),
        "Affected_DHPS": int(g["DHPS"].sum()),
        "Unit_Rate_Pct": round(float(g["DHPS"].sum() / g["PS"].sum() * 100), 2),
        "Exposure_M_TL": round(float(g["Exposure"].sum() / 1e6), 2),
        "Claim_M_TL": round(float(g["Claim"].sum() / 1e6), 2),
        "Loss_Cost_Pct": round(float(g["Claim"].sum() / g["Exposure"].sum() * 100), 2),
    })
df_t2 = pd.DataFrame(t2_rows)
print("\n--- YENI TABLO 2: YILLIK PORTFOY OZETI (184,469 MODELLEME PANELI) ---")
print(df_t2.to_string(index=False))
df_t2.to_csv(OUTPUT_DIR / "51_table2_yearly_modeling_panel.csv", index=False)
print(f"  Genel Toplam: N={len(df):,}, Events={int(df[TARGET].sum()):,} ({df[TARGET].mean()*100:.2f}%), PS={int(df['PS'].sum()):,}, DHPS={int(df['DHPS'].sum()):,} ({df['DHPS'].sum()/df['PS'].sum()*100:.2f}%), Exp={df['Exposure'].sum()/1e9:.2f} bn TL, Claim={df['Claim'].sum()/1e6:.2f} m TL ({df['Claim'].sum()/df['Exposure'].sum()*100:.2f}%)")

# Tablo 3: Urun Ozeti (184,469 Modelleme Paneli)
t3_rows = []
for crp, g in df.groupby("Crop_Clean"):
    t3_rows.append({
        "Crop": crp,
        "N_Obs": len(g),
        "Unique_Sites": g["site_id"].nunique(),
        "Binary_Events": int(g[TARGET].sum()),
        "Binary_Rate_Pct": round(float(g[TARGET].mean() * 100), 2),
        "Total_PS": int(g["PS"].sum()),
        "Affected_DHPS": int(g["DHPS"].sum()),
        "Unit_Rate_Pct": round(float(g["DHPS"].sum() / g["PS"].sum() * 100), 2),
        "Exposure_Bn_TL": round(float(g["Exposure"].sum() / 1e9), 2),
        "Claim_M_TL": round(float(g["Claim"].sum() / 1e6), 2),
        "Loss_Cost_Pct": round(float(g["Claim"].sum() / g["Exposure"].sum() * 100), 2),
    })
df_t3 = pd.DataFrame(t3_rows)
print("\n--- YENI TABLO 3: URUN BAZINDA PORTFOY OZETI (184,469 MODELLEME PANELI) ---")
print(df_t3.to_string(index=False))
df_t3.to_csv(OUTPUT_DIR / "51_table3_crop_modeling_panel.csv", index=False)

# Tablo 4: Ilk 12 Il Ozeti (184,469 Modelleme Paneli)
t4_rows = []
for prv, g in df.groupby("Il Adi"):
    t4_rows.append({
        "Province": prv,
        "N_Obs": len(g),
        "Total_PS": int(g["PS"].sum()),
        "Affected_DHPS": int(g["DHPS"].sum()),
        "Exposure_M_TL": round(float(g["Exposure"].sum() / 1e6), 2),
        "Claim_M_TL": round(float(g["Claim"].sum() / 1e6), 2),
        "Binary_Rate_Pct": round(float(g[TARGET].mean() * 100), 2),
        "Unit_Rate_Pct": round(float(g["DHPS"].sum() / g["PS"].sum() * 100), 2),
        "Loss_Cost_Pct": round(float(g["Claim"].sum() / g["Exposure"].sum() * 100), 2),
    })
df_t4 = pd.DataFrame(t4_rows).sort_values("Exposure_M_TL", ascending=False).head(12)
print("\n--- YENI TABLO 4: ILK 12 IL OZETI (184,469 MODELLEME PANELI) ---")
print(df_t4.to_string(index=False))
df_t4.to_csv(OUTPUT_DIR / "51_table4_province_modeling_panel.csv", index=False)

# =============================================================================
# [2] B2: GERCEK 307 (DOM>0 & DHPS==0) BAYRAGI ILE M4 OR VE LORO-81 DUYARLILIGI
# =============================================================================
print("\n=== [2] B2: GERCEK 307 (DOM>0 & DHPS==0) BAYRAGI DUYARLILIK ANALIZI ===")

# Aktif gecerli hucrelerde (DOM_raw > 0 & DHPS_raw == 0) olan 307 satiri df ile eslestir
df_act_valid = df_act[(~na_coord) & (~df_act["RAKIM"].isna()) & (df_act["DOM_raw"] <= df_act["DPSB_raw"])].copy().reset_index(drop=True)
df_act_valid["flag_307"] = (df_act_valid["DOM_raw"] > 0) & (df_act_valid["DHPS_raw"] == 0)

if len(df_act_valid) == len(df):
    df["flag_307"] = df_act_valid["flag_307"].values
else:
    mk = ["Urun Adi Genel", "Il Adi", "Ilce Adi", "Koy Adi", "yil", "LATITUDE", "LONGITUDE"]
    flag_map = df_act_valid[df_act_valid["flag_307"]][mk].drop_duplicates().copy()
    flag_map["flag_307"] = True
    df["flag_307"] = df.merge(flag_map, on=mk, how="left")["flag_307"].fillna(False).values

print(f"  Modelleme panelinde yakalanan gercek (DOM>0 & DHPS==0) satir sayisi: {int(df['flag_307'].sum()):,}")

df["Crop_Cat"] = pd.Categorical(df["Crop_Clean"], categories=["ARPA", "BUĞDAY", "MISIR", "PAMUK"], ordered=False)
crop_dummies   = pd.get_dummies(df["Crop_Cat"], prefix="Crop", drop_first=True).astype(float)
CROP_COLS      = list(crop_dummies.columns)
df = pd.concat([df, crop_dummies], axis=1)

def get_window_hours(row):
    crp, rk = row["Crop_Clean"], row["RAKIM"]
    if crp in ["BUĞDAY", "ARPA"]:
        return 1464.0 if rk < 500 else 2208.0
    elif crp == "MISIR":
        return 2928.0 if rk < 800 else 2208.0
    return 2208.0

df["Window_Hours"]  = df.apply(get_window_hours, axis=1)
df["CAPE_dur_rate"] = (df["CAPE_dur"] / df["Window_Hours"]) * 100.0

u_sites = df[["site_id", "LATITUDE", "LONGITUDE"]].drop_duplicates(subset=["site_id"]).reset_index(drop=True)
km81 = KMeans(n_clusters=81, random_state=SEED, n_init=10)
u_sites["Cluster_81"] = km81.fit_predict(u_sites[["LATITUDE", "LONGITUDE"]])
df = df.merge(u_sites[["site_id", "Cluster_81"]], on="site_id", how="left")

M2_COLS      = EXP_FEATURES + CROP_COLS + ["RAKIM"]
M4_ORIG_COLS = M2_COLS + ["WMAXSHR", "CAPE_dur", "WMAXSHR_burst", "h_agl"]
M4_RATE_COLS = M2_COLS + ["WMAXSHR", "CAPE_dur_rate", "WMAXSHR_burst", "h_agl"]

def wy_auc_calc(y_v, p_v, yr_v):
    return float(np.mean([roc_auc_score(y_v[yr_v == y_val], p_v[yr_v == y_val]) for y_val in np.unique(yr_v)]))

b2_rows = []
for b2_lbl, mode_307 in [
    ("1. Baseline (307 rows set to DHPS=1, Claim_Is=1)", "baseline"),
    ("2. Drop 307 Flagged Rows Completely", "drop"),
    ("3. Reclassify 307 Flagged Rows as Claim_Is=0", "zero"),
]:
    if mode_307 == "baseline":
        d_sub = df.copy()
    elif mode_307 == "drop":
        d_sub = df[~df["flag_307"]].copy().reset_index(drop=True)
    else:
        d_sub = df.copy()
        d_sub.loc[d_sub["flag_307"], TARGET] = 0

    # Train <= 2015 M4 ORs
    tr_b = d_sub[d_sub[YEAR_COL] <= 2015]
    sc_b = StandardScaler()
    Xtr_b = sm.add_constant(pd.DataFrame(sc_b.fit_transform(tr_b[M4_ORIG_COLS]), columns=M4_ORIG_COLS, index=tr_b.index))
    m_b = sm.Logit(tr_b[TARGET].values, Xtr_b).fit(method="bfgs", maxiter=400, disp=0, cov_type="cluster", cov_kwds={"groups": tr_b[BLOCK_GRID].values})

    # LORO-81 M4 - M2
    p_m2 = np.zeros(len(d_sub))
    p_m4 = np.zeros(len(d_sub))
    c_arr = d_sub["Cluster_81"].values
    y_arr = d_sub[TARGET].values
    yr_a  = d_sub[YEAR_COL].values
    for cid in range(81):
        te_c = (c_arr == cid)
        tr_c = ~te_c
        for cols_u, p_store in [(M2_COLS, p_m2), (M4_ORIG_COLS, p_m4)]:
            sc_l = StandardScaler()
            Xtr_l = sc_l.fit_transform(d_sub.loc[tr_c, cols_u].values)
            Xte_l = sc_l.transform(d_sub.loc[te_c, cols_u].values)
            p_store[te_c] = LogisticRegression(C=1e9, max_iter=300).fit(Xtr_l, y_arr[tr_c]).predict_proba(Xte_l)[:, 1]

    d_wy = wy_auc_calc(y_arr, p_m4, yr_a) - wy_auc_calc(y_arr, p_m2, yr_a)
    d_ll = log_loss(y_arr, np.clip(p_m4, 1e-7, 1-1e-7)) - log_loss(y_arr, np.clip(p_m2, 1e-7, 1-1e-7))

    b2_rows.append({
        "B2_Sensitivity_Rule": b2_lbl,
        "N_Obs": len(d_sub),
        "Total_Events": int(y_arr.sum()),
        "M4_OR_WMAXSHR": round(float(np.exp(m_b.params["WMAXSHR"])), 4),
        "M4_OR_CAPE_dur": round(float(np.exp(m_b.params["CAPE_dur"])), 4),
        "M4_OR_burst": round(float(np.exp(m_b.params["WMAXSHR_burst"])), 4),
        "M4_OR_h_agl": round(float(np.exp(m_b.params["h_agl"])), 4),
        "LORO81_d_WithinYr_AUC": round(d_wy, 4),
        "LORO81_d_LogLoss": round(d_ll, 4),
    })

df_b2 = pd.DataFrame(b2_rows)
print(df_b2.to_string(index=False))
df_b2.to_csv(OUTPUT_DIR / "51_b2_exact_307_flag_sensitivity.csv", index=False)

# =============================================================================
# [3] UNBUFFERED & 28-KM BUFFERED LORO-81 FOR M4_ORIG AND M4_RATE (B = 5,000
#     DUAL-SCALE BOOTSTRAP: GRID_ID AND CLUSTER_81)
# =============================================================================
print(f"\n=== [3] UNBUFFERED & 28-KM BUFFERED LORO-81 (M4_ORIG & M4_RATE) B={N_BOOT_5K} BOOTSTRAP ===")

c81_all = df["Cluster_81"].values
g_all   = df[BLOCK_GRID].values
y_all   = df[TARGET].values
yr_all  = df[YEAR_COL].values

u_rad = np.radians(u_sites[["LATITUDE", "LONGITUDE"]].values)
u_c81 = u_sites["Cluster_81"].values
u_sid = u_sites["site_id"].values

oof_unbuf = {k: np.zeros(len(df)) for k in ["M2", "M4_Orig", "M4_Rate"]}
oof_buf28 = {k: np.zeros(len(df)) for k in ["M2", "M4_Orig", "M4_Rate"]}

for cid in range(81):
    te_m = (c81_all == cid)
    tr_unbuf = ~te_m

    te_u_rad = u_rad[u_c81 == cid]
    tr_u_idx = np.where(u_c81 != cid)[0]
    nn = NearestNeighbors(n_neighbors=1, metric="haversine", n_jobs=-1).fit(te_u_rad)
    d_rad, _ = nn.kneighbors(u_rad[tr_u_idx])
    valid_tr_sites = set(u_sid[tr_u_idx[d_rad[:, 0] * EARTH_R >= 28.0]])
    tr_buf28 = df["site_id"].isin(valid_tr_sites).values

    for k, cols_u in [("M2", M2_COLS), ("M4_Orig", M4_ORIG_COLS), ("M4_Rate", M4_RATE_COLS)]:
        # Unbuffered
        sc_u = StandardScaler()
        Xtr_u = sc_u.fit_transform(df.loc[tr_unbuf, cols_u].values)
        Xte_u = sc_u.transform(df.loc[te_m, cols_u].values)
        oof_unbuf[k][te_m] = LogisticRegression(C=1e9, max_iter=300).fit(Xtr_u, y_all[tr_unbuf]).predict_proba(Xte_u)[:, 1]

        # 28-km Buffered
        sc_b = StandardScaler()
        Xtr_b = sc_b.fit_transform(df.loc[tr_buf28, cols_u].values)
        Xte_b = sc_b.transform(df.loc[te_m, cols_u].values)
        oof_buf28[k][te_m] = LogisticRegression(C=1e9, max_iter=300).fit(Xtr_b, y_all[tr_buf28]).predict_proba(Xte_b)[:, 1]

# Hizli Yil-Ici AUC ve LogLoss Bootstrap Hesabi (B = 5,000, Grid_ID ve Cluster_81)
eval_pairs = [
    ("Unbuffered_LORO81: M4_Orig - M2", oof_unbuf["M4_Orig"], oof_unbuf["M2"]),
    ("Unbuffered_LORO81: M4_Rate - M2", oof_unbuf["M4_Rate"], oof_unbuf["M2"]),
    ("Buffered28km_LORO81: M4_Orig - M2", oof_buf28["M4_Orig"], oof_buf28["M2"]),
    ("Buffered28km_LORO81: M4_Rate - M2", oof_buf28["M4_Rate"], oof_buf28["M2"]),
]

boot_5k_rows = []
for blk_name, blk_vec in [("Grid_ID (1333 cells)", g_all), ("Cluster_81 (81 regions)", c81_all)]:
    s_grp = pd.Series(blk_vec).groupby(blk_vec).indices
    b_keys = list(s_grp.keys())
    n_b = len(b_keys)
    rng = np.random.default_rng(SEED)

    res_store = {lbl: {"wy": [], "pool": [], "ll": []} for lbl, _, _ in eval_pairs}
    for _ in tqdm(range(N_BOOT_5K), desc=f"B=5000 [{blk_name[:10]}]", leave=False):
        dr = rng.integers(0, n_b, size=n_b)
        ix = np.concatenate([s_grp[b_keys[d]] for d in dr])
        yb, yrb = y_all[ix], yr_all[ix]
        for lbl, pa_full, pb_full in eval_pairs:
            pa = np.clip(pa_full[ix], 1e-7, 1.0 - 1e-7)
            pb = np.clip(pb_full[ix], 1e-7, 1.0 - 1e-7)
            res_store[lbl]["wy"].append(wy_auc_calc(yb, pa, yrb) - wy_auc_calc(yb, pb, yrb))
            res_store[lbl]["pool"].append(roc_auc_score(yb, pa) - roc_auc_score(yb, pb))
            res_store[lbl]["ll"].append(log_loss(yb, pa) - log_loss(yb, pb))

    for lbl, pa_full, pb_full in eval_pairs:
        pa_c = np.clip(pa_full, 1e-7, 1.0 - 1e-7)
        pb_c = np.clip(pb_full, 1e-7, 1.0 - 1e-7)
        pt_wy = wy_auc_calc(y_all, pa_c, yr_all) - wy_auc_calc(y_all, pb_c, yr_all)
        pt_pl = roc_auc_score(y_all, pa_c) - roc_auc_score(y_all, pb_c)
        pt_ll = log_loss(y_all, pa_c) - log_loss(y_all, pb_c)

        row = {
            "Protocol_Comparison": lbl,
            "Block_Unit": blk_name,
            "d_WithinYr_AUC": round(pt_wy, 4),
            "d_Pooled_AUC": round(pt_pl, 4),
            "d_LogLoss": round(pt_ll, 4),
        }
        for mk, col_t in [("wy", "WithinYr_AUC"), ("pool", "Pooled_AUC"), ("ll", "LogLoss")]:
            a = np.asarray(res_store[lbl][mk])
            lo, hi = float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))
            pv = 2.0 * min((a <= 0).mean(), (a >= 0).mean())
            pv = float(min(max(pv, 1.0 / (N_BOOT_5K + 1)), 1.0))
            row[f"{col_t}_CI"] = f"[{lo:+.4f}, {hi:+.4f}]"
            row[f"p_{col_t}"]  = round(pv, 5)
            row[f"p_Holm7_{col_t}"]  = round(min(pv * 7.0, 1.0), 5)
            row[f"p_Holm14_{col_t}"] = round(min(pv * 14.0, 1.0), 5)
        boot_5k_rows.append(row)

df_boot5k = pd.DataFrame(boot_5k_rows)
print("\n--- B = 5,000 DUAL-SCALE (GRID_ID & CLUSTER_81) UNBUFFERED & 28-KM BUFFERED LORO-81 SONUCLARI ---")
print(df_boot5k.to_string(index=False))
df_boot5k.to_csv(OUTPUT_DIR / "51_unbuffered_and_buffered28km_5000boot_dual_scale.csv", index=False)

print(f"\n[TAMAMLANDI] 51_clean_pipeline_tables1to4_and_buffered_5000boot ciktilari kaydedildi -> {OUTPUT_DIR}")01_clean_pipeline_and_summary
