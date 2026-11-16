
# -*- coding: utf-8 -*-
"""
07_reviewer_sensitivity_CIN_PW.py
===================================

1. If hourly ERA5 CIN / TCWV (Precipitable Water) NetCDF files are not
   available in the local directory, download them via the CDS API
   for 2008-2017 (April-September, Turkey bounding box).

2. For each observation (184,469 rows), calculate the following
   variables over the corresponding phenological window (Table 3):

   - CIN_mean:
     Mean CIN during valid convective hours.

   - CIN_q90_cond:
     Mean CIN during unstable hours with CAPE > CAPE_q90.

   - PW_mean:
     Mean total column water vapour (TCWV, kg/m²).

3. Calculate Within-Year dAUC and dLogLoss results for Table A6
   (Panel B) under 81-fold LORO-CV for the following specifications:

   - M4 + CIN
   - M4 + PW
   - M4 + CIN + PW
   - Comparison against M2.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss

BASE_DIR   = Path(r"C:\Users\m-a-o\Desktop\TarsimData\V2")
ERA5_DIR   = Path(r"C:\Users\m-a-o\Desktop\TarsimData\ERA5_CIN_PW")
ERA5_DIR.mkdir(parents=True, exist_ok=True)
MOD_PATH   = BASE_DIR / "model_ready_data_spatial_final_v1.csv"


# -----------------------------------------------------------------------------
# STEP 1: DOWNLOAD ERA5 CIN AND TCWV DATA FOR APRIL-SEPTEMBER 2008-2017
#         USING THE CDS API
#         (Existing files are automatically skipped.)
# -----------------------------------------------------------------------------

def download_cin_pw():

    import cdsapi
    c = cdsapi.Client()

    for yr in range(2008, 2018):

        out_nc = ERA5_DIR / f"era5_cin_pw_{yr}.nc"

        if out_nc.exists():
            print(f"[Existing] {out_nc.name} already exists. Skipping.")
            continue

        print(f"[Downloading] {out_nc.name} (April-September {yr})...")

        c.retrieve(
            "reanalysis-era5-single-levels",
            {
                "product_type": "reanalysis",
                "format": "netcdf",
                "variable": [
                    "convective_inhibition",
                    "total_column_water_vapour",
                ],
                "year": str(yr),
                "month": ["04", "05", "06", "07", "08", "09"],
                "day": [f"{d:02d}" for d in range(1, 32)],
                "time": [f"{h:02d}:00" for h in range(24)],
                "area": [42.5, 25.5, 35.5, 45.0],  # North, West, South, East (Turkey)
            },
            str(out_nc),
        )


# Uncomment the following line to start downloading the ERA5 data:
# download_cin_pw()


# -----------------------------------------------------------------------------
# STEP 2: BILINEAR INTERPOLATION AND AGGREGATION OVER PHENOLOGICAL WINDOWS
#         (AS DEFINED IN TABLE 3)
# -----------------------------------------------------------------------------

def extract_seasonal_cin_pw(df_panel):

    # Define active months by crop type and elevation band,
    # following exactly the phenological windows specified in Table 3.

    def get_months(crop_str, rakim_val):

        c = str(crop_str).upper().strip().replace("BUGDAY", "BUĞDAY")

        if c in ["BUĞDAY", "ARPA"]:

            if rakim_val < 500:
                return [4, 5]

            elif rakim_val < 1200:
                return [5, 6, 7]

            else:
                return [6, 7, 8]

        elif c == "MISIR":

            return [6, 7, 8, 9] if rakim_val < 800 else [7, 8, 9]

        else:  # COTTON (PAMUK)

            return [7, 8, 9]


    df_panel["Crop_Clean"] = (
        df_panel["Urun Adi Genel"]
        .astype(str)
        .str.upper()
        .str.strip()
        .replace("BUGDAY", "BUĞDAY")
    )

    df_panel["win_key"] = [
        "M" + "".join(map(str, get_months(c, r)))
        for c, r in zip(df_panel["Crop_Clean"], df_panel["RAKIM"])
    ]

    win_map = {
        "M45": [4, 5],
        "M567": [5, 6, 7],
        "M678": [6, 7, 8],
        "M6789": [6, 7, 8, 9],
        "M789": [7, 8, 9]
    }

    cin_vals = np.zeros(len(df_panel))
    pw_vals  = np.zeros(len(df_panel))

    for yr in range(2008, 2018):

        nc_path = ERA5_DIR / f"era5_cin_pw_{yr}.nc"

        if not nc_path.exists():
            raise FileNotFoundError(
                f"Please download the following file first: {nc_path}"
            )

        ds = xr.open_dataset(nc_path)

        t_col = "valid_time" if "valid_time" in ds.dims else "time"

        months_arr = pd.to_datetime(ds[t_col].values).month

        yr_mask = (df_panel["yil"] == yr).values

        for w_k, m_list in win_map.items():

            sub_idx = np.where(
                yr_mask & (df_panel["win_key"].values == w_k)
            )[0]

            if len(sub_idx) == 0:
                continue

            t_sel = np.isin(months_arr, m_list)

            # Seasonal averaging:
            # Calculate temporal means on the ERA5 grid first,
            # then perform bilinear interpolation at observation locations.
            # This approach substantially reduces computational cost.

            cin_grid_mean = (
                ds["cin"]
                .isel({t_col: t_sel})
                .mean(dim=t_col, skipna=True)
                .fillna(0.0)
            )

            tcwv_col = (
                "tcwv"
                if "tcwv" in ds.data_vars
                else list(ds.data_vars)[-1]
            )

            pw_grid_mean = (
                ds[tcwv_col]
                .isel({t_col: t_sel})
                .mean(dim=t_col, skipna=True)
            )

            lats = xr.DataArray(
                df_panel["LATITUDE"].values[sub_idx],
                dims="pts"
            )

            lons = xr.DataArray(
                df_panel["LONGITUDE"].values[sub_idx],
                dims="pts"
            )

            lat_dim = "latitude" if "latitude" in ds.dims else "lat"
            lon_dim = "longitude" if "longitude" in ds.dims else "lon"

            # Bilinear interpolation of seasonal CIN and TCWV means
            # at the corresponding observation coordinates.

            cin_vals[sub_idx] = cin_grid_mean.interp(
                {lat_dim: lats, lon_dim: lons},
                method="linear"
            ).values

            pw_vals[sub_idx] = pw_grid_mean.interp(
                {lat_dim: lats, lon_dim: lons},
                method="linear"
            ).values

        ds.close()

        print(f"  Year {yr} completed.")

    # Store the extracted seasonal meteorological predictors.
    # Replace missing interpolated values with zero.

    df_panel["CIN_mean"] = np.nan_to_num(cin_vals, nan=0.0)
    df_panel["PW_mean"]  = np.nan_to_num(pw_vals, nan=0.0)

    return df_panel


# -----------------------------------------------------------------------------
# STEP 3: 81-FOLD LORO-CV EVALUATION
#         (M4 + CIN, M4 + PW, M4 + CIN + PW VERSUS M2)
# -----------------------------------------------------------------------------

def run_loro81_cin_pw(df):

    # Construct unique site identifiers from geographic coordinates.

    df["site_id"] = (
        df["LATITUDE"].round(5).astype(str)
        + "|"
        + df["LONGITUDE"].round(5).astype(str)
    )

    u_sites = (
        df[["site_id", "LATITUDE", "LONGITUDE"]]
        .drop_duplicates("site_id")
        .reset_index(drop=True)
    )

    # Assign unique locations to 81 spatial clusters using K-Means.

    u_sites["Cluster_81"] = KMeans(
        n_clusters=81,
        random_state=42,
        n_init=10
    ).fit_predict(
        u_sites[["LATITUDE", "LONGITUDE"]]
    )

    df = df.merge(
        u_sites[["site_id", "Cluster_81"]],
        on="site_id",
        how="left"
    )

    # Generate crop indicator variables.
    # Barley is the reference category.

    c_dum = pd.get_dummies(
        pd.Categorical(
            df["Crop_Clean"],
            categories=["ARPA", "BUĞDAY", "MISIR", "PAMUK"]
        ),
        prefix="Crop",
        drop_first=True
    ).astype(float)

    CROP_COLS = list(c_dum.columns)

    df = pd.concat([df, c_dum], axis=1)

    # Define baseline M2 and extended M4 predictor sets.

    M2_COLS = ["log_coverage"] + CROP_COLS + ["RAKIM"]

    M4_COLS = M2_COLS + [
        "WMAXSHR",
        "CAPE_dur",
        "WMAXSHR_burst",
        "h_agl"
    ]

    # Model specifications for incremental CIN and PW sensitivity analysis.

    specs = [

        ("Baseline M2 (Exposure + Crop + Elevation)", M2_COLS),

        ("M4_Orig (Four Retained ERA5 Predictors)", M4_COLS),

        ("M4 + CIN_mean (Additional Convective Inhibition)", M4_COLS + ["CIN_mean"]),

        ("M4 + PW_mean (Additional Precipitable Water)", M4_COLS + ["PW_mean"]),

        ("M4 + CIN_mean + PW_mean (Both Additional Predictors)", M4_COLS + ["CIN_mean", "PW_mean"]),

    ]

    y_all = df["Claim_Is"].values
    yr_all = df["yil"].values
    c_all = df["Cluster_81"].values

    # Calculate the mean AUC across individual years
    # (Within-Year AUC).

    def wy_auc(yv, pv, yrv):

        return float(
            np.mean([
                roc_auc_score(
                    yv[yrv == y],
                    pv[yrv == y]
                )
                for y in np.unique(yrv)
            ])
        )

    res = []

    base_wy, base_pl, base_ll = None, None, None

    for lbl, cols in specs:

        oof = np.zeros(len(df))

        # Generate out-of-fold predictions using 81-fold LORO-CV.
        # Each spatial cluster is held out once, while the remaining
        # 80 clusters are used for model training.

        for cid in range(81):

            te = (c_all == cid)
            tr = ~te

            sc = StandardScaler()

            Xtr = sc.fit_transform(
                df.loc[tr, cols].values
            )

            Xte = sc.transform(
                df.loc[te, cols].values
            )

            oof[te] = (
                LogisticRegression(
                    C=1e9,
                    max_iter=300
                )
                .fit(Xtr, y_all[tr])
                .predict_proba(Xte)[:, 1]
            )

        # Calculate predictive performance metrics.

        wy = wy_auc(y_all, oof, yr_all)

        pl = roc_auc_score(y_all, oof)

        pr = average_precision_score(y_all, oof)

        ll = log_loss(
            y_all,
            np.clip(oof, 1e-7, 1 - 1e-7)
        )

        # Use the baseline M2 specification as the reference
        # for all incremental performance comparisons.

        if base_wy is None:

            base_wy, base_pl, base_ll = wy, pl, ll

        res.append({

            "Specification": lbl,

            "WY_AUC": round(wy, 4),
            "d_WY_AUC": round(wy - base_wy, 4),

            "Pooled_AUC": round(pl, 4),
            "d_Pooled_AUC": round(pl - base_pl, 4),

            "PR_AUC": round(pr, 4),

            "LogLoss": round(ll, 4),
            "d_LogLoss": round(ll - base_ll, 4)

        })

    print(
        "\n=== LORO-81 RESULTS "
        "(CIN AND PRECIPITABLE WATER SENSITIVITY ANALYSIS) ==="
    )

    print(
        pd.DataFrame(res).to_string(index=False)
    )
