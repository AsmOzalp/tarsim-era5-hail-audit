# -*- coding: utf-8 -*-
"""
04_shap_refitted_bootstrap.py
==================================
Replaces 20_shap_changepoint_BOOTSTRAP_FIX.py.

What was wrong with the old bootstrap
  * It resampled the 3,000 fixed SHAP points of ONE fixed model. It therefore
    measured only the noise of the breakpoint search, not model uncertainty,
    and ignored spatial dependence (hence the very narrow WMAXSHR interval).
  * The two-segment fit had separate intercepts (discontinuous), so it can
    "find" a jump instead of a change of slope.
  * A single breakpoint was assumed; h_agl gave [1648, 3805] m, i.e. the
    bootstrap distribution is almost surely multi-modal.

What this script does
  1. Each bootstrap iteration RESAMPLES TRAINING Grid_ID BLOCKS (spatial block
     bootstrap, all years of a cell stay together), REFITS LightGBM, recomputes
     SHAP contributions and RESAMPLES evaluation Grid_ID blocks.
  2. Continuous broken-stick fit  s = a + b*x + c*max(x - tau, 0), tau on a grid
     of quantiles of x (5-95 %). Reports the slope before / after tau.
  3. Reports percentile CI AND a mode check (histogram peaks, share of draws in
     each mode). If the distribution is multi-modal, do NOT report a single
     "threshold": describe the dependence curve instead (it is plotted with a
     bootstrap band).
  4. SHAP values come from LightGBM's native pred_contrib (log-odds scale,
     identical to TreeSHAP), so the `shap` package is not needed.

RUNTIME: controlled by N_BOOT and LEARNING_RATE. Start with N_BOOT = 50 to check,
then raise to 500-1000 and let it run in the background.

Outputs (OUTPUT_DIR):
  shap_transition_summary.csv, shap_transition_boot_<feature>.csv,
  shap_transition_<feature>.png
"""

from pathlib import Path
import time
import warnings

import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

warnings.filterwarnings("ignore")

# =============================================================================
# [1] CONFIG
# =============================================================================
FILE_PATH = Path(r"C:\Users\m-a-o\Desktop\TarsimData\V2\model_ready_data_spatial_final_v1.csv")
OUTPUT_DIR = FILE_PATH.parent / "model_outputs"

TARGET = "Claim_Is"
YEAR = "yil"
BLOCK = "Grid_ID"
FEATURES = ["WMAXSHR", "CAPE_dur", "WMAXSHR_burst", "h_agl", "log_coverage"]
TARGET_FEATURES = ["WMAXSHR", "h_agl"]      # features whose transition points are reported

N_BOOT = 200            # raise to 500-1000 for the final run
EVAL_N = 5000           # rows of the 2017 test year used for SHAP evaluation
LEARNING_RATE = 0.03    # larger than 0.01 to keep refits affordable
MAX_TREES = 2000
N_TAU_CANDIDATES = 60   # candidate breakpoints (quantile grid)
N_BINS = 20             # bins for the dependence-curve band
MIN_BOOT_FOR_PLOT = 20
SEED = 42


# =============================================================================
# [2] HELPERS
# =============================================================================
def lgb_params():
    return dict(n_estimators=MAX_TREES, learning_rate=LEARNING_RATE, max_depth=6,
                num_leaves=31, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                random_state=SEED, n_jobs=-1, verbose=-1)


def fit_lgb(X_tr, y_tr, X_va, y_va):
    m = lgb.LGBMClassifier(**lgb_params())
    m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], eval_metric="binary_logloss",
          callbacks=[lgb.early_stopping(100, verbose=False)])
    return m


def shap_contrib(model, X):
    """TreeSHAP values on the log-odds scale; last column (bias) dropped."""
    c = model.predict(X, pred_contrib=True)
    c = np.asarray(c)
    return c[:, :-1]


def broken_stick(x, s, n_cand=N_TAU_CANDIDATES):
    """
    Continuous piecewise-linear fit with ONE breakpoint:
        s = a + b*x + c*max(x - tau, 0)
    tau is searched on quantiles of x (5-95%). Returns tau, slope_left, slope_right, rss.
    """
    x = np.asarray(x, float)
    s = np.asarray(s, float)
    qs = np.unique(np.quantile(x, np.linspace(0.05, 0.95, n_cand)))
    best = (np.inf, None, None, None)
    for tau in qs:
        A = np.column_stack([np.ones_like(x), x, np.maximum(x - tau, 0.0)])
        coef, *_ = np.linalg.lstsq(A, s, rcond=None)
        resid = s - A @ coef
        rss = float(resid @ resid)
        if rss < best[0]:
            best = (rss, tau, coef[1], coef[1] + coef[2])
    rss, tau, sl, sr = best
    return tau, sl, sr, rss


def binned_curve(x, s, edges):
    idx = np.clip(np.digitize(x, edges[1:-1]), 0, len(edges) - 2)
    out = np.full(len(edges) - 1, np.nan)
    for b in range(len(edges) - 1):
        m = idx == b
        if m.any():
            out[b] = s[m].mean()
    return out


def block_index(groups_indices, group_keys, rng, size=None):
    """Concatenate row indices of Grid_ID blocks drawn with replacement."""
    n_g = len(group_keys)
    draw = rng.integers(0, n_g, size=n_g if size is None else size)
    return np.concatenate([groups_indices[group_keys[g]] for g in draw])


def mode_report(taus, name, unit=""):
    taus = np.asarray(taus, float)
    hist, edges = np.histogram(taus, bins=40)
    peaks, props = find_peaks(hist, prominence=max(2, 0.05 * len(taus)))
    centers = 0.5 * (edges[:-1] + edges[1:])
    info = {"n_modes": int(len(peaks)),
            "mode_centers": [round(float(centers[p]), 1) for p in peaks]}
    if len(peaks) >= 2:
        top2 = peaks[np.argsort(hist[peaks])[::-1][:2]]
        c1, c2 = sorted(centers[top2])
        mid = 0.5 * (c1 + c2)
        info["share_below_mid"] = round(float((taus < mid).mean()), 3)
        info["share_above_mid"] = round(float((taus >= mid).mean()), 3)
    print(f"  [{name}] histogram peaks: {info}")
    return info


# =============================================================================
# [3] MAIN
# =============================================================================
def main(df=None):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if df is None:
        df = pd.read_csv(FILE_PATH, low_memory=False)
    df = df.dropna(subset=FEATURES + [TARGET, YEAR, BLOCK]).reset_index(drop=True)

    tr_m, va_m, te_m = (df[YEAR] <= 2015), (df[YEAR] == 2016), (df[YEAR] == 2017)
    X_tr, y_tr = df.loc[tr_m, FEATURES].reset_index(drop=True), df.loc[tr_m, TARGET].reset_index(drop=True)
    X_va, y_va = df.loc[va_m, FEATURES], df.loc[va_m, TARGET]
    g_tr = df.loc[tr_m, BLOCK].reset_index(drop=True)

    pool = df.loc[te_m, FEATURES + [BLOCK]]
    pool = pool.sample(n=min(EVAL_N, len(pool)), random_state=SEED).reset_index(drop=True)
    X_ev = pool[FEATURES]
    g_ev = pool[BLOCK]

    tr_groups = g_tr.groupby(g_tr).indices
    tr_keys = list(tr_groups.keys())
    ev_groups = g_ev.groupby(g_ev).indices
    ev_keys = list(ev_groups.keys())
    print(f"Train blocks: {len(tr_keys)} | eval blocks: {len(ev_keys)} | eval rows: {len(X_ev)}")

    # ---- point estimate (original model, no resampling) -----------------------
    print("\n--- Point estimate: fitting original model ---")
    m0 = fit_lgb(X_tr, y_tr, X_va, y_va)
    print(f"best_iteration = {m0.best_iteration_}")
    S0 = shap_contrib(m0, X_ev)

    point, edges, curves0 = {}, {}, {}
    for f in TARGET_FEATURES:
        j = FEATURES.index(f)
        x = X_ev[f].values
        tau, sl, sr, _ = broken_stick(x, S0[:, j])
        point[f] = (tau, sl, sr)
        edges[f] = np.unique(np.quantile(x, np.linspace(0, 1, N_BINS + 1)))
        curves0[f] = binned_curve(x, S0[:, j], edges[f])
        print(f"  [{f}] point tau = {tau:.2f}  slope_left = {sl:.4g}  slope_right = {sr:.4g}")

    # ---- bootstrap: resample blocks, refit, recompute SHAP, resample eval blocks
    print(f"\n--- Spatial block bootstrap with refit: {N_BOOT} iterations ---")
    rng = np.random.default_rng(SEED)
    boot_tau = {f: [] for f in TARGET_FEATURES}
    boot_slopes = {f: [] for f in TARGET_FEATURES}
    boot_curves = {f: [] for f in TARGET_FEATURES}
    t0 = time.time()
    for b in range(N_BOOT):
        idx = block_index(tr_groups, tr_keys, rng)
        mb = fit_lgb(X_tr.iloc[idx], y_tr.iloc[idx], X_va, y_va)
        Sb = shap_contrib(mb, X_ev)
        eidx = block_index(ev_groups, ev_keys, rng)
        for f in TARGET_FEATURES:
            j = FEATURES.index(f)
            x, s = X_ev[f].values[eidx], Sb[eidx, j]
            tau, sl, sr, _ = broken_stick(x, s)
            boot_tau[f].append(tau)
            boot_slopes[f].append((sl, sr))
            boot_curves[f].append(binned_curve(X_ev[f].values, Sb[:, j], edges[f]))
        if (b + 1) % 10 == 0 or b == 0:
            el = time.time() - t0
            print(f"  iter {b + 1}/{N_BOOT} | elapsed {el / 60:.1f} min | "
                  f"ETA {el / (b + 1) * (N_BOOT - b - 1) / 60:.1f} min")

    # ---- summaries -------------------------------------------------------------
    rows = []
    for f in TARGET_FEATURES:
        t = np.array(boot_tau[f])
        sl = np.array(boot_slopes[f])
        lo, med, hi = np.percentile(t, [2.5, 50, 97.5])
        modes = mode_report(t, f)
        rows.append({
            "feature": f, "point_estimate": round(point[f][0], 2),
            "boot_median": round(med, 2), "ci_lower": round(lo, 2), "ci_upper": round(hi, 2),
            "ci_width": round(hi - lo, 2),
            "share_slope_right_lt_left": round(float((sl[:, 1] < sl[:, 0]).mean()), 3),
            "n_modes": modes["n_modes"], "mode_centers": modes["mode_centers"],
            "n_boot": N_BOOT})
        pd.DataFrame({"tau": t, "slope_left": sl[:, 0], "slope_right": sl[:, 1]}) \
            .to_csv(OUTPUT_DIR / f"shap_transition_boot_{f}.csv", index=False)

        # figure: dependence curve with bootstrap band + histogram of tau
        centers = 0.5 * (edges[f][:-1] + edges[f][1:])
        C = np.array(boot_curves[f])
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        if len(C) >= MIN_BOOT_FOR_PLOT:
            ax[0].fill_between(centers, np.nanpercentile(C, 2.5, axis=0),
                               np.nanpercentile(C, 97.5, axis=0), alpha=0.25,
                               label="95% block-bootstrap band")
        ax[0].plot(centers, curves0[f], "o-", label="original model")
        ax[0].axvline(point[f][0], ls="--", c="k", lw=1, label="point estimate of tau")
        ax[0].set_xlabel(f)
        ax[0].set_ylabel("mean SHAP (log-odds)")
        ax[0].legend(fontsize=8)
        ax[1].hist(t, bins=40)
        ax[1].axvline(lo, c="r", ls=":")
        ax[1].axvline(hi, c="r", ls=":")
        ax[1].set_xlabel(f"bootstrap tau ({f})")
        ax[1].set_ylabel("count")
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / f"shap_transition_{f}.png", dpi=200)
        plt.close(fig)

    res = pd.DataFrame(rows)
    res.to_csv(OUTPUT_DIR / "shap_transition_summary.csv", index=False)
    print("\n" + "=" * 78)
    print("MODEL-DERIVED TRANSITION POINTS (refit + spatial block bootstrap)")
    print("=" * 78)
    print(res.to_string(index=False))
    print("\nHow to read: report a transition point ONLY if n_modes == 1 and the CI is "
          "physically narrow. If n_modes >= 2 (expected for h_agl), describe the SHAP "
          "dependence curve and its band instead of a single threshold.")
    return res


if __name__ == "__main__":
    main()
