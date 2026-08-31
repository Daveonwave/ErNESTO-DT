"""
Diagnostic: sweep the RLS forgetting factor and the ARX sample time `Ts` on
the real PRBS-like ground-truth file, to separate two competing hypotheses
for why ARX/RLS voltage tracking degrades on this data (see bugs 1 and 2 in
/root/.claude/plans/i-am-trying-to-replicated-volcano.md):

  (a) The RLS filter adapts too slowly relative to how fast the PRBS current
      changes sign (forgetting_factor=0.999 gives ~1000-sample memory).
  (b) `data/config/sim_opt_region_shift_validation_rls.yaml` sets
      `arx_rls.Ts: 1` (second), but the ground file
      `data/ground/online_learning/region_shift_validation_new.csv` is
      actually sampled at 0.1 s. Since `theta_to_physical_1RC` computes
      `tau1 = -Ts/log(alpha)` and `C1 = tau1/R1`, a 10x-wrong Ts makes the
      recovered tau1/C1 come out ~10x too large -- and can push C1 outside
      the configured `search_bounds` (5000-50000 F), causing the estimator
      to silently freeze on stale values far more often than intended.

This script does NOT modify any production code. It reuses the exact same
building blocks the runtime estimator (`ernesto.adaptation.arx_rls_estimator
.ARXRLS1RCEstimator`) wraps -- `RLS`, `build_regressor_1RC_at_k`,
`theta_to_physical_1RC` -- but orchestrates them directly so every
intermediate (one-step prediction error, physical validity, bounds
rejection) is directly observable, which the production wrapper does not
expose.

For each (Ts, forgetting_factor, bounds) combination it reports:
  - fraction of samples with a physically-valid theta
  - fraction rejected specifically for being outside search_bounds
  - RMSE of the raw ARX one-step-ahead voltage prediction ("metric A":
    is the linear regression itself tracking the data well?)
  - RMSE of a "shadow" Euler-integrated ECM driven by the live recovered
    (R0, R1, C1, OCV), using the same backward-Euler recursion as
    `ernesto.digital_twin.battery_models.electrical.ecm.FirstOrderThevenin
    .step_current_driven`, at the real per-sample dt ("metric B": if you
    fed these estimated parameters into the actual digital twin, would the
    simulated voltage match?)

Sign convention: `battery.sign_convention: "passive"` in the yaml above, and
`ARXRLS1RCEstimator.update()` unconditionally negates incoming current
(bug 4 in the plan). For this dataset the two happen to agree (passive also
negates current in ecm.py), so `i_used = -i_raw` is applied consistently in
both metric A and metric B below.

Usage:
    python scripts/arx_forgetting_sweep.py [--max-rows N] [--out-dir DIR]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ernesto.adaptation.rls import RLS
from ernesto.adaptation.arx import build_regressor_1RC_at_k
from ernesto.adaptation.ARX_1RC import theta_to_physical_1RC

DEFAULT_CSV = "data/ground/online_learning/region_shift_validation_new.csv"

# Same search_bounds as data/config/sim_opt_region_shift_validation_rls.yaml
PROD_BOUNDS = {
    "r0": {"low": 0.001, "high": 0.01},
    "r1": {"low": 0.001, "high": 0.01},
    "c1": {"low": 5000.0, "high": 50000.0},
}

WARMUP_SAMPLES = 20
P0_SCALE = 1e9


def load_ground(csv_path, max_rows=None):
    df = pd.read_csv(csv_path)
    if max_rows:
        df = df.iloc[:max_rows]
    v = df["Voltage [V]"].to_numpy(dtype=float)
    i_raw = df["Current [A]"].to_numpy(dtype=float)
    t = df["Time"].to_numpy(dtype=float)
    return v, i_raw, t


def inside_bounds(params, bounds):
    if bounds is None:
        return True
    for name, value in params.items():
        if name not in bounds:
            continue
        lo = bounds[name].get("low", -np.inf)
        hi = bounds[name].get("high", np.inf)
        if not (lo <= value <= hi):
            return False
    return True


def run_config(v, i_used, t, Ts, forgetting_factor, bounds,
               warmup=WARMUP_SAMPLES, P0_scale=P0_SCALE):
    """Replays the trace through RLS + theta_to_physical_1RC sample-by-sample."""
    n = len(v)
    rls = RLS(n_params=4, theta0=np.zeros(4), P0_scale=P0_scale,
              forgetting_factor=forgetting_factor)

    arx_pred_err = np.full(n, np.nan)
    r0h = np.full(n, np.nan)
    r1h = np.full(n, np.nan)
    c1h = np.full(n, np.nan)
    ocvh = np.full(n, np.nan)

    n_valid = 0
    n_rejected_bounds = 0
    n_rejected_physical = 0
    last_valid = None

    for k in range(1, n):
        phi = np.array([v[k - 1], i_used[k], i_used[k - 1], 1.0])

        # One-step-ahead prediction with theta BEFORE seeing v[k] (metric A).
        pred = phi @ rls.theta
        arx_pred_err[k] = v[k] - pred

        theta, _error, _gain = rls.update(phi, v[k])

        if k < warmup:
            continue

        physical = theta_to_physical_1RC(theta, Ts)

        if not physical["valid"]:
            n_rejected_physical += 1
            if last_valid is not None:
                r0h[k], r1h[k], c1h[k], ocvh[k] = last_valid
            continue

        params = {"r0": physical["R0"], "r1": physical["R1"], "c1": physical["C1"]}

        if not inside_bounds(params, bounds):
            n_rejected_bounds += 1
            if last_valid is not None:
                r0h[k], r1h[k], c1h[k], ocvh[k] = last_valid
            continue

        n_valid += 1
        last_valid = (physical["R0"], physical["R1"], physical["C1"], physical["OCV"])
        r0h[k], r1h[k], c1h[k], ocvh[k] = last_valid

    n_considered = n - warmup
    return {
        "arx_pred_err": arx_pred_err,
        "r0": r0h, "r1": r1h, "c1": c1h, "ocv": ocvh,
        "n_valid": n_valid,
        "n_rejected_bounds": n_rejected_bounds,
        "n_rejected_physical": n_rejected_physical,
        "n_considered": n_considered,
    }


def shadow_euler_simulate(v, i_used, t, r0h, r1h, c1h, ocvh):
    """
    Same backward-Euler recursion as FirstOrderThevenin.step_current_driven
    (ecm.py), driven by the *live*, continuously-updated recovered
    parameters, at the real per-sample dt. v_rc is re-initialized to 0 at
    the first valid sample (matching a cold-started shadow twin).
    """
    n = len(v)
    v_sim = np.full(n, np.nan)

    valid_idx = np.flatnonzero(np.isfinite(r0h))
    if valid_idx.size == 0:
        return v_sim

    first_valid = valid_idx[0]
    v_sim[first_valid] = v[first_valid]
    v_rc = 0.0

    for k in range(first_valid + 1, n):
        dt = t[k] - t[k - 1]
        r0, r1, c, ocv = r0h[k], r1h[k], c1h[k], ocvh[k]

        i_load = i_used[k]
        v_r0 = i_load * r0
        v_rc = (v_rc / dt + i_load / c) / (1.0 / dt + 1.0 / (c * r1))
        v_sim[k] = ocv - v_r0 - v_rc

    return v_sim


def rmse(a, b):
    mask = np.isfinite(a) & np.isfinite(b)
    if not np.any(mask):
        return np.nan
    return float(np.sqrt(np.mean((a[mask] - b[mask]) ** 2)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default=DEFAULT_CSV)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--out-dir", default="arx_diagnostics_output")
    parser.add_argument("--forgetting-factors", type=float, nargs="+",
                         default=[0.999, 0.99, 0.95, 0.9])
    parser.add_argument("--ts-values", type=float, nargs="+", default=[1.0, 0.1])
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    v, i_raw, t = load_ground(args.csv, max_rows=args.max_rows)
    i_used = -i_raw  # passive convention, matches ARXRLS1RCEstimator + battery.sign_convention
    n = len(v)
    print(f"Loaded {n} samples from {args.csv} "
          f"(dt={np.median(np.diff(t)):.4g}s, span={t[-1]-t[0]:.1f}s)")

    rows = []
    tracked_runs = {}

    for Ts in args.ts_values:
        for ff in args.forgetting_factors:
            for bounds_name, bounds in [("unconstrained", None), ("prod_bounds", PROD_BOUNDS)]:
                result = run_config(v, i_used, t, Ts=Ts, forgetting_factor=ff, bounds=bounds)
                v_shadow = shadow_euler_simulate(v, i_used, t, result["r0"], result["r1"],
                                                  result["c1"], result["ocv"])

                metric_a_rmse = rmse(v, v - result["arx_pred_err"])
                metric_b_rmse = rmse(v, v_shadow)

                row = {
                    "Ts": Ts,
                    "forgetting_factor": ff,
                    "bounds": bounds_name,
                    "frac_valid": result["n_valid"] / result["n_considered"],
                    "frac_rejected_bounds": result["n_rejected_bounds"] / result["n_considered"],
                    "frac_rejected_physical": result["n_rejected_physical"] / result["n_considered"],
                    "metric_A_rmse_V": metric_a_rmse,
                    "metric_B_rmse_V": metric_b_rmse,
                    "median_r0_mOhm": np.nanmedian(result["r0"]) * 1e3,
                    "median_r1_mOhm": np.nanmedian(result["r1"]) * 1e3,
                    "median_c1_F": np.nanmedian(result["c1"]),
                    "median_ocv_V": np.nanmedian(result["ocv"]),
                }
                rows.append(row)
                tracked_runs[(Ts, ff, bounds_name)] = (result, v_shadow)
                print(f"Ts={Ts:<4} ff={ff:<6} bounds={bounds_name:<13} "
                      f"valid={row['frac_valid']:.2%} "
                      f"rej_bounds={row['frac_rejected_bounds']:.2%} "
                      f"metricA_rmse={metric_a_rmse:.4f}V "
                      f"metricB_rmse={metric_b_rmse:.4f}V")

    summary = pd.DataFrame(rows)
    summary_path = os.path.join(args.out_dir, "forgetting_sweep_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"\nSummary written to {summary_path}")
    print(summary.to_string(index=False))

    # Plot: metric B RMSE vs forgetting factor, one line per Ts, faceted by bounds.
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, bounds_name in zip(axes, ["unconstrained", "prod_bounds"]):
        for Ts in args.ts_values:
            sub = summary[(summary["bounds"] == bounds_name) & (summary["Ts"] == Ts)]
            sub = sub.sort_values("forgetting_factor")
            ax.plot(sub["forgetting_factor"], sub["metric_B_rmse_V"], marker="o", label=f"Ts={Ts}")
        ax.set_xlabel("forgetting_factor")
        ax.set_title(f"bounds={bounds_name}")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("metric B RMSE [V] (shadow Euler ECM vs measured)")
    axes[0].legend()
    fig.suptitle("Voltage-tracking RMSE vs. forgetting factor and Ts")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "metricB_rmse_vs_forgetting.png"), dpi=130)

    # Plot: best and worst config voltage traces (by metric B RMSE).
    metric_b_by_key = {key: np.nan_to_num(rmse(v, v_shadow), nan=np.inf)
                        for key, (_result, v_shadow) in tracked_runs.items()}
    best_key = min(metric_b_by_key, key=metric_b_by_key.get)
    worst_key = max(metric_b_by_key, key=lambda k: (-np.inf if np.isinf(metric_b_by_key[k]) else metric_b_by_key[k]))

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    for ax, key, tag in zip(axes, [best_key, worst_key], ["best (lowest metric B RMSE)", "worst"]):
        result, v_shadow = tracked_runs[key]
        ax.plot(t, v, label="measured", lw=0.8)
        ax.plot(t, v_shadow, label="shadow Euler ECM (live params)", lw=0.8, alpha=0.8)
        ax2 = ax.twinx()
        ax2.plot(t, i_raw, color="gray", alpha=0.3, lw=0.5)
        ax2.set_ylabel("current [A] (raw csv sign)")
        ax.set_title(f"{tag}: Ts={key[0]}, ff={key[1]}, bounds={key[2]}")
        ax.set_ylabel("voltage [V]")
        ax.legend(loc="upper right")
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("time [s]")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "best_worst_voltage_traces.png"), dpi=130)

    print(f"\nBest config (lowest metric B RMSE): Ts={best_key[0]}, ff={best_key[1]}, bounds={best_key[2]}")
    print(f"Worst config: Ts={worst_key[0]}, ff={worst_key[1]}, bounds={worst_key[2]}")
    print(f"Plots written to {args.out_dir}/")


if __name__ == "__main__":
    main()
