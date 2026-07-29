import pandas as pd
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
from math import ceil, floor
import traceback
import json
import sys
import argparse
from scipy.optimize import curve_fit
import secrets

class RunLog:
    """Line-oriented logger."""

    def __init__(self, log_name=None, program=None):
        default = "ViscoFit.log"
        base = Path(log_name or default)
        base.parent.mkdir(parents=True, exist_ok=True)
        self.path = self._resolve_unique_path(base)
        self.program = program or "generic"
        self._lines = []
        self._write_header()

    def _resolve_unique_path(self, base):
        stem, suffix = base.stem, base.suffix or ""
        i = 0
        while True:
            candidate = base if i == 0 else base.with_name(f"{stem}_{i + 1}{suffix}")
            try:
                # Atomic create for safety
                with open(candidate, "x", encoding="utf-8") as fh:
                    fh.write("")
                return candidate
            except FileExistsError:
                i += 1

    def _stamp(self):
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _write_header(self):
        """Write header block at instantiation."""
        if self.program == "generic":
            program_name = "RheoKit"
        elif self.program == "platchk":
            program_name = "RheoKit: Plateau Check"
        elif self.program == "cvfit":
            program_name = "RheoKit: Stress Fit"
        elif self.program == "runGK":
            program_name = "RheoKit: GK Integrate"
        elif self.program == "bstperr":
            program_name = "RheoKit: Error Bootstrap"

        author_name = "Daniel Relix"
        timestamp = self._stamp()

        box_width = 64
        left_margin = " " * 13
        pad_inside = box_width - 4  # subtract 2 chars for each '##'
        line = lambda text="": f"{left_margin}##  {text:<{pad_inside - 2}}##"

        header = [
            "",
            left_margin + "#" * box_width,
            left_margin + "##" + " " * (box_width - 4) + "##",
            line(program_name),
            line(f"By {author_name}"),
            left_margin + "##" + " " * (box_width - 4) + "##",
            left_margin + "#" * box_width,
            "",
            f"Job started: {timestamp}",
            f"Log file: {self.path.name}",
            "",
        ]

        self._lines.extend(header)
        self.write()

    def add(self, line=""):
        self._lines.append("   " + line)

    def section(self, title):
        self._lines.append("-" * 90)
        self._lines.append(title)

    def subsection(self, title):
        self._lines.append(title)

    def write(self):
        self.path.write_text("\n".join(self._lines) + "\n", encoding="utf-8")

    def exception(self, logtext, e):
        self._lines.append(f"Exception: {logtext}")
        tb = "".join(traceback.format_exception(type(e), e, e.__traceback__))
        self._lines.append(tb.rstrip("\n"))
        self.write()

def eta_fit(t, A_eta, alpha, tau1, tau2):
    """
    Double exponential fit for the Green–Kubo integral.
    eta(0)=0 and eta(t->inf)=A_eta * (alpha * tau1 + (1 - alpha) * tau2)
    """
    return A_eta * alpha * tau1 * (1.0 - np.exp(-t / tau1)) + A_eta * (
        1.0 - alpha) * tau2 * (1.0 - np.exp(-t / tau2))


def sigma_fit(x, A_sigma, b):
    """Power function fit to get weighing parameter."""
    return A_sigma * (x**b)


def _clamp(x, lo, hi):
    """Clamp x between lo and hi."""
    return max(lo, min(hi, x))

def make_bootstrap(n_boot,n_reps, seed=None, log=None):
    """
    Generate bootstrap replicate indices and replicate counts.

    Parameters
    ----------
    n_boot : int
        Number of bootstrap iterations.
    n_reps : int
        Number of replicate simulations available.
    seed : int or None
        Random seed. If None, a 64-bit seed is generated.

    Returns
    -------
    boot_counts : np.ndarray
        Array of shape (n_boot, n_reps). Each row gives the number of
        times each original replicate was selected in that bootstrap sample.
    seed : int
        Seed used to initialize the random number generator.
    """

    if seed is None:
        seed = secrets.randbits(64)

    rng = np.random.default_rng(seed)
    boot_indices = rng.integers(low=0, high=n_reps, size=(n_boot, n_reps))
    boot_counts = np.zeros((n_boot, n_reps), dtype=int)

    for boot_id in range(n_boot):
        boot_counts[boot_id] = np.bincount(
            boot_indices[boot_id],
            minlength=n_reps)

    return boot_counts, seed

def build_masks_bootstrap_ver(time_ps, tmin_ps=2.0, tmax_ps=None, log=None, *, silent=False):
    # Precompute grid facts
    n = len(time_ps)
    t_lo = float(time_ps[0])
    t_hi = float(time_ps[-1])
    dt_ps = float(time_ps[1] - time_ps[0]) if n > 1 else 0.0
    tmax_user = t_hi if tmax_ps is None else float(tmax_ps)

    # Clamp requested window
    tmin_c = _clamp(float(tmin_ps), t_lo, t_hi)
    tmax_c = _clamp(float(tmax_user), tmin_c, t_hi)

    # Map times to inclusive index range on a uniform grid
    i0_eta = int(ceil((tmin_c - t_lo) / dt_ps))
    i1_eta = int(floor((tmax_c - t_lo) / dt_ps))

    # Clamp mapped idx on [0, n-1] for safety
    i0_eta = _clamp(i0_eta, 0, n - 1)
    i1_eta = _clamp(i1_eta, i0_eta, n - 1)

    # Sigma starts at 0 and ends at same cutoff as eta
    i1_sigma = i1_eta

    # Make the masks
    N = time_ps.size
    eta_mask = np.zeros(N, dtype=bool)
    sigma_mask = np.zeros(N, dtype=bool)
    eta_mask[i0_eta : i1_eta + 1] = True
    sigma_mask[0 : i1_sigma + 1] = True

    if not silent and log is not None:
        log.section("MASK INFORMATION")
        log.add()
        log.add(f"Sigma Fit Range    : {0.0:.6g} - {tmax_c:.6g} ps ")

        if tmax_ps is not None and tmax_ps > t_hi:
            log.add(f"Requested end time ({tmax_ps:.3f} ps)")
            log.add(f"exceeds available data ({t_hi:.3f} ps).")
            log.add(f"Using {tmax_c:.3f} ps instead.")
            log.add()

        log.add(f"Eta Fit Range      : {tmin_c:.6g} - {tmax_c:.6g} ps")
        log.add()

    return sigma_mask, eta_mask

def _init_eta_params_bootstrap_ver(time_ps, mean_curve, eta_mask):
    """
    Build initial guess [A_eta0, alpha0, tau1_0, tau2_0] from the
    estimated plateau at the tail of the eta window.
    """

    # alpha is a mixing coefficient.
    alpha0 = 0.5

    # tau guesses are based on length of simulation.
    t_masked = time_ps[eta_mask]
    n = len(t_masked)
    if n > 1:
        dt_med = float(np.median(np.diff(t_masked)))
        span = float(t_masked[-1] - t_masked[0])
    else:
        dt_med, span = 1.0, 1.0
    if span <= 0:
        span = max(dt_med, 1.0)

    tau1_0 = _clamp(0.01 * span, 2.0 * dt_med, 0.10 * span)
    tau2_0 = _clamp(0.35 * span, 5.0 * dt_med, 1.00 * span)
    if tau1_0 > tau2_0:
        tau1_0, tau2_0 = tau2_0, tau1_0

    # A_eta guess based on median of eta(t) tail.
    # (Scaled from symbolic limit)
    # First, get the median of the last 5% of the tail.
    K = int(round(0.05 * n))
    K = max(500, K)
    K = min(K, 1000000)

    # Use a stride if there's a lot of sampling
    if K >= 100000:
        s = -(-K // 100000)
        eta_inf0 = float(np.median(mean_curve[-K::s]))
    else:
        eta_inf0 = float(np.median(mean_curve[-K:]))

    # Then, calculate symbolic limit
    denom = alpha0 * tau1_0 + (1.0 - alpha0) * tau2_0
    if denom <= 0 or not np.isfinite(denom):
        A_eta0 = max(0.0, (mean_curve[-1] - mean_curve[0]) / max(span, dt_med))
    else:
        A_eta0 = max(0.0, eta_inf0 / denom)

    return np.array([A_eta0, alpha0, tau1_0, tau2_0], dtype=float)

def run_curve_fit_bootstrap_ver(time_ps, mean_curve, stdev_curve, sigma_mask, eta_mask, weight_mode="soft"):
    # Sigma: Power-law fit
    t_sigma = time_ps[sigma_mask]
    y_sigma = stdev_curve[sigma_mask]

    popt_sigma, pcov_sigma = curve_fit(sigma_fit, t_sigma, y_sigma, maxfev=1000)

    # Collect ending sigma value for ~40% comparison
    std_fit_cf = sigma_fit(t_sigma, *popt_sigma)
    sigma_val_cf = float(std_fit_cf[-100])

    # Eta: Double exponential fit
    y_eta = mean_curve[eta_mask]
    t_eta = time_ps[eta_mask]
    dt_med = float(np.median(np.diff(t_eta)))
    tmax = t_eta[-1]

    # Initial guesses for eta params and their bounds
    p0 = _init_eta_params_bootstrap_ver(time_ps, mean_curve, eta_mask)
    A_eta0, alpha0, tau1_0, tau2_0 = map(float, p0)
    bounds = (
        [-np.inf, 0.000001, (dt_med * 0.5), (dt_med * 100)],
        [np.inf, 0.999999, (tmax * 0.15), (tmax * 0.5)],
    )

    # Build eta weights with b_cf
    A_sigma_cf, b_cf = map(float, popt_sigma)
    if weight_mode == "soft":
        weights = t_eta ** (b_cf / 2.0)  # weight ~ x^{-b}
    elif weight_mode == "heavy":
        weights = t_eta ** (b_cf)  # weight ~ x^{-2b}
    else:
        raise ValueError("weight_mode must be 'soft' or 'heavy'")

    try:
        popt_eta, pcov_eta = curve_fit(
            eta_fit,
            t_eta,
            y_eta,
            p0=p0,
            bounds=bounds,
            sigma=weights,
            absolute_sigma=True,
            maxfev=10000,
        )
        reordered = False
        A_eta, alpha, tau1, tau2 = map(float, popt_eta)
        if tau1 > tau2:
            tau1, tau2 = tau2, tau1
            alpha = 1.0 - alpha
            popt_eta = np.array([A_eta, alpha, tau1, tau2])
            idx = [0, 1, 3, 2]
            pcov_eta = pcov_eta[np.ix_(idx, idx)]
            reordered = True
        eta_inf = A_eta * (alpha * tau1 + (1 - alpha) * tau2)
        eta_fit_cf = eta_fit(t_eta, *popt_eta)
        eta_val_cf = float(eta_fit_cf[-100])
        sig_eta_ratio = ((sigma_val_cf/eta_val_cf))

    except RuntimeError as e:
        raise RuntimeError(
             "GK running integral fit failed: curve_fit did not converge "
            f"within maxfev={maxfev}."
            ) from e

    return {
        "viscosity": eta_inf,
        "A_eta": A_eta,
        "alpha": alpha,
        "tau1": tau1,
        "tau2": tau2,
        "A_sigma": A_sigma_cf,
        "b": b_cf,
        "sigma_eta_ratio_cutoff": sig_eta_ratio
    }

def make_visco_record(
    boot_id,
    status,
    failure_stage="",
    fit_result=None,
    verbose=False,
):
    record = {
        "boot_id": boot_id,
        "status": status,
        "viscosity": np.nan,
        "failure_stage": failure_stage,
    }

    if status == "pass":
        record["viscosity"] = fit_result["viscosity"]
        
    if verbose:
        record.update({
            "A_eta": np.nan,
            "alpha": np.nan,
            "tau1": np.nan,
            "tau2": np.nan,
            "A_sigma": np.nan,
            "b": np.nan,
            "sigma_eta_ratio_cutoff": np.nan,
        })

        if status == "pass":
            record.update({
                "A_eta": fit_result["A_eta"],
                "alpha": fit_result["alpha"],
                "tau1": fit_result["tau1"],
                "tau2": fit_result["tau2"],
                "A_sigma": fit_result["A_sigma"],
                "b": fit_result["b"],
                "sigma_eta_ratio_cutoff": fit_result["sigma_eta_ratio_cutoff"],
            })

    return record

def summarize_bootstrap_ci(visco_df, alpha=0.05):
    eta_pass = visco_df.loc[
        visco_df["status"] == "pass",
        "viscosity"
    ].dropna()

    n_total = len(visco_df)
    n_pass = len(eta_pass)
    n_fail = n_total - n_pass

    if n_pass == 0:
        raise ValueError("Bootstrap summary failed: no successful viscosity fits.")

    q_low = 100.0 * (alpha / 2.0)
    q_high = 100.0 * (1.0 - alpha / 2.0)

    ci_low, ci_high = np.percentile(eta_pass, [q_low, q_high])

    summary = {
        "n_boot": n_total,
        "n_pass": n_pass,
        "n_fail": n_fail,
        "alpha": alpha,
        "confidence": 1.0 - alpha,
        "viscosity_mean": eta_pass.mean(),
        "viscosity_median": eta_pass.median(),
        "viscosity_std": eta_pass.std(ddof=1),
        "ci_low": ci_low,
        "ci_high": ci_high,
    }

    return summary

def mean_stdev_from_bootstrap(df, rep_cols, counts):
    n_reps = counts.sum()

    # First pass: weighted mean
    mean_curve = np.zeros(len(df), dtype=np.float64)

    for col, count in zip(rep_cols, counts):
        if count > 0:
            x = df[col].to_numpy(copy=False)
            mean_curve += count * x

    mean_curve /= n_reps

    # Second pass: weighted variance
    var_curve = np.zeros(len(df), dtype=np.float64)

    for col, count in zip(rep_cols, counts):
        if count > 0:
            x = df[col].to_numpy(copy=False)
            var_curve += count * (x - mean_curve) ** 2

    var_curve /= (n_reps - 1)

    # Protect from negative variance caused by numerical noise
    tol = 1e-12
    min_var = var_curve.min()
    if min_var < -tol:
        raise ValueError(f"Negative weighted variance, min(var) = {min_var:.6e}.")

    var_curve = np.maximum(var_curve, 0.0)

    stdev_curve = np.sqrt(var_curve)
    stdev_curve = np.maximum(stdev_curve, tol)

    return mean_curve, stdev_curve

def _fmt_cutoff_suffix(tmax_ps):
    """Return a filename-safe cutoff suffix rounded to the nearest picosecond."""
    return f"{int(round(float(tmax_ps)))}ps"

def parse_cutoff(value):
    text = str(value).strip().lower()
    if text == "none":
        return None
    return float(value)

def bootstrap_procedure(
    path,
    n_boot,
    tmin_ps=2.0,
    tmax_ps=None,
    verbose=False,
    log_name=None,
    alpha=0.05,
    weight_mode="soft",
    viscosity_filename=None,
    counts_filename=None,
    seed=None,
):
    parquet_path = Path(path)
    stem = parquet_path.stem

    data = pd.read_parquet(parquet_path)
    time_ps = data["Time (ps)"].to_numpy(copy=False)

    if tmax_ps is None:
        tmax_ps = float(time_ps[-1])
    else:
        tmax_ps = float(tmax_ps)

    cutoff_suffix = _fmt_cutoff_suffix(tmax_ps)

    if log_name is None:
        log_name = f"{stem}_Bootstrap_{cutoff_suffix}.log"

    if viscosity_filename is None:
        viscosity_filename = f"{stem}_bootstrap_viscosities_{cutoff_suffix}.csv"

    if counts_filename is None:
        counts_filename = f"{stem}_bootstrap_counts_{cutoff_suffix}.csv"

    log = RunLog(log_name=log_name, program="bstperr")

    try:
        rep_cols = [
            col for col in data.columns
            if col not in ["Time (ps)", "Mean", "StDev"]
        ]

        n_reps = len(rep_cols)

        boot_counts, seed_used = make_bootstrap(
            n_boot,
            n_reps,
            seed=seed,
            log=log,
        )

        log.section("INPUT DATA")
        log.add()
        log.add(f"File               : {parquet_path}")
        log.add(f"Replicates         : {n_reps}")
        log.add(f"Bootstrap samples  : {n_boot}")
        log.add(f"Seed               : {seed_used}")
        log.add(f"alpha              : {alpha}")
        log.add(f"weight_mode        : {weight_mode}")
        log.add(f"verbose            : {verbose}")
        log.add()

        sigma_mask, eta_mask = build_masks_bootstrap_ver(
            time_ps,
            tmin_ps=tmin_ps,
            tmax_ps=tmax_ps,
            log=log,
        )

        visco_records = []

        for boot_id in range(n_boot):
            counts = boot_counts[boot_id]

            try:
                failure_stage = "build_bootstrap_curve"

                mean_curve, stdev_curve = mean_stdev_from_bootstrap(
                    data,
                    rep_cols,
                    counts,
                )

                failure_stage = "curve_fit"

                fit_result = run_curve_fit_bootstrap_ver(
                    time_ps,
                    mean_curve,
                    stdev_curve,
                    sigma_mask,
                    eta_mask,
                    weight_mode=weight_mode,
                )

                visco_records.append(
                    make_visco_record(
                        boot_id=boot_id,
                        status="pass",
                        failure_stage="",
                        fit_result=fit_result,
                        verbose=verbose,
                    )
                )

            except Exception as e:
                log.exception(
                    f"Bootstrap iteration {boot_id} failed during {failure_stage}.",
                    e,
                )

                visco_records.append(
                    make_visco_record(
                        boot_id=boot_id,
                        status="fail",
                        failure_stage=failure_stage,
                        fit_result=None,
                        verbose=verbose,
                    )
                )

                continue

        visco_df = pd.DataFrame(visco_records)
        visco_df.to_csv(viscosity_filename, index=False)

        boot_counts_df = pd.DataFrame(
            boot_counts,
            columns=[f"rep_{i + 1}" for i in range(boot_counts.shape[1])],
        )
        boot_counts_df.insert(0, "boot_id", np.arange(boot_counts.shape[0]))
        boot_counts_df.to_csv(counts_filename, index=False)

        ci_summary = summarize_bootstrap_ci(
            visco_df=visco_df,
            alpha=alpha,
        )

        log.section("OUTPUT FILES")
        log.add()
        log.add(f"Viscosity CSV      : {viscosity_filename}")
        log.add(f"Counts CSV         : {counts_filename}")
        log.add()

        log.section("BOOTSTRAP RESULTS")
        log.add()
        log.add(f"Num. Bstp. Samples : {ci_summary['n_boot']}")
        log.add(f"Successful fits    : {ci_summary['n_pass']}")
        log.add(f"Failed fits        : {ci_summary['n_fail']}")
        log.add()

        log.section("VISCOSITY DISTRIBUTION RESULTS")
        log.add()
        log.add(f"Mean viscosity     : {ci_summary['viscosity_mean']:.6g}")
        log.add(f"Median viscosity   : {ci_summary['viscosity_median']:.6g}")
        log.add(f"Std. dev.          : {ci_summary['viscosity_std']:.6g}")
        log.add(
            f"{100 * ci_summary['confidence']:.1f}% percentile CI: "
            f"[{ci_summary['ci_low']:.6g}, {ci_summary['ci_high']:.6g}]"
        )
        log.add()

    except Exception as e:
        log.exception("Fatal bootstrap procedure failure.", e)
        raise

    finally:
        log.write()

def build_parser():
    p = argparse.ArgumentParser(
        prog="bstperr",
        description="Run bootstrap uncertainty estimation over a parquet dataset.",
    )

    p.add_argument(
        "-p", "--parquet",
        required=True,
        help="Path to the input parquet file.",
    )

    p.add_argument(
        "--n-boot",
        type=int,
        required=True,
        help="Number of bootstrap samples to generate.",
    )

    p.add_argument(
        "--tmin-ps",
        type=float,
        default=2.0,
        help="Lower time cutoff for fitting, in ps. Default: 2.0",
    )

    p.add_argument(
        "--cutoff",
        dest="tmax_ps",
        type=parse_cutoff,
        default=None,
        metavar="VALUE|none",
        help=(
            "Upper time cutoff for fitting, in ps. "
        ),
    )

    p.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Alpha value for the percentile confidence interval. Default: 0.05",
    )

    p.add_argument(
        "--weight-mode",
        choices=("soft", "heavy"),
        default="soft",
        help="Weighting mode for the eta fit. Default: soft",
    )

    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for bootstrap sampling. Default: choose a seed automatically.",
    )

    p.add_argument(
        "-v", "--verbose",
        action="store_true",
        default=False,
        help="Include extended fitting output for each bootstrap sample. Default: False",
    )

    p.add_argument(
        "--log-name",
        default=None,
        help="Output log filename. Defaults to <parquetname>_Bootstrap.log",
    )

    p.add_argument(
        "--v-out", "--viscosity-filename",
        dest="viscosity_filename",
        default=None,
        help=(
            "Output CSV filename for bootstrap viscosity values. "
            "Defaults to <parquetname>_bootstrap_viscosities.csv"
        ),
    )

    p.add_argument(
        "--c-out", "--counts-filename",
        dest="counts_filename",
        default=None,
        help=(
            "Output CSV filename for bootstrap replicate counts. "
            "Defaults to <parquetname>_bootstrap_counts.csv"
        ),
    )

    return p

def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    parquet_path = Path(args.parquet)

    if not parquet_path.is_file():
        parser.error(f"Parquet file not found: {parquet_path}")

    if args.n_boot < 1:
        parser.error("--n-boot must be at least 1.")

    if args.tmin_ps < 0:
        parser.error("--tmin-ps must be nonnegative.")

    if not (0.0 < args.alpha < 1.0):
        parser.error("--alpha must be between 0 and 1.")

    try:
        bootstrap_procedure(
            path=str(parquet_path),
            n_boot=int(args.n_boot),
            tmin_ps=float(args.tmin_ps),
            tmax_ps=args.tmax_ps,
            verbose=bool(args.verbose),
            log_name=args.log_name,
            alpha=float(args.alpha),
            weight_mode=str(args.weight_mode),
            viscosity_filename=args.viscosity_filename,
            counts_filename=args.counts_filename,
            seed=args.seed,
        )

    except KeyboardInterrupt:
        print("Interrupted by user.", file=sys.stderr)
        return 130

    except Exception as exc:
        print(f"Bootstrap ERROR: {exc}", file=sys.stderr)
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())
