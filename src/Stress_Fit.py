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


### Objects
# Helper, ensures parquet data is immuatable
def _ro_f64(a):
    a = np.asarray(a, dtype=np.float64)
    a.setflags(write=False)
    return a


@dataclass(frozen=True)
class IngestData:
    """Arrays extracted from Parquet."""

    time_ps: np.ndarray
    mean: np.ndarray
    stdev: np.ndarray
    n_rows: int

    @classmethod
    def make(cls, time_ps, mean, stdev):
        """ "Normalize data types"""
        tp = _ro_f64(time_ps)
        m = _ro_f64(mean)
        sd = _ro_f64(stdev)
        if not (len(tp) == len(m) == len(sd)):
            raise ValueError("Column lengths differ.")
        return cls(tp, m, sd, len(tp))


@dataclass(frozen=True)
class MaskInfo:
    """Masks and index windows for eta and sigma fits."""

    tmin_ps: float
    tmax_ps: float
    i0_eta: int
    i1_eta: int
    i1_sigma: int
    eta_mask: np.ndarray
    sigma_mask: np.ndarray
    n_eta: int
    n_sigma: int


class FitResult:
    """
    Container for fitted parameters with built-in JSON I/O.

    Parameters
    ----------
    path : str
        File path for saving/loading parameter JSON.
    """

    def __init__(self, path=None, sigma_params=None, eta_params=None):
        default = "Stress_Fit.json"
        base = Path(path or default)
        base.parent.mkdir(parents=True, exist_ok=True)
        self.path = self._resolve_unique_path(base)
        self.sigma_params = sigma_params or {}
        self.eta_params = eta_params or {}

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

    def write(self):
        """Write parameters to self.path as JSON."""
        data = {
            "sigma_params": self.sigma_params,
            "eta_params": self.eta_params,
        }
        with open(self.path, "w") as f:
            json.dump(data, f, indent=2)

    def __repr__(self):
        return f"FitResult(path='{self.path}', sigma={self.sigma_params}, eta={self.eta_params})"


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


### Models and helper functions


def eta_fit(t, A_eta, alpha, tau1, tau2):
    """
    Double exponential fit for the Green–Kubo integral.
    eta(0)=0 and eta(t->inf)=A_eta.
    """
    return A_eta * alpha * tau1 * (1.0 - np.exp(-t / tau1)) + A_eta * (
        1.0 - alpha
    ) * tau2 * (1.0 - np.exp(-t / tau2))


def sigma_fit(x, A_sigma, b):
    """Power function fit to get weighing parameter."""
    return A_sigma * (x**b)


def _clamp(x, lo, hi):
    """Clamp x between lo and hi."""
    return max(lo, min(hi, x))


def _auto_fmt(arr, decimals=12):
    """Return a NumPy array2string formatter that aligns floats by decimal."""

    max_val = np.max(np.abs(arr))
    int_width = len(str(int(max_val))) + 1
    total_width = int_width + decimals + 1
    fmt = f"{{:{total_width}.{decimals}f}}"
    return {"float_kind": lambda x, f=fmt: f.format(x)}


def _init_eta_params(data, mask, log=None):
    """
    Build initial guess [A_eta0, alpha0, tau1_0, tau2_0] from the
    estimated plateau at the tail of the eta window.
    """

    # alpha is a mixing coefficient.
    alpha0 = 0.5

    # tau guesses are based on length of simulation.
    t_masked = data.time_ps[mask.eta_mask]
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
        eta_inf0 = float(np.median(data.mean[-K::s]))
    else:
        eta_inf0 = float(np.median(data.mean[-K:]))

    # Then, calculate symbolic limit
    denom = alpha0 * tau1_0 + (1.0 - alpha0) * tau2_0
    if denom <= 0 or not np.isfinite(denom):
        A_eta0 = max(0.0, (data.mean[-1] - data.mean[0]) / max(span, dt_med))
    else:
        A_eta0 = max(0.0, eta_inf0 / denom)

    return np.array([A_eta0, alpha0, tau1_0, tau2_0], dtype=float)


### Main Logic:
### 1. Read in Data
### 2. Make Masks
### 3. run curve_fit()


def load_parquet(path, log=None):
    """
    Load Parquet with required columns.

    Parameters
    ----------
    path : str
        Path to the Parquet file.
    log : RunLog or None
        Optional logger to record a brief ingest block.

    Returns
    -------
    IngestData
        time_ps, mean, stdev as float64 arrays and row count.

    Raises
    ------
    ValueError
        If required columns are missing.
    """

    required_cols = ("Time (ps)", "Mean", "StDev")
    df = pd.read_parquet(path, columns=["Time (ps)", "Mean", "StDev"])
    data = IngestData.make(df["Time (ps)"], df["Mean"], df["StDev"])

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if log is not None:
        log.section("INPUT DATA")
        log.add()

        log.add(f"File               : {path}")
        log.add(f"Total Data Points  : {data.n_rows}")
        log.add()

    return data


def build_masks(time_ps, tmin_ps=2.0, tmax_ps=None, log=None, *, silent=False):
    """
    Build eta and sigma windows/masks using times in ps.

    Windows:
      - eta   : [tmin_ps, tmax_ps]  (default tmin_ps=2.0)
      - sigma : [0,       tmax_ps]  (shares the same cutoff)

    Parameters
    ----------
    time_ps : np.ndarray
        1D array of times (ps), length N.
    tmin_ps : float
        Start time (ps) for eta window. Default is 2.0 ps.
    tmax_ps : float or None
        End time (ps). If None, uses the last time in the array.
    log : RunLog or None
        Optional logger to record window details.

    Returns
    -------
    MaskInfo
        Indices, masks, and counts for eta and sigma windows.
    """

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

    n_eta = int(eta_mask.sum())
    n_sigma = int(sigma_mask.sum())

    if not silent and log is not None:
        log.section("MASK INFORMATION")
        log.add()
        log.add(f"Sigma Fit Range    : {0.0:.6g} - {tmax_c:.6g} ps ")
        log.add(f"Index Range        : [0, {i1_sigma}]")
        log.add()

        if tmax_ps is not None and tmax_ps > t_hi:
            log.add(f"Requested end time ({tmax_ps:.3f} ps)")
            log.add(f"exceeds available data ({t_hi:.3f} ps).")
            log.add(f"Using {tmax_c:.3f} ps instead.")
            log.add()

        log.add(f"Eta Fit Range      : {tmin_c:.6g} - {tmax_c:.6g} ps")
        log.add(f"Index Range        : [{i0_eta}, {i1_eta}]")
        log.add(f"Points in Eta Fit  : {n_eta}")
        log.add()

    masks = MaskInfo(
        tmin_ps=tmin_c,
        tmax_ps=tmax_c,
        i0_eta=i0_eta,
        i1_eta=i1_eta,
        i1_sigma=i1_sigma,
        eta_mask=eta_mask,
        sigma_mask=sigma_mask,
        n_eta=n_eta,
        n_sigma=n_sigma,
    )
    return masks


def run_curve_fit(data, mask, weight_mode="soft", json_name=None, log=None):
    # Sigma: Power-law fit
    t_sigma = data.time_ps[mask.sigma_mask]
    y_sigma = data.stdev[mask.sigma_mask]

    popt_sigma, pcov_sigma = curve_fit(sigma_fit, t_sigma, y_sigma, maxfev=1000)

    # Collect ending sigma value for ~40% comparison
    std_fit_cf = sigma_fit(t_sigma, *popt_sigma)
    sigma_val_cf = float(std_fit_cf[-100])

    # Eta: Double exponential fit
    y_eta = data.mean[mask.eta_mask]
    t_eta = data.time_ps[mask.eta_mask]
    dt_med = float(np.median(np.diff(t_eta)))
    tmax = t_eta[-1]

    # Initial guesses for eta params and their bounds
    p0 = _init_eta_params(data, mask=mask, log=log)
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
        rse_eta = np.sqrt(np.diag(pcov_eta)) / np.abs(popt_eta) * 100
        A_eta_rse, alpha_rse, tau1_rse, tau2_rse = map(float, rse_eta)
        eta_inf = A_eta * (alpha * tau1 + (1 - alpha) * tau2)
        eta_fit_cf = eta_fit(t_eta, *popt_eta)
        eta_val_cf = float(eta_fit_cf[-100])

    except Exception as e:
        if log is not None:
            log.exception("curve_fit() failed with error:", e)
            raise Exception

    if log is not None:
        log.section("FIT SUMMARY")
        log.add()

        log.subsection("Eta parameter initial guesses:")
        log.add(f"A_eta              : {A_eta0:0.8f}")
        log.add(f"alpha              : {alpha0}")
        log.add(f"tau1               : {tau1_0:0.4f}")
        log.add(f"tau2               : {tau2_0:0.4f}")
        log.add()

        log.subsection("Sigma fit results:")
        log.add(f"A_sigma            : {A_sigma_cf:.6f}")
        log.add(f"b                  : {b_cf:.6f}")
        log.add(f"St. Dev. at cutoff : {sigma_val_cf:.6f}")
        log.add(
            f"Covariance Matrix  : {np.array2string(pcov_sigma, formatter=_auto_fmt(pcov_sigma)).replace('\n', '\n' + ' ' * 23)}"
        )
        log.add()

        log.subsection("Eta fit results:")
        log.add(f"A_eta              : {A_eta:.6f}")
        log.add(f"alpha              : {alpha}")
        log.add(f"tau1               : {tau1:.6f} (RSE = {tau1_rse:12.2f}%)")
        log.add(f"tau2               : {tau2:.6f} (RSE = {tau2_rse:12.2f}%)")
        if weight_mode == "heavy":
            log.add("weight mode        : heavy")
        log.add(f"Viscosity at cutoff: {eta_val_cf:.6f}")
        log.add(f"sigma(t_cut)/eta(t_cut): {(sigma_val_cf / eta_val_cf):6g}")
        log.add("Covariance Matrix  :")
        log.add(
            f"{np.array2string(pcov_eta, formatter=_auto_fmt(pcov_eta), max_line_width=np.inf, threshold=np.inf).replace('\n', '\n' + ' ' * 3)}"
        )
        if reordered:
            log.add()
            log.subsection(
                "NOTE: tau1 and tau2 switched during fit. Parameters and covariance reordered."
            )

        log.add()

        log.section("RESULTS")
        log.add()
        log.add(f"Final Viscosity    : {eta_inf}")

    cf_results = FitResult(
        path=json_name,
        sigma_params={"A_sigma": A_sigma_cf, "b": b_cf},
        eta_params={"A_eta": A_eta, "alpha": alpha, "tau1": tau1, "tau2": tau2},
    )
    return cf_results


### Wrapper for main logic
def fit_procedure(
    log_name, json_name, parquet_path, tmin_ps=2.0, tmax_ps=None, weight_mode="soft"
):
    log = RunLog(log_name=log_name, program="cvfit")
    data = load_parquet(parquet_path, log=log)
    masks = build_masks(data.time_ps, tmin_ps=tmin_ps, tmax_ps=tmax_ps, log=log)
    results = run_curve_fit(
        data, mask=masks, json_name=json_name, weight_mode=weight_mode, log=log
    )
    log.write()
    results.write()
    return results, log


### CLI functions
def parse_range_list(specs=None, allow_none=True, inclusive=True, tol=1e-12):
    """
    Parse CLI range/list specs into a flat list of unique floats (optionally with None).

    Accepted forms (mixable; repeatable):
      - Single value:         "800"
      - Comma list:           "400,600,800"
      - Range (MATLAB style): "200:1000:200"   # start:stop:step
      - None token:           "none" (if allow_none=True)

    Parameters
    ----------
    specs : iterable of str or None
        Strings from repeated CLI flags (e.g., argparse with action='append').
        If None or empty, returns [] (caller decides default behavior).
    allow_none : bool
        If True, the token "none" (case-insensitive) yields a None entry.
    inclusive : bool
        If True, range end is included when it lands on the grid within tolerance.
        If False, emulate half-open logic (exclude end).
    tol : float
        Tolerance for floating-point comparisons when deciding inclusion of stop.

    Returns
    -------
    list
        A list like [200.0, 400.0, 600.0, None, 800.0].

    """

    if not specs:
        return []

    out = []
    for s in specs:
        for token in s.split(","):
            token = token.strip()
            if not token:
                continue
            low = token.lower()

            # handle "none"
            if allow_none and low == "none":
                out.append(None)
                continue

            # handle "start:stop:step"
            if ":" in token:
                parts = [p.strip() for p in token.split(":")]
                if len(parts) != 3:
                    raise ValueError("Bad range '%s': expected start:stop:step" % token)
                try:
                    start = float(parts[0])
                    stop = float(parts[1])
                    step = float(parts[2])
                except Exception as e:
                    raise ValueError("Bad range '%s': %s" % (token, e))
                if step == 0:
                    raise ValueError("Range step must be non-zero")

                vals = []
                v = start
                if step > 0:
                    while v <= stop + (tol if inclusive else -tol):
                        vals.append(v)
                        v += step
                else:
                    while v >= stop - (tol if inclusive else -tol):
                        vals.append(v)
                        v += step

                out.extend(vals)
                continue

            # handle single numeric value
            try:
                out.append(float(token))
            except Exception as e:
                raise ValueError("Invalid value '%s': %s" % (token, e))

    # Deduplicate while preserving order
    seen = set()
    uniq = []
    for v in out:
        key = ("None",) if v is None else ("val", v)
        if key not in seen:
            seen.add(key)
            uniq.append(v)

    return uniq


def _fmt_cutoff_suffix(tmax_ps, width=0):
    """Return suffix string for cutoff, or empty string if None."""
    if tmax_ps is None:
        return ""
    if float(tmax_ps).is_integer():
        t_str = f"{int(tmax_ps)}"
    else:
        t_str = str(tmax_ps).replace(".", "p")
    if width:
        t_str = t_str.zfill(width)
    return f"{t_str}ps"


def _append_suffix_before_ext(path_str, suffix):
    """Append suffix before file extension if suffix is nonempty."""
    if not suffix:
        return path_str
    p = Path(path_str)
    stem = p.stem
    ext = "".join(p.suffixes)
    return str(p.with_name(f"{stem}_{suffix}{ext}"))


def build_parser():
    p = argparse.ArgumentParser(
        prog="cvfit",
        description="Run the end-to-end curve fitting pipeline over a parquet dataset.",
    )
    p.add_argument(
        "-p", "--parquet", required=True, help="Path to the input parquet file."
    )
    p.add_argument(
        "--log-name",
        default=None,
        help="Output log filename. Defaults to <parquetname>_StressFit.log",
    )
    p.add_argument(
        "--json-name",
        default=None,
        help="Output JSON filename. Defaults to <parquetname>_StressFit.json",
    )
    p.add_argument(
        "--tmin-ps",
        type=float,
        default=2.0,
        help="Lower time cutoff (ps). Default: 2.0",
    )
    p.add_argument(
        "--cutoff",
        dest="cutoffs",
        default=None,
        action="append",
        nargs="+",
        metavar="VALUE|none",
        help="Upper time cutoff (ps).",
    )
    p.add_argument(
        "--weight-mode",
        default="soft",
        help="Weighting mode to prioritize eariler timscales. Can be 'soft' or 'heavy'. Default: soft",
    )
    return p


def main(argv=None):
    exit_code = 0
    parser = build_parser()
    args = parser.parse_args(argv)

    parquet_path = Path(args.parquet)
    if not parquet_path.is_file():
        parser.error(f"Parquet file not found: {parquet_path}")

    if args.log_name:
        base_log = args.log_name
    else:
        base_log = f"{Path(args.parquet).stem}_StressFit.log"

    if args.json_name:
        base_json = args.json_name
    else:
        base_json = f"{Path(args.parquet).stem}_StressFit.json"

    specs = [token for group in (args.cutoffs or []) for token in group]
    cutoff_list = parse_range_list(specs, allow_none=True)
    numeric_cutoffs = [c for c in cutoff_list if c is not None]
    pad_width = len(str(int(max(numeric_cutoffs)))) if numeric_cutoffs else 0
    loop = len(cutoff_list) > 1

    for time in cutoff_list:
        suff = _fmt_cutoff_suffix(time, pad_width)
        log_out = _append_suffix_before_ext(base_log, suff) if loop else base_log
        json_out = _append_suffix_before_ext(base_json, suff) if loop else base_json

        try:
            results, log = fit_procedure(
                log_name=log_out,
                json_name=json_out,
                parquet_path=str(parquet_path),
                tmin_ps=float(args.tmin_ps),
                tmax_ps=time,
                weight_mode=str(args.weight_mode),
            )
        except KeyboardInterrupt:
            msg = "Interrupted by user."
            print(msg, file=sys.stderr)
            if log is not None:
                log.add_line(msg)
                log.write()
            return 130
        except Exception as exc:
            msg = (
                f"Stress Fit ERROR for cutoff {time or 'last frame'}: {exc}"
            )
            print(msg, file=sys.stderr)
            if log is not None:
                log.add_line(msg)
                log.write()
            exit_code = 1
            continue
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
