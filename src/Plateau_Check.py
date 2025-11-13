import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import traceback
from dataclasses import dataclass


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


@dataclass(frozen=True)
class CutoffResult:
    chosen_index: float
    chosen_time_ps: float
    pass_streak: int
    params: dict
    candidate_times_ps: np.ndarray
    mapped_indices: np.ndarray
    pass_flags: np.ndarray
    rel_range: np.ndarray
    abs_range: np.ndarray
    window_sizes_ps: np.ndarray
    window_counts: np.ndarray
    eta_median: np.ndarray
    eta_mad: np.ndarray
    table: str


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


def _mad(x: np.ndarray) -> float:
    m = float(np.nanmedian(x))
    return float(np.nanmedian(np.abs(x - m)))


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
        log.add(f"Simulation Time    : {data.time_ps[-1]} ps")
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

    tmin = float(tmin_ps)
    tmax = float(time_ps[-1]) if tmax_ps is None else float(tmax_ps)

    # Map times to indices (inclusive right end for eta/sigma)
    i0_eta = int(np.searchsorted(time_ps, tmin, side="left"))
    i1_eta = int(np.searchsorted(time_ps, tmax, side="right") - 1)
    i1_eta = max(i1_eta, i0_eta)  # ensure non-empty ordering

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
        log.section("MASKS")
        log.add()
        log.add(f"Sigma Fit Range    : {0.0:.6g} - {tmax:.6g} ps ")
        log.add(f"Index Range        : [0, {i1_sigma}]")

        log.add()
        log.add(f"Eta Fit Range      : {tmin:.6g} - {tmax:.6g} ps")
        log.add(f"Index Range        : [{i0_eta}, {i1_eta}]")
        log.add(f"Points in Eta Fit  : {n_eta}")
        log.add()

    masks = MaskInfo(
        tmin_ps=tmin,
        tmax_ps=tmax,
        i0_eta=i0_eta,
        i1_eta=i1_eta,
        i1_sigma=i1_sigma,
        eta_mask=eta_mask,
        sigma_mask=sigma_mask,
        n_eta=n_eta,
        n_sigma=n_sigma,
    )
    return masks


# Main Logic
def cutoff_checker(
    parquet_path,
    *,
    candidate_times_ps,
    W_fraction=0.10,
    eps_rel=0.02,
    eps_abs=None,
    pass_streak=2,
    log_name="plateau_check.log",
):
    """
    Test the provided cutoff times to see if the median of the data is flat.

    Candidate mapping: for a provided cutoff time tc, we use j = max { i : t[i] <= tc }.
    A candidate is considered invalid if it does not map inside the eta_mask.
    """

    # Load data & masks
    log = RunLog(log_name=log_name, program="platchk")
    data = load_parquet(parquet_path, log=log)
    t = np.asarray(data.time_ps, dtype=float)
    eta = np.asarray(data.mean, dtype=float)
    _sig = np.asarray(data.stdev, dtype=float)

    masks = build_masks(t, tmin_ps=0.0, tmax_ps=None, log=log, silent=True)
    eta_idx = np.where(masks.eta_mask)[0]
    if eta_idx.size == 0:
        msg = "No eta candidates available."
        e = RuntimeError("Empty Eta Mask")
        if log:
            log.section("CUTOFF CHECKER – RESULT")
            log.exception(msg, RuntimeError, e)
        raise RuntimeError(msg)

    t_eta0 = t[eta_idx[0]]

    # Map candidate cutoff times to array idxs
    user_times = np.asarray(candidate_times_ps, dtype=float)
    mapped_idx = np.full(user_times.shape, -1, dtype=int)

    for k, tc in enumerate(user_times):
        j = int(np.searchsorted(t, tc, side="right") - 1)
        if 0 <= j < t.size and masks.eta_mask[j]:
            mapped_idx[k] = j

    # Log settings
    if log:
        log.section("PLATEAU WINDOW SETTINGS")
        log.add()
        log.add(f"Window fraction     : {W_fraction}")
        log.add(f"eps relative        : {eps_rel}")
        log.add(
            f"eps absolute        : {eps_abs if eps_abs is not None else 'No Threshold Set'}"
        )
        log.add(f"Min. pass streak    : {pass_streak}")
        log.add()

    # Evaluate medians for each plateau window
    n = user_times.size
    rel_range = np.full(n, np.nan, dtype=float)
    abs_range = np.full(n, np.nan, dtype=float)
    pass_flags = np.zeros(n, dtype=bool)
    floor_used = np.zeros(n, dtype=bool)

    W_used_ps = np.full(n, np.nan, dtype=float)
    npts_win = np.full(n, 0, dtype=int)
    eta_med = np.full(n, np.nan, dtype=float)
    eta_mad = np.full(n, np.nan, dtype=float)

    use_abs_check = eps_abs is not None

    def _window_start_index(tj):
        frac = float(W_fraction) * max(tj - t_eta0, 0.0)
        # apply internal floor
        Wj = frac if frac >= 0.5 else 0.5
        floor = (Wj == 0.5) and (frac < 0.5)
        left_t = tj - Wj
        i0 = int(np.searchsorted(t, left_t, side="left"))
        i0 = max(0, min(i0, int(eta_idx[-1])))
        return i0, Wj, floor

    running_streak = 0
    chosen_index = None
    chosen_time = None

    for k, (tc, j) in enumerate(zip(user_times, mapped_idx)):
        if j < 0:
            running_streak = 0
            continue

        tj = t[j]
        i0, Wj, floor = _window_start_index(tj)
        W_used_ps[k] = Wj
        floor_used[k] = floor

        if j - i0 + 1 < 3:
            npts_win[k] = j - i0 + 1
            running_streak = 0
            continue

        etaw = eta[i0 : j + 1]
        npts_win[k] = etaw.size

        emax = float(np.nanmax(etaw))
        emin = float(np.nanmin(etaw))
        emed = float(np.nanmedian(etaw))
        d_abs = emax - emin
        d_rel = d_abs / max(abs(emed), 1e-12)

        eta_med[k] = emed
        eta_mad[k] = _mad(etaw)
        abs_range[k] = d_abs
        rel_range[k] = d_rel

        ok_rel = d_rel <= eps_rel
        ok_abs = (d_abs <= eps_abs) if use_abs_check else True

        passed = bool(ok_rel and ok_abs)
        pass_flags[k] = passed

        running_streak = running_streak + 1 if passed else 0
        if chosen_index is None and running_streak >= int(pass_streak):
            chosen_index = int(j)
            chosen_time = float(tc)

    ### Write Table
    W_cut, W_w, W_n, W_med, W_mad, W_rel, W_abs, W_pass, W_strk = (
        10,
        10,
        9,
        10,
        9,
        9,
        9,
        4,
        6,
    )

    def H(txt, w):
        return f"{txt:>{w}}"

    width = 84
    sep = "-" * width
    title_line = "Plateau Window Evaluation"

    hdr = " ".join(
        [
            H("cutoff(ps)", W_cut),
            H("Window(ps)", W_w),
            H("#pts", W_n),
            H("eta_med", W_med),
            H("MAD", W_mad),
            H("eps_rel", W_rel),
            H("eps_abs", W_abs),
            H("pass", W_pass),
            H("streak", W_strk),
        ]
    )

    rows = []
    streak_now = 0
    for k in range(len(user_times)):
        tc, j = user_times[k], mapped_idx[k]
        Wj, npt = W_used_ps[k], npts_win[k]
        emed, mad = eta_med[k], eta_mad[k]
        dr, da = rel_range[k], abs_range[k]
        p = pass_flags[k]

        if j < 0 or not np.isfinite(Wj):
            # Unmapped / invalid candidate: keep alignment with dashes
            pass_char = "N" if j >= 0 else "-"
            # keep streak logic only for mapped entries
            if j >= 0:
                streak_now = streak_now + 1 if p else 0
                streak_str = f"{streak_now}/{pass_streak}"
            else:
                streak_str = "-"
            row = " ".join(
                [
                    f"{tc:{W_cut}.0f}",
                    f"{'-':>{W_w}}",
                    f"{'-':>{W_n}}",
                    f"{'-':>{W_med}}",
                    f"{'-':>{W_mad}}",
                    f"{'-':>{W_rel}}",
                    f"{'-':>{W_abs}}",
                    f"{pass_char:>{W_pass}}",
                    f"{streak_str:>{W_strk}}",
                ]
            )
            rows.append(row)
            continue

        # mapped candidate
        streak_now = streak_now + 1 if p else 0
        streak_str = f"{streak_now}/{pass_streak}"

        row = " ".join(
            [
                f"{tc:{W_cut}.0f}",
                f"{Wj:{W_w}.1f}",
                f"{npt:{W_n}d}",
                f"{emed:{W_med}.4f}",
                f"{mad:{W_mad}.4f}",
                f"{dr:{W_rel}.4f}",
                f"{da:{W_abs}.4f}",
                f"{('Y' if p else 'N'):>{W_pass}}",
                f"{streak_str:>{W_strk}}",
            ]
        )
        rows.append(row)

    table_lines = [sep, title_line, sep, hdr, sep, *rows, sep]

    table = "\n".join(table_lines)

    if log:
        for line in table.splitlines():
            log.add(line)
        if np.any(floor_used):
            log.add("NOTE:")
            log.add(
                "The trailing window was clamped to the internal minimum (0.5 ps) for one or"
            )
            log.add(
                "more candidates. This typically happens for early cutoffs where the fractional"
            )
            log.add(
                "window would be smaller than 0.5 ps. Consider increasing W_fraction or starting"
            )
            log.add("candidates later if this is not desired.")
        log.write()

    return CutoffResult(
        chosen_index=chosen_index,
        chosen_time_ps=chosen_time,
        pass_streak=pass_streak,
        params={
            "W_fraction": W_fraction,
            "eps_rel": eps_rel,
            "eps_abs": eps_abs,
            "pass_streak": pass_streak,
        },
        candidate_times_ps=user_times,
        mapped_indices=mapped_idx,
        pass_flags=pass_flags,
        rel_range=rel_range,
        abs_range=abs_range,
        window_sizes_ps=W_used_ps,
        window_counts=npts_win,
        eta_median=eta_med,
        eta_mad=eta_mad,
        table=table,
    )


# CLI interface
def parse_range_list(
    specs=None,
    allow_none=False,
    inclusive=True,
    tol=1e-12,
):
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
        A list of floats.
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


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        prog="Pleateau Check",
        description="Tests the stability of Green–Kubo viscosity integrals using trailing-window medians to detect plateaus.",
    )
    # Required arguments: parquet file & candidate cutoff times
    p.add_argument(
        "--parquet",
        required=True,
        help="Path to input Parquet with columns time_ps, mean, stdev.",
    )
    p.add_argument(
        "--cutoff",
        dest="cutoffs",
        action="append",
        nargs="+",
        metavar="PS",
        help="Explicit candidate cutoff times (ps), e.g. --cutoff 800 900 1000.",
    )
    # Optional arguements: plateau parameters
    p.add_argument(
        "--W-fraction",
        type=float,
        default=0.10,
        help="Trailing window fraction (default: 0.10).",
    )
    p.add_argument(
        "--eps-rel",
        type=float,
        default=0.05,
        help="Relative tolerance for plateau acceptance (default: 0.05).",
    )
    p.add_argument(
        "--eps-abs",
        type=float,
        default=None,
        help="Absolute tolerance (optional). If omitted, absolute check is disabled.",
    )
    p.add_argument(
        "--streak",
        type=int,
        default=2,
        help="Consecutive pass streak required (default: 2).",
    )
    # Optional arguements: logging/output
    p.add_argument(
        "--log-name",
        default="plateau_check.log",
        help='Output log file (default: "plateau_check.log").',
    )
    p.add_argument(
        "--quiet", action="store_true", help="Suppress results table output to stdout."
    )
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    specs = [token for group in (args.cutoffs or []) for token in group]
    candidate_times = parse_range_list(specs, allow_none=False)

    result = cutoff_checker(
        args.parquet,
        candidate_times_ps=candidate_times,
        W_fraction=args.W_fraction,
        eps_rel=args.eps_rel,
        eps_abs=args.eps_abs,
        pass_streak=args.streak,
        log_name=args.log_name,
    )

    if not args.quiet and hasattr(result, "table"):
        print(result.table)

    try:
        if getattr(result, "chosen_time_ps", None) is None:
            mapped = int(np.sum(getattr(result, "mapped_indices", []) >= 0))
            sys.exit(4 if mapped == 0 else 3)
        else:
            sys.exit(0)
    except Exception:
        sys.exit(0)


if __name__ == "__main__":
    main()
