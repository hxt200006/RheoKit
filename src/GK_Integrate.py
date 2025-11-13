import argparse
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.signal import correlate
from scipy.integrate import cumulative_trapezoid
import traceback


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


def _read_stress_add_traceless(path):
    """Read a stress tensor file, return a DataFrame with added traceless components."""

    names = [
        "MD Step",
        "Stress(xx)",
        "Stress(yy)",
        "Stress(zz)",
        "Stress(xy)",
        "Stress(yz)",
        "Stress(xz)",
    ]

    df = pd.read_csv(
        path,
        sep=r"\s+",
        engine="python",
        skiprows=1,
        names=names,
    )

    trace_third = (df["Stress(xx)"] + df["Stress(yy)"] + df["Stress(zz)"]) / 3.0
    df["Sym(xx)"] = df["Stress(xx)"] - trace_third
    df["Sym(yy)"] = df["Stress(yy)"] - trace_third
    df["Sym(zz)"] = df["Stress(zz)"] - trace_third
    return df


def _acf_cumint_one(y, *, dt, L):
    """
    Compute autocorrelation of y (via scipy.signal.correlate),
    normalize by decreasing sample count, then cumulative-trapz
    integrate. Returns the integral array (length L).
    """
    # Full ACF take non-negative lags
    corr_full = correlate(y, y, mode="full", method="fft")
    corr = corr_full[len(y) - 1 : (len(y) - 1) + (L + 1)]

    # Normalize by # of overlapping samples at each lag
    norm = np.arange(len(y), len(y) - (L + 1), -1, dtype=float)
    component_acf = corr / norm

    # Cumulative integral
    return cumulative_trapezoid(component_acf, dx=dt)


def _clamp(x, lo, hi):
    """Clamp x between lo and hi."""
    return max(lo, min(hi, x))


def rep_iterator(manifest_path, base_dir=None):
    """Yield replicate stress DataFrames for Green–Kubo analysis."""

    manifest_path = Path(manifest_path).resolve()
    manifest_dir = manifest_path.parent
    root_dir = Path(base_dir).resolve() if base_dir else manifest_dir

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

    with manifest_path.open() as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            abs_path = (root_dir / line).resolve()

            if not abs_path.exists():
                print(f"Warning: {abs_path} not found — skipping.")
                continue

            df = _read_stress_add_traceless(abs_path)
            label = abs_path.stem
            print(f"Processing {label}")
            yield label, df


def run_GreenKubo(
    df, *, box_edge, temp=300, start_idx=0, cutoff_lag=500000, dt_fs=1.0, stress_freq=1
):
    """
    Use Green-Kubo relation to compute viscosity for each replicate.

    Parameters
    ----------
    df         : DataFrame
        Stress tensor component data. (Required)
    box_edge   : float or sequence of 3 floats
        Simulation box length(s) in Å.
        - If a single float, assumed cubic (Lx = Ly = Lz = box_edge)
        - If a 3-element iterable, treated as [Lx, Ly, Lz].
    temp       : float
        Simulation temperature. (Optional, default=300K)
    start_idx  : int
        First index for autocorrelation. (Optional, default=0)
    cutoff_lag : int
        Last index for autocorrelation. (Optional, default=500000)
    dt_fs      : float
        Length of time step in femtoseconds. (Optional, default=1.0)
    """
    dt = dt_fs * 1e-15 * stress_freq

    # Slice and Convert to SI units
    def _col(name):
        return (df[name].to_numpy(dtype=float)[start_idx:]) * 101325.0

    # Handle Frame semantics:
    # K = number of frames to include
    # L = max lag index; for internal use. cumtrapz will yield length L.
    xy_probe = _col("Stress(xy)")
    N = len(xy_probe)
    K = _clamp(int(cutoff_lag), 2, N)
    L = K - 1

    # Local helper to avoid repetition
    def _acf_integral(name, scale):
        y = _col(name)[:K]
        return _acf_cumint_one(y, dt=dt, L=L) * scale

    # Off-diagonal terms (*2)
    XY = _acf_integral("Stress(xy)", 2.0)
    YZ = _acf_integral("Stress(yz)", 2.0)
    XZ = _acf_integral("Stress(xz)", 2.0)

    # Traceless diagonal terms (*4/3)
    XX = _acf_integral("Sym(xx)", 4.0 / 3.0)
    YY = _acf_integral("Sym(yy)", 4.0 / 3.0)
    ZZ = _acf_integral("Sym(zz)", 4.0 / 3.0)

    total = XY + YZ + XZ + XX + YY + ZZ

    # Prefactors
    kB = 1.381e-23

    # Check if PBC is a cube or rectangular prism
    if np.isscalar(box_edge):
        V = (box_edge * 1e-10) ** 3
    else:
        Lx, Ly, Lz = np.array(box_edge, dtype=float)
        V = (Lx * Ly * Lz) * (1e-10) ** 3

    factor = V / (10.0 * kB * temp)

    # Convert to cP
    stotal = total * factor * 1e3

    return stotal


### run_all logic
### 1. read in .str file, add traceless components.
### 2. run GK integral for that replicate
### 3. Add final GK array to parquet
### 4. Add time array to parquet based on sampling rate
### 5. Repeat 1-3 for n replicates
### 6. Calculate mean, st dev of GK arrays


def run_all(
    out_parquet,
    manifest_path,
    base_dir=None,
    *,
    box_edge,
    temp,
    start_idx=0,
    cutoff_lag=500_000,
    dt_fs=1.0,
    stress_freq=1,
    log_name="batch_GK.log",
):
    log = RunLog(log_name, program="runGK")
    results_dict = {}
    n_time = None

    for name, df in rep_iterator(manifest_path, base_dir=base_dir):
        gk = run_GreenKubo(
            df=df,
            box_edge=box_edge,
            temp=temp,
            start_idx=start_idx,
            cutoff_lag=cutoff_lag,
            dt_fs=dt_fs,
            stress_freq=stress_freq,
        )

        if n_time is None:
            n_time = gk.shape[0]
            dt_ps = dt_fs * stress_freq * 1e-3
            time_ps = np.arange(1, n_time + 1) * dt_ps
            results_dict["Time (ps)"] = time_ps
        else:
            if gk.shape[0] != n_time:
                raise ValueError(
                    f"'{name}' has length {gk.shape[0]}, "
                    "which does not match {n_time}"
                )
        results_dict[name] = gk

    if not results_dict:
        raise ValueError(f"{manifest_path} is empty.")

    num_reps = len(results_dict) - 1
    replicate_results = pd.DataFrame(results_dict)

    replicate_results = replicate_results.assign(
        Mean=replicate_results.iloc[:, 1:].mean(axis=1),
        StDev=replicate_results.iloc[:, 1:].sem(axis=1),
    )

    num_pts = len(replicate_results)
    K_used = n_time + 1
    T_ps = K_used * dt_ps
    last_sample_ps = (K_used - 1) * dt_ps

    log.section("GREEN-KUBO ANALYSIS")
    log.add(f"Manifest            : {manifest_path}")
    if base_dir:
        log.add(f"Base Path           : {base_dir}")
    log.add(f"Box Dimensions      : {box_edge} Ang.")
    log.add(f"Temperature         : {temp} K")
    log.add(f"Output file         : {out_parquet}")
    log.add(f"Replicates averaged : {num_reps}")
    if cutoff_lag > K_used:
        log.add(f"NOTE: cutoff-lag {cutoff_lag} > available frames {K_used}.")
    log.add(f"Frames Used         : {num_pts}")
    log.add(f"Rep. Sim. Time      : {T_ps:.3f} ps")
    log.add(f"Integral Range      : {dt_ps:.3f} – {last_sample_ps:.3f} ps")
    log.write()
    replicate_results.to_parquet(Path(out_parquet), index=False)

    return replicate_results


### CLI
def build_parser():
    parser = argparse.ArgumentParser(
        prog="run_gk_cli.py",
        description=(
            "Run Green–Kubo viscosity calculation from a stress manifest.\n\n"
            "Example:\n"
            "  python replicate_gk_cli.py "
            "--manifest stress_manifest.txt "
            "--box-edge 25.0 25.0 25.0 "
            "--temp 298 "
            "--cutoff-lag 500000 "
            "--out-parquet results.parquet\n"
        ),
        formatter_class=lambda prog: argparse.RawTextHelpFormatter(
            prog, max_help_position=40
        ),
    )

    req = parser.add_argument_group("Required arguments")
    opt = parser.add_argument_group("Optional arguments")

    # Required
    req.add_argument(
        "--parquet",
        type=Path,
        required=True,
        help=(
            "Path to save the output parquet file.\n"
            "If only a filename is provided, it is\n"
            "written to the current directory."
        ),
    )
    req.add_argument(
        "--box-edge",
        type=float,
        nargs="+",
        required=True,
        help=(
            "Simulation box dimensions in Angstroms.\n"
            "Provide one value for a cubic box (L),\n"
            "or three values for an orthorhombic box\n"
            "(Lx Ly Lz)."
        ),
    )
    req.add_argument(
        "--temp", type=float, required=True, help="Simulation temperature in Kelvin."
    )
    req.add_argument(
        "--cutoff-lag",
        type=int,
        required=True,
        help="Maximum lag in frames for the ACF integration.",
    )

    # Optional
    opt.add_argument(
        "--manifest",
        type=Path,
        default=Path("stress_manifest.txt"),
        help=("Path to stress manifest text file.\nDefault: stress_manifest.txt"),
    )
    opt.add_argument(
        "--base-dir",
        type=Path,
        default=None,
        help=(
            "Base directory for resolving relative\n"
            "paths in the manifest. If omitted,\n"
            "paths are resolved relative to\n"
            "the manifest directory."
        ),
    )
    opt.add_argument(
        "--start-idx",
        type=int,
        default=0,
        help=(
            "Index (frame) at which to start the ACF\n"
            "calculation. Use this to skip equilibration.\n"
            "Default: 0."
        ),
    )
    opt.add_argument(
        "--dt-fs",
        type=float,
        default=1.0,
        help="MD time step in femtoseconds.\nDefault: 1.0.",
    )
    opt.add_argument(
        "--stress-freq",
        type=int,
        default=1,
        help=(
            "Interval between stress tensor prints\n"
            "in MD steps. Matches the STRESS-FREQ\n"
            "keyword in Tinker. For Tinker-HP, leave\n"
            "at 1 (default)."
        ),
    )
    opt.add_argument(
        "--log-name",
        type=Path,
        default=Path("batch_GK.log"),
        help="Path to the log file.\nDefault: batch_GK.log",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    box_edge = args.box_edge
    if len(box_edge) == 1:
        box_edge = box_edge * 3

    replicate_results = run_all(
        out_parquet=args.parquet,
        manifest_path=args.manifest,
        base_dir=args.base_dir,
        box_edge=box_edge,
        temp=args.temp,
        start_idx=args.start_idx,
        cutoff_lag=args.cutoff_lag,
        dt_fs=args.dt_fs,
        stress_freq=args.stress_freq,
        log_name=args.log_name,
    )


if __name__ == "__main__":
    main()
