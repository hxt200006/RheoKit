# RheoKit

**RheoKit** is a modular toolkit for calculating liquid viscosity via the **Green–Kubo (GK) formalism** following best practices recommended by Maginn *et al.* (2019, DOI: [10.33011/livecoms.1.1.6324](https://doi.org/10.33011/livecoms.1.1.6324)) and using data from **Tinker** and **Tinker-HP** molecular dynamics simulations.

It provides a reproducible, end-to-end workflow for stress-tensor extraction, integration, plateau analysis, and model fitting.

---

## 🧩 Components

RheoKit is organized into four standalone utilities:

| Tool | Description |
|------|--------------|
| **Shear** | Extracts stress tensor components from **Tinker-HP** output files and writes them into `.str` files. *(Only needed for Tinker-HP simulations.)* |
| **GKIntegrate** | Computes the stress autocorrelation function (ACF) and its Green–Kubo time integral from `.str` files. |
| **PlateauCheck** | Evaluates the running viscosity integral to identify when it reaches a stable (flat) plateau region. |
| **StressFit** | Fits averaged integrals to a **double-exponential model** to obtain the final viscosity. |

## 🧭 Usage

Each RheoKit component can be run independently. For convenience, you can store the paths to each utility in a small shell file and source it when starting a session.

### 1️⃣ Compile Shear

**Shear** is a fortran program. Compile it with `gfortran`:
```bash
cd /path/to/rheokit/src
gfortran -O3 -o shear shear.f
```

### 2️⃣ Create a setup file

Make a file named `RheoKit_commands.sh` with the absolute paths to each utility:
```bash
shear="/path/to/RheoKit/shear"
gkintegrate="/path/to/RheoKit/src/GK_Integrate.py"
platchk="/path/to/RheoKit/src/PlateauCheck.py"
stressfit="/path/to/RheoKit/src/StressFit.py"
```

### 3️⃣ Source the setup file

```bash
source RheoKit_commands.sh
```

### 4️⃣ Run the tools

You can now call each program using its variable:

```bash

# Extract stress tensors from Tinker-HP output (if needed)
$shear --list=HPOutputs.txt

# Compute the Green–Kubo integral accross replicates
python $gkintegrate --parquet water_298.parquet --box-edge 32.0 --temp 298.15 --cutoff-lag 2000000

# Analyze plateau regions
python $platchk --parquet water_298.parquet --manifest stress_manifest.txt --cutoff 200:1000:100

# Fit double-exponential model
python $stressfit --parquet water_298.parquet --cutoff 800
```

Each program supports `--help` to display available options

## 📓 Notebooks

The`notebooks/`directory contains interactive **RheoKit** utilities that mirror the python based command-line tools. Each notebook removes the CLI interface and exposes the core functions directly. Users can set input arguments in code cells and call the corresponding utility functions within the same environment. These notebooks are meant to help users understand the workflow and adapt it for their own data.

## 🧾 Citation

If you use RheoKit in your research, please cite the associated preprint:

> Viscosity Calculations of Liquids Modeled with Induced Polarizable Force Fields
> *ChemRxiv* (2025). DOI: [10.26434/chemrxiv-2025-n1rqb](https://doi.org/10.26434/chemrxiv-2025-n1rqb)

A peer-reviewed version of this work is currently in preparation.
