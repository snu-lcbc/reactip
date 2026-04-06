# ReactIP — Reactive Interatomic Potential

ReactIP is an inference package for NequIP/Allegro interatomic potential
models trained on Halo8, reaction-pathway datasets, including off-equilibrium
structures sampled along bond-breaking and bond-forming pathways ([read more](https://www.nature.com/articles/s41597-025-05944-3)), with an
ASE-native calculator and a
[pyGSM](https://github.com/ZimmermanGroup/pyGSM) SE-GSM runner by
[Zimmerman's Group](https://github.com/ZimmermanGroup/molecularGSM).
ReactIP is optimized primarily for GPU inference and target-specific
compilation, while packaged `.nequip.zip` models remain usable for CPU runs.

## Core rules

- `ReactIPCalculator` stays in ASE units: `eV` and `eV/Angstrom`.
- If you want standalone reporting in `kcal/mol`, convert a returned result
  dict explicitly with `ReactIPCalculator.convert_results_to_units(...)`.
- The supported SE-GSM path is `pyGSM` + `ReactIPLoT`.
- Model weights are not assumed to be committed to Git. Pass `--model` or set
  `REACTIP_MODEL`.

## Install

From the `reactip/` directory:

```bash
pip install -e .
```

SE-GSM functionality also requires `pyGSM`. Install it from the upstream
repository:

```bash
git clone https://github.com/ZimmermanGroup/pyGSM.git
cd pyGSM
pip install -e .
```

For local development, ReactIP also tries `REACTIP_PYGSM_DIR` and a sibling
`../se-gsm/pyGSM` checkout as import fallbacks when `pyGSM` is not installed in
the active environment.


## Quick start

```python
from reactip import ReactIPCalculator

calc = ReactIPCalculator("models/model_e1f9_l2_f32.nequip.zip", device="cpu")
result = calc.calculate_xyz(xyz_string)

energy_ev = result["energy"]
forces_ev_per_a = result["forces"]

result_kcal = ReactIPCalculator.convert_results_to_units(result, energy_units="kcal/mol")
energy_kcal = result_kcal["energy"]
```

## Model formats

`ReactIPCalculator` and `run_se_gsm.py` accept two model formats:

| Format | Extension | Load speed | Environment requirement |
| --- | --- | --- | --- |
| Compiled | `.nequip.pt2` / `.nequip.pth` | Fast | `torch` + `ase` + `nequip` |
| Raw checkpoint | `.ckpt` | Slower | Full training environment |

The `.ckpt` path loads the model directly for inference. For production use,
pre-compile with `nequip-compile`.

The default bundled model package in this repository is:

- `models/model_e1f9_l2_f32.nequip.zip`

The `models/` directory contains packaged and compiled model artifacts. For
target-specific compilation workflows, see `deployment/README.md`,
`deployment/compile_from_package.sh`, and `deployment/compile_model.slurm`.

## Calculator CLI

Use `reactip_calculator.py` for single-point energy and force evaluation from
an XYZ file:

```bash
python reactip_calculator.py \
    --model models/model_e1f9_l2_f32.nequip.zip \
    --xyz examples/benchmark_cases/butadiene_ethylene_diels_alder__C6H10/reactant.xyz \
    --device cpu
```

If you want converted standalone output:

```bash
python reactip_calculator.py \
    --model models/model_e1f9_l2_f32.nequip.zip \
    --xyz examples/benchmark_cases/butadiene_ethylene_diels_alder__C6H10/reactant.xyz \
    --units kcal/mol
```

For the full CLI surface:

```bash
python reactip_calculator.py --help
```

## SE-GSM CLI

Use `run_se_gsm.py` for pyGSM runs plus exported artifacts:

```bash
python run_se_gsm.py \
    --model models/model_e1f9_l2_f32.nequip.zip \
    --xyz examples/benchmark_cases/butadiene_ethylene_diels_alder__C6H10/reactant.xyz \
    --isomers examples/benchmark_cases/butadiene_ethylene_diels_alder__C6H10/isomers.txt \
    --label butadiene_ethylene_diels_alder__C6H10 \
    --reaction-label "Butadiene + ethylene Diels-Alder" \
    --formula C6H10 \
    --case-kind benchmark \
    --device cpu \
    --num-nodes 15 \
    --max-iters 25 \
    --max-opt-steps 15 \
    --optimizer eigenvector_follow \
    --rtype 2 \
    --output-dir runs/diels_alder \
    --ID 1
```

`run_se_gsm.py` writes both raw pyGSM output and exported report artifacts in
the chosen run directory:

- `summary.json`
- `trajectory.sdf`
- `trajectory.gif`
- raw pyGSM files such as `grown_string_*.xyz`, `TSnode_*.xyz`, and `scratch/`

For the full CLI surface:

```bash
python run_se_gsm.py --help
```

## Python API

### Build the dedicated pyGSM adapter

```python
from reactip import ReactIPCalculator
from reactip.se_gsm import build_reactip_lot

calc = ReactIPCalculator("models/model_e1f9_l2_f32.nequip.zip", device="cpu")
lot = build_reactip_lot(
    geom=reactant_geom,
    calculator=calc,
    charge=0,
    multiplicity=1,
    adiabatic_state=0,
)
```

### Run SE-GSM from a model path

```python
from reactip.se_gsm import run_se_gsm

result = run_se_gsm(
    model_path="models/model_e1f9_l2_f32.nequip.zip",
    xyz_file="reactant.xyz",
    driving_coords="isomers.txt",
    device="cpu",
)
```

### Run SE-GSM with a prebuilt calculator

```python
from reactip import ReactIPCalculator
from reactip.se_gsm import run_se_gsm_with_calculator

calc = ReactIPCalculator("models/model_e1f9_l2_f32.nequip.zip", device="cpu")
result = run_se_gsm_with_calculator(
    calc,
    xyz_file="reactant.xyz",
    driving_coords="isomers.txt",
)
```

Returned SE-GSM metadata includes:

- `status`
- `converged`
- `has_ts`
- `npeaks`
- `ran_out`
- `end_early`
- `ts_node`, `ts_energy`, and `delta_e` only when a unique TS is actually present

For a reproducible named example plus exported trajectory artifacts, use
`run_se_gsm.py` with the bundled benchmark inputs under
`examples/benchmark_cases/`.

## Bundled input data

The repository includes input files that you can use directly with the two
CLI tools or the Python API:

- `examples/benchmark_cases/`
  - public benchmark inputs
- `examples/seed_structures/`
  - supplemental XYZ structures for calculator checks or future SE-GSM work
- `examples/exploratory_cases/`
  - research-only exploratory inputs

Current benchmark case:

- `examples/benchmark_cases/butadiene_ethylene_diels_alder__C6H10/reactant.xyz`
- `examples/benchmark_cases/butadiene_ethylene_diels_alder__C6H10/isomers.txt`

Current exploratory case:

- `examples/exploratory_cases/2_4_diphenylpentane__C17H20/reactant.xyz`

## State support

The interface accepts `charge`, `multiplicity`, and `adiabatic_state`, but the
current runtime support is intentionally restricted to `(0, 1, 0)`. Any other
state request fails fast.
