# ReactIP — Reactive Interatomic Potential

ReactIP is an inference package for NequIP/Allegro interatomic potential
models trained on Halo8, reaction-pathway datasets, including off-equilibrium
structures sampled along bond-breaking and bond-forming pathways ([read more](https://www.nature.com/articles/s41597-025-05944-3)), with an
ASE-native calculator and a
[pyGSM](https://github.com/ZimmermanGroup/pyGSM) SE-GSM runner by
[Zimmerman's Group](https://github.com/ZimmermanGroup/molecularGSM).
ReactIP is optimized primarily for GPU inference and target-specific
compilation, while packaged `.nequip.zip` models remain usable for CPU runs.

## Install

Requires Python >= 3.10. We recommend [`uv`](https://docs.astral.sh/uv/) for
fast, reproducible installs.

```bash
git clone https://github.com/snu-lcbc/reactip.git
cd reactip
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e .
```

With plain `pip`:

```bash
pip install --extra-index-url https://download.pytorch.org/whl/cu128 -e .
```

(`uv` reads the `[tool.uv.sources]` block in `pyproject.toml` and pulls the
CUDA 12.8 build of `torch` automatically; with plain `pip` you pass the
index URL explicitly.)

Dependencies (resolved automatically from `pyproject.toml`):

- `torch` (CUDA 12.8 build)
- `nequip` 0.17
- `ase` 3.28
- `numpy` 2.4
- `imageio` 2.37
- `matplotlib` 3.10
- `pyGSM` — patched fork at [`snu-lcbc/pyGSM`](https://github.com/snu-lcbc/pyGSM)

The commands below use `--device cpu` for a portable first run. Use
`--device cuda` on a machine with an NVIDIA GPU and driver.

## SE-GSM + ReactIP (MLIP Calculator)

```bash
python run_se_gsm.py \
    --model models/model_e1f9_l2_f32.nequip.zip \
    --xyz examples/benchmark_cases/butadiene_ethylene_diels_alder__C6H10/reactant.xyz \
    --isomers examples/benchmark_cases/butadiene_ethylene_diels_alder__C6H10/isomers.txt \
    --formula C6H10 \
    --device cuda \
    --num-nodes 30 \
    --max-iters 30 \
    --max-opt-steps 15 \
    --optimizer eigenvector_follow \
    --rtype 2 \
    --output-dir runs/diels_alder
```
This is a short installation smoke run that writes `summary.json`,
`trajectory.sdf`, and `trajectory.gif`. For production SE-GSM searches,
increase `--num-nodes`, `--max-iters`, and `--max-opt-steps` after confirming
the model and reaction remain within the calculator safety thresholds.

For the full CLI surface: `python run_se_gsm.py --help`

This SE-GSM program calls the MLIP single-point calculator internally to evaluate energies and forces.

### Sample and rank candidate products

Standalone feature documentation: [`docs/reaction_path_sampling.md`](docs/reaction_path_sampling.md)

Use `--sample-products` to generate candidate driving-coordinate sets from a
reactant, run SE-GSM for each candidate, and print Boltzmann-ranked products:

```bash
python run_se_gsm.py \
    --model models/model_e1f9_l2_f32.nequip.zip \
    --xyz examples/benchmark_cases/butadiene_ethylene_diels_alder__C6H10/reactant.xyz \
    --sample-products \
    --sample-count 10 \
    --sample-iterations 3 \
    --resample-top-k 3 \
    --print-top 5 \
    --sample-score-mode thermodynamic \
    --device cuda \
    --output-dir runs/sample_search
```

The reliability-critical settings — `--coord-type DLC`, `--max-force 500`,
`--num-nodes 30`, `--sample-min-quality completed` — are now the **defaults**, so
the command above uses them without needing the flags. (Before this update the
defaults were `TRIC` / `100` / `20` / `converged`, which is what produced the
force-threshold errors and `0/10` result in early tests.) All values remain
overridable; pass `--sample-min-quality converged` (or `ts`) to be stricter. The
default `--sample-score-mode thermodynamic` reports equilibrium-like product populations
with weights proportional to `exp(-dE / RT)`, where `dE` is the cumulative
product energy relative to the original root reactant. This cumulative score is
important for two-step and three-step searches because products sampled from
different parents do not share a local energy reference.

Use `--sample-score-mode kinetic` only when you want a TST-like path proxy based
on cumulative TS barriers; candidates without a unique TS are excluded from
kinetic ranking. Use `--sample-min-quality finite` only for debugging or smoke
runs because it can include early-ended strings. The default ADD search cutoff is
5.0 Å for heavy atoms, matching the MLIP neighborhood radius used in training as
a defensible upper bound for candidate generation; it is still a sampling
heuristic, not the MLIP message-passing cutoff itself. Two-ADD candidates do not
share a new-bond atom unless `--sample-allow-shared-add-atoms` is set.

Sampled searches write per-candidate `summary.json`, `isomers.txt`,
`reactant.xyz`, `product.xyz`, and a top-level `candidate_search_summary.json`;
add `--sample-export-artifacts` if you also want SDF/GIF artifacts for every
sampled candidate. The model is loaded once and reused across sampled candidates;
add `--sample-reload-model-each-candidate` only when debugging calculator state.

## ReactIP Single-point Calculator

`reactip_calculator.py` is a standalone program for single-point energy and force evaluation. It only requires molecular coordinates.

```bash
python reactip_calculator.py \
    --model models/model_e1f9_l2_f32.nequip.zip \
    --xyz examples/benchmark_cases/butadiene_ethylene_diels_alder__C6H10/reactant.xyz \
    --device cpu
```

For the full CLI surface: `python reactip_calculator.py --help`



## Model formats

`ReactIPCalculator` and `run_se_gsm.py` accept two model formats:

| Format | Extension | Load speed | Environment requirement |
| --- | --- | --- | --- |
| Compiled | `.nequip.pt2` / `.nequip.pth` | Fast | `torch` + `ase` + `nequip` |
| Raw checkpoint | `.ckpt` | Slower | Full training environment |

The `.ckpt` path loads the model directly for inference. For production use,
pre-compile with `nequip-compile`.

The default model package in this repository is:

- `models/model_e1f9_l2_f32.nequip.zip`

The `models/` directory contains packaged and compiled model artifacts. For
target-specific compilation workflows, see `deployment/README.md`,
`deployment/compile_from_package.sh`, and `deployment/compile_model.slurm`.

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
`run_se_gsm.py` with the benchmark inputs under `examples/benchmark_cases/`.

## Conventions

- `ReactIPCalculator` stays in ASE units: `eV` and `eV/Angstrom`.
- If you want standalone reporting in `kcal/mol`, convert a returned result
  dict explicitly with `ReactIPCalculator.convert_results_to_units(...)`.
- The supported SE-GSM path is `pyGSM` + `ReactIPLoT`.
- A default model is included at `models/model_e1f9_l2_f32.nequip.zip`.
  For any other model, pass `--model` or set `REACTIP_MODEL`.
- The interface accepts `charge`, `multiplicity`, and `adiabatic_state`, but
  the current runtime support is intentionally restricted to `(0, 1, 0)`. Any
  other state request fails fast.
