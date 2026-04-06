# NequIP Official Integration Notes
## Reference + Ideas for SE-GSM Integration

Archived from: `docs/integrations/` in the NequIP repository (v0.15.0)
Purpose: local reference + extracted learnings for future SE-GSM/workflow integration

---

## Part 1: Official ASE Integration (verbatim)

> Source: `docs/integrations/ase.md`

### Introduction

The [Atomic Simulation Environment (ASE)](https://wiki.fysik.dtu.dk/ase/) is a popular Python package
providing a framework for working with atomic data, reading and writing common formats,
and running various simulations and calculations.
NequIP provides the `NequIPCalculator` for integration with ASE.

### Creating an ASE Calculator

Three steps:

1. **Start with a trained model** — checkpoint file (`.ckpt`) or packaged model (`.nequip.zip`)

2. **Compile the model for ASE**:

```bash
nequip-compile \
  path/to/model.ckpt \
  path/to/compiled_model.nequip.pt2 \
  --device cuda \
  --mode aotinductor \
  --target ase
```

> The device specified during compilation should match the device you'll use with the calculator.

3. **Create the ASE calculator**:

```python
from nequip.ase import NequIPCalculator   # newer; in v0.15.0 use nequip.ase.nequip_calculator

calculator = NequIPCalculator.from_compiled_model(
    compile_path="path/to/compiled_model.nequip.pt2",
    device="cuda",
)
```

### Mapping types from NequIP to ASE

ASE uses atomic numbers/chemical symbols. NequIP can use arbitrary alphanumeric names.
If `chemical_symbols` is not given, nequip assumes type names ARE chemical symbols (with warning).

```python
calculator = NequIPCalculator.from_compiled_model(
    compile_path="path/to/compiled_model.nequip.pt2",
    device="cuda",
    chemical_symbols={"H": "myHydrogen", "C": "someCarbonType"}  # custom mapping
)
```

### Units from NequIP to ASE

> "The ASE convention uses eV energy units and Å length units while the NequIP framework follows the
> internally consistent units of the underlying dataset."

Use `energy_units_to_eV` and `length_units_to_A` for conversions.

### Example: Energy-Volume Curve

```python
from ase.build import bulk
import numpy as np
import matplotlib.pyplot as plt
from nequip.ase import NequIPCalculator
import torch

calculator = NequIPCalculator.from_compiled_model(
    compile_path="path/to/compiled_model.nequip.pt2",
    chemical_symbols=["Si"],
    device="cuda" if torch.cuda.is_available() else "cpu",
)

scaling_factors = np.linspace(0.95, 1.05, 10)
volumes, energies = [], []

for scale in scaling_factors:
    scaled_si = bulk("Si", crystalstructure="diamond", a=5.43 * scale, cubic=True)
    scaled_si *= (3, 3, 3)   # 216-atom supercell
    scaled_si.calc = calculator
    volumes.append(scaled_si.get_volume())
    energies.append(scaled_si.get_potential_energy())

plt.plot(volumes, energies, marker="o", label="E-V Curve")
```

### Example: Structural Relaxations (production pattern)

Key choices:
- **GOQN** optimizer (faster than FIRE — see SI of https://arxiv.org/abs/2412.19330)
- **FrechetCellFilter** for volume relaxation (recommended over ExpCellFilter)
- **Explosion detection**: break if `max(|F|) > 1e6 eV/Å`
- Use `optimizer.irun()` (generator interface) for per-step control

```python
from ase.filters import ExpCellFilter, FrechetCellFilter
import ase.optimize as opt
from nequip.ase import NequIPCalculator
import numpy as np, torch

compile_path = "path/to/compiled_model.nequip.pt2"
ase_optimizer = "GOQN"      # faster than "FIRE" from tests
ase_filter    = "frechet"   # recommended filter
max_steps     = 500
force_max     = 0.05        # eV/Å

calculator = NequIPCalculator.from_compiled_model(
    compile_path=compile_path,
    device="cuda" if torch.cuda.is_available() else "cpu",
)

optimizer_dict = {
    "GOQN":           opt.GoodOldQuasiNewton,
    "BFGS":           opt.BFGS,
    "LBFGSLineSearch": opt.LBFGSLineSearch,
    "FIRE":           opt.fire.FIRE,
    "FIRE2":          opt.fire2.FIRE2,
    # ... others
}
filter_cls  = FrechetCellFilter  # or ExpCellFilter
optim_cls   = optimizer_dict[ase_optimizer]

for atoms in structures_to_relax:
    atoms.calc = calculator
    atoms_filtered = filter_cls(atoms)
    with optim_cls(atoms_filtered, logfile="/dev/null") as optimizer:
        for _ in optimizer.irun(fmax=force_max, steps=max_steps):
            forces = atoms_filtered.get_forces()
            if np.max(np.linalg.norm(forces, axis=1)) > 1e6:
                raise RuntimeError("Forces are exorbitant, exploding relaxation!")
    energy = atoms_filtered.get_potential_energy()
```

---

## Part 2: Official LAMMPS Integration (verbatim)

> Source: `docs/integrations/lammps/`

LAMMPS is a production-grade MD engine. NequIP provides **two** LAMMPS integrations:

| Feature | `pair_nequip_allegro` | ML-IAP |
|---|---|---|
| LAMMPS compilation | [`pair_nequip_allegro` repo](https://github.com/mir-group/pair_nequip_allegro) | Custom build with Kokkos+CUDA |
| Multirank | `pair_nequip`: single rank only; `pair_allegro`: multirank | multirank |
| Model prep | `nequip-compile` | `nequip-prepare-lmp-mliap` |
| Acceleration | Allegro: TritonContracter | OpenEquivariance, CuEquivariance, Triton |

### pair_nequip_allegro

Compile for LAMMPS:

```bash
# TorchScript
nequip-compile path/to/ckpt path/to/model.nequip.pth --device cuda --mode torchscript

# AOTInductor
nequip-compile path/to/ckpt path/to/model.nequip.pt2 \
    --device cuda --mode aotinductor --target [pair_nequip|pair_allegro]
```

> `nequip-compile` should be performed on the **same machine** as the LAMMPS simulation.

### ML-IAP (beta)

Prepare model:

```bash
nequip-prepare-lmp-mliap ckpt_or_package.ckpt output.nequip.lmp.pt \
    --modifiers enable_OpenEquivariance
```

LAMMPS script:

```lammps
units         metal
boundary      p p p
atom_style    atomic
newton        on
pair_style    mliap unified output.nequip.lmp.pt 0
pair_coeff    * * H C N O F S Cl Br
```

Run with Kokkos GPU:

```bash
srun -n 1 lmp -in in.lammps -k on g 1 -sf kk -pk kokkos newton on neigh half
```

---

## Part 3: Official OpenMM Integration (summary)

> Source: `docs/integrations/openmm.md`

Via [OpenMM-ML](https://github.com/openmm/openmm-ml):
- Uses raw `.ckpt` files directly — **no compilation needed**
- Package files (`.nequip.zip`) not yet supported
- Compilation/GPU kernel modifiers not yet available for this path

---

## Part 4: Learnings and Ideas for SE-GSM Integration

### What is SE-GSM?

Single-Ended Growing String Method: searches for reaction paths and transition states.
At each string node: call backend → get energy (scalar) + gradient (N×3).
Currently uses DFT as backend. We want to replace with `ReactIPCalculator`.

---

### Key Learnings from Official Integration Docs

#### 1. GOQN > FIRE for geometry optimization

The official docs explicitly recommend **GOQN (GoodOldQuasiNewton)** over FIRE:

> "GOQN is faster than FIRE from tests; see SI of https://arxiv.org/abs/2412.19330"

**Implication for SE-GSM**: If SE-GSM performs local energy minimisation at each
string node (constrained or unconstrained), switch from BFGS/FIRE to GOQN.
Also note the newer faster GOQN MR: https://gitlab.com/ase/ase/-/merge_requests/3570

#### 2. Per-step explosion detection is critical for production

```python
for _ in optimizer.irun(fmax=force_max, steps=max_steps):
    forces = atoms.get_forces()
    if np.max(np.linalg.norm(forces, axis=1)) > 1e6:
        raise RuntimeError("Forces are exorbitant, exploding relaxation!")
```

**Implication for SE-GSM**: Wrap every MLIP call with a force magnitude check.
If forces exceed a threshold (e.g., 100 eV/Å for molecular systems), the geometry
is unphysical (e.g., atoms too close) — flag the node and either skip it or trigger
re-interpolation. This prevents GSM from hanging on bad geometries.

#### 3. Use `optimizer.irun()` not `optimizer.run()` for per-step control

`irun()` is a generator that yields after each step, giving access to forces at
every iteration. `run()` runs to completion without hooks.

**Implication for SE-GSM**: Use `irun()` in the GSM driver when doing local minimisation
so you can log per-step data, implement early stopping, and check for explosions.

#### 4. FrechetCellFilter for periodic-system relaxation

For any GSM run on periodic systems (crystalline reaction paths), use `FrechetCellFilter`
rather than `ExpCellFilter` for volume relaxation.

#### 5. `chemical_symbols` should always be specified explicitly

The warning `"Trying to use model type names as chemical symbols"` shows up when
`chemical_symbols` is not passed. For our 8-element model, always pass:

```python
ReactIPCalculator("model.nequip.pt2",
               chemical_symbols=["H","C","N","O","F","S","Cl","Br"])
```

This also silences warnings during GSM runs (which produce a lot of output).

#### 6. Device selection should be runtime-detected, not hard-coded

The official docs always use:
```python
device="cuda" if torch.cuda.is_available() else "cpu"
```

**Implication for SE-GSM integration**: Don't hard-code `device="cuda"` — detect at runtime
so the same SE-GSM script works on workstations (CPU) and HPC nodes (GPU).

#### 7. Compile on the same machine you'll run on (LAMMPS rule, also good practice for ASE)

AOTInductor-compiled models (`.nequip.pt2`) are optimised for the specific CPU/GPU
microarchitecture they were compiled on. TorchScript (`.nequip.pth`) is more portable.

**Implication for SE-GSM deployment**: When shipping to collaborators' clusters, either:
- Ship TorchScript `.nequip.pth` (portable, safe)
- Or include a `recompile.sh` that runs `nequip-compile` on their machine

#### 8. LAMMPS for high-throughput sampling (future)

Once the reaction mechanism is known from SE-GSM, LAMMPS + `pair_nequip` or `pair_allegro`
with Kokkos GPU acceleration is the right tool for:
- NVT/NPT MD on the product/reactant basins
- Free energy calculations
- Rate constant estimation

Use `--mode aotinductor --target pair_nequip` (for NequIP GNN) or `--target pair_allegro`
(for the Allegro model) during compilation. **Multirank works for `pair_allegro` only**
(not `pair_nequip`) — relevant for large-scale LAMMPS jobs.

#### 9. OpenMM for reaction path MD with explicit solvent (future)

If the reaction occurs in solution, OpenMM-ML with `.ckpt` is the simplest integration
(no compilation step). Useful for:
- Running NVT trajectories from SE-GSM-found transition state
- Checking solvent effects on the barrier
- Generating training data around the TS (for active learning)

#### 10. torch-sim for batched, GPU-native SE-GSM (future)

torch-sim is a PyTorch-native MD engine. In principle one could run all GSM string nodes
simultaneously as a batch on a single GPU. This could dramatically speed up SE-GSM
because the bottleneck is the N_nodes sequential calls to the quantum chemistry backend.

**Potential architecture**: Collect all N string-node geometries → batch them → single
forward pass through the model → extract energies and forces for all nodes at once.
NequIP supports batched inference (AtomicDataDict has a BATCH_KEY). Currently our
`ReactIPCalculator` is single-structure only — a `calculate_batch()` method could be added.

---

### Concrete Next Steps for SE-GSM Integration

| Priority | Action | Benefit |
|---|---|---|
| P0 | Replace hard-coded `device="cuda"` with runtime detection | Works on any machine |
| P0 | Pass `chemical_symbols=["H","C","N","O","F","S","Cl","Br"]` explicitly | Silences warnings |
| P1 | Add force explosion check in SE-GSM node eval | Prevents hanging on bad geometries |
| P1 | Switch local minimiser to GOQN | Faster per-node relaxation |
| P1 | Use `optimizer.irun()` in node relaxation | Per-step logging + early stopping |
| P2 | Add `ReactIPCalculator.calculate_batch()` | Batch all string nodes in one GPU call |
| P3 | Investigate LAMMPS for production MD on found paths | 10-100× faster than ASE MD |
| P3 | Investigate OpenMM-ML for solution-phase TS validation | Solvent effects |
