# ReactIP Deployment Guide

## Model Lifecycle

```
Training server                          Target server
─────────────────                        ─────────────
best.ckpt
    │  nequip-package build
    ▼
model.nequip.zip  ──── transfer ────►  model.nequip.zip
    (portable)          scp/rsync          │  nequip-compile
                                           ▼
                                       model_cuda_sm86.nequip.pt2
                                           (device-specific)
```

| Format | Extension | Portable? | GPU-specific? | Needs training env? |
|--------|-----------|-----------|---------------|---------------------|
| Checkpoint | `.ckpt` | No | No | Yes (lightning, hydra, data) |
| Package | `.nequip.zip` | **Yes** | No | No |
| Compiled | `.nequip.pt2` | No | **Yes** (per SM arch) | No |

The `.nequip.zip` package is the recommended distribution format. It bundles
model weights, code, and example data for compilation. End-users compile it
once for their specific GPU.

---

## Quick Start

### 1. Compile the model for your GPU

```bash
cd reactip/
sbatch -p g3090_veryshort --qos=veryshort deployment/compile_model.slurm
```

Or without SLURM:

```bash
cd reactip/
bash deployment/compile_from_package.sh models/model_e1f9_l2_f32.nequip.zip
```

This produces `models/model_e1f9_l2_f32_cuda_sm86.nequip.pt2` (filename
includes the GPU architecture automatically). Without arguments, it compiles
all `.nequip.zip` files found in `models/`.

### 2. Verify

```python
from reactip import ReactIPCalculator

calc = ReactIPCalculator("models/model_e1f9_l2_f32_cuda_sm86.nequip.pt2", device="cuda")
result = calc.calculate_xyz(
    open("examples/benchmark_cases/butadiene_ethylene_diels_alder__C6H10/reactant.xyz").read()
)
print("Energy:", result["energy"], "eV")
print("Forces:", result["forces"].shape)
```

### 3. Run SE-GSM

```python
from reactip.se_gsm import run_se_gsm

result = run_se_gsm(
    model_path="models/model_e1f9_l2_f32_cuda_sm86.nequip.pt2",
    xyz_file="examples/benchmark_cases/butadiene_ethylene_diels_alder__C6H10/reactant.xyz",
    driving_coords="examples/benchmark_cases/butadiene_ethylene_diels_alder__C6H10/isomers.txt",
    device="cuda",
)
```

---

## Compilation Details

### Why compile?

The `.nequip.zip` package contains a portable model graph. `nequip-compile`
converts it to native code (C++ and CUDA kernels) optimized for your
specific GPU — this is the only inference format supported in PyTorch >= 2.10.

### GPU architecture tags

AOTInductor output is tied to a specific CUDA compute capability:

| GPU | SM | Architecture tag |
|-----|----|----|
| A100 | 8.0 | `sm80` |
| RTX 3090 / A5000 / A6000 | 8.6 | `sm86` |
| RTX 4090 | 8.9 | `sm89` |
| H100 | 9.0 | `sm90` |

The compile script auto-detects the GPU and appends the tag to the filename.
The benchmark `config.py` auto-selects the correct model at runtime via
`torch.cuda.get_device_capability()`.

### Compile for CPU

```bash
bash deployment/compile_from_package.sh --device cpu models/model_e1f9_l2_f32.nequip.zip
```

Produces `models/model_e1f9_l2_f32_cpu.nequip.pt2`. CPU models are portable
across x86-64 machines (no SM dependency).

### Cross-version compatibility

- `.nequip.zip` packages survive PyTorch version upgrades (within reason)
- `.nequip.pt2` compiled models are tied to the **exact PyTorch version** and
  GPU architecture. If you upgrade PyTorch, recompile from the package.

---

## Supported Elements

All ReactIP models support: **H, C, N, O, F, S, Cl, Br**

---

## Inspect a Package

```bash
nequip-package info models/model_e1f9_l2_f32.nequip.zip
```

---

## Minimal Dependencies

For inference with a compiled `.nequip.pt2`:

```
torch (matching the version used to compile)
ase
nequip>=0.15.0
```

No lightning, hydra, wandb, or training data required.

---

## Troubleshooting

### `OSError: CUDA_HOME environment variable is not set`

```bash
export CUDA_HOME=/appl/cuda/cuda-12.8  # or your CUDA install path
```

### `RuntimeError: device mismatch`

The model was compiled for a different device. Recompile with `--device cuda`
or `--device cpu`.

### Model hangs on load / all jobs timeout

The `.nequip.pt2` was compiled with a different PyTorch version or on a
different GPU architecture. Recompile from the `.nequip.zip` package on the
target machine.

### `KeyError: 'H'` or missing element

This model supports: H, C, N, O, F, S, Cl, Br only.
