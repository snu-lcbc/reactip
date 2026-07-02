# pyGSM / SE-GSM Workflow and Parameters

A practical reference for the SE-GSM engine that ReactIP drives through
`run_se_gsm.py` and `reactip/se_gsm.py`: what the algorithm does, what every
important parameter means, **when to change it**, and the settings that are
empirically validated on this repo's benchmarks.

This is the engine-level companion to the task-level docs:
[`sampled_product_search.md`](sampled_product_search.md),
[`reaction_path_sampling.md`](reaction_path_sampling.md), and
[`sampled_search_workflow.md`](sampled_search_workflow.md).

All code citations are to the local pyGSM checkout
(`growth_string_method/pyGSM/pyGSM/…`) and the ReactIP adapter
(`reactip/…`) as of this writing; line numbers drift, so treat them as anchors.

---

## TL;DR — the settings that matter

| Flag | CLI default | **Recommended** | Why |
|------|-------------|-----------------|-----|
| `--coord-type` | **`DLC`** | **`DLC`** | DLC connects fragments into one internal-coordinate basis → far more stable for reactive / bimolecular systems. **~96% of validated runs use DLC.** |
| `--max-force` | **`500`** | **`500`** (eV/Å) | 100 clips legitimate steep-but-real gradients and aborts too eagerly. 500 is the benchmark standard. |
| `--num-nodes` | **`30`** | **`30`** | 20 often runs out before the string crosses the barrier. |
| `--sample-min-quality` | **`completed`** | `completed` | `converged` rejects the common `ran_out_with_ts_candidate` status and can filter out even a correct TS (the `0/10` failure mode). |
| `--optimizer` | `eigenvector_follow` | `eigenvector_follow` | Required to actually **find** a transition state (`--rtype 1/2`). |
| `--rtype` | `2` | `2` (find + climb) | Grow, locate the peak, then climb it to a saddle. |

> **As of the GUI-integration update, the CLI defaults now MATCH the validated
> configuration** (`--coord-type DLC --max-force 500 --num-nodes 30
> --sample-min-quality completed`). Previously the defaults were `TRIC`,
> `--max-force 100`, `--num-nodes 20`, `--sample-min-quality converged`, which is
> what produced the force-threshold errors and the `0/10` result in the GUI
> team's Diels–Alder test. Every benchmark, demo, and production run in `runs/`
> uses the DLC configuration. All values remain overridable on the command line;
> the recipes in [§7](#7-recommended-configurations-recipes) still apply.

Empirical footing (survey of 1032 run summaries in `runs/`):

- coordinate type: **DLC 992, TRIC 40** (the TRIC runs were early diagnostics, all at `max_force=100`, `num_nodes=4`).
- max_force: **500 → 842 runs**, 1000 → 74, 2000 → 74, 100 → 44.
- num_nodes: **30 → 961 runs**.

---

## 1. What SE-GSM is

**GSM** (Growing String Method) finds a minimum-energy reaction path — and the
transition state (TS) on it — between structures. **SE-GSM** is the
*single-ended* variant: you give it **only the reactant** plus a set of
**driving coordinates** describing which bonds should form/break, and it grows a
string of intermediate structures step-by-step toward the (initially unknown)
product, optimizing each new node under a constraint along the growth direction.
Once the string crosses the energy peak it climbs that peak to a true first-order
saddle.

Contrast with **DE-GSM** (double-ended, `de_gsm.py`), which needs both reactant
*and* product geometries. SE-GSM is what ReactIP uses because in a discovery
setting we don't know the product yet — that's the whole point.

### The ReactIP → pyGSM call stack

```
run_se_gsm.py (CLI, reporting, sampling, safety knobs)
  └─ reactip/se_gsm.py :: run_se_gsm()          # the orchestration below
       ├─ build_reactip_lot()                   # MLIP as a pyGSM "level of theory"
       │    └─ ReactIPLoT (reactip/reactip_lot.py)  # ASE eV → Hartree, + force/E guards
       ├─ _build_internal_coordinates()         # DLC / TRIC / HDLC  ← §3
       ├─ Molecule.from_options(Form_Hessian=…) # reactant molecule + (approx) Hessian
       ├─ _build_optimizer()                    # eigenvector_follow / lbfgs  ← §4
       ├─ opt.optimize(reactant, opt_steps=100) # pre-optimize reactant (unless --no-pre-opt)
       ├─ SE_GSM.from_options(DQMAG_MAX, BDIST_RATIO, CONV_TOL, ADD_NODE_TOL, …)  ← §5
       ├─ gsm.go_gsm(max_iters, max_opt_steps, rtype)   # grow → optimize → climb
       └─ _analyze_gsm_result(gsm)              # status, TSnode, barriers, ΔE
```

### The run, stage by stage

1. **Load MLIP once** and wrap it as a pyGSM level-of-theory (`ReactIPLoT`). It
   converts ASE units (eV, eV/Å) to pyGSM units (Hartree, Hartree/Bohr) and
   applies the safety guards ([§6](#6-safety-guards-the-loterror-family)).
2. **Build internal coordinates** for the reactant, adding the driving-coordinate
   bonds to the bond graph so they are representable ([§3](#3-coordinate-systems-dlc-vs-tric-vs-hdlc)).
3. **Pre-optimize the reactant** (100 steps) unless `--no-pre-opt`. A relaxed
   reactant is the energy zero for the whole path.
4. **Grow the string**: add one node at a time along the driving-coordinate
   tangent, constrained-optimizing each, until the growth-termination criterion
   fires ([§5](#5-growth-controls)).
5. **Optimize / climb**: reparameterize the grown string and, for `--rtype 1/2`,
   drive the highest node up to a saddle point ([§4](#4-optimizers-and-rtype)).
6. **Analyze**: locate `TSnode` (energy maximum), compute barrier and reaction
   energy, classify the run status.

### Driving coordinates (the `isomers.txt` file)

A driving coordinate names an internal coordinate and a direction of change. The
file format (parsed in `read_isomers_file`, `se_gsm.py`) is one per line, atom
indices **1-based**, optionally preceded by a `NEW` header:

```
NEW
ADD 6 4          # form a bond between atoms 6 and 4
ADD 5 1          # form a bond between atoms 5 and 1  (this pair = Diels–Alder)
BREAK 2 3        # break the bond between atoms 2 and 3
ANGLE 1 2 3 109  # drive an angle (deg)
TORSION 1 2 3 4 180
OOP 1 2 3 4 0    # out-of-plane
```

Multiple lines in one set = a **concerted** change (all driving coordinates
advance together). `ADD`/`BREAK` may take an optional target distance. The string
stops growing once these coordinates have reached their targets *and* the profile
is over the barrier (see [§5](#5-growth-controls)).

### Run status outcomes

`_analyze_gsm_result` (`se_gsm.py`) turns pyGSM's flags into a status string.
`has_ts = (npeaks == 1) and (not end_early)`.

| Status | Meaning | Usable? |
|--------|---------|---------|
| `converged_ts` | String converged **and** a unique TS was climbed. | ✅ best |
| `converged_no_ts` | String converged; no single clean peak (barrierless or multi-peak). | ✅ product ok |
| `ran_out_with_ts_candidate` | Hit `max-iters` but a TS peak *was* found (just not fully tightened). | ✅ often correct — **very common** |
| `ran_out_no_ts` | Hit `max-iters`, no clean peak. | ⚠️ product maybe |
| `ended_early_*` | Dissociated, all-uphill, or `< 3` nodes. | ❌ usually |
| `runtime_error_after_partial_output` | An exception aborted the run (force guard, pyGSM bug…). | ❌ |

Key subtlety for filtering: `ran_out_with_ts_candidate` is **not** in the
`completed` or `converged` quality tiers, only in `finite`. A correct TS found at
the node budget can be filtered out by `--sample-min-quality converged`. See
[§8](#8-troubleshooting).

---

## 2. How the pieces fit — the two run modes

`run_se_gsm.py` has two modes:

- **Direct mode** (`--isomers file`): run **one** known driving-coordinate set.
  Deterministic. Use when you know which bonds react.
- **Sampling mode** (`--sample-products`): auto-enumerate many candidate driving
  coordinates from the reactant's bond graph, run one SE-GSM each, rank products
  by Boltzmann population, optionally recurse. Use for discovery. Details in
  [`sampled_product_search.md`](sampled_product_search.md).

Both modes feed the **same SE-GSM engine** and honor the same engine parameters
below. Sampling mode simply calls the engine many times and tolerates individual
failures — per-candidate failures are expected and do not stop the search.

---

## 3. Coordinate systems: DLC vs TRIC vs HDLC

Internal coordinates (bonds, angles, dihedrals) optimize molecular geometries far
better than raw Cartesians. pyGSM builds **delocalized** internals (linear
combinations of primitives via the G-matrix). The `--coord-type` flag chooses how
*multiple fragments* and *global position* are represented. In the ReactIP adapter
(`se_gsm.py::_build_internal_coordinates`) the mapping is:

```python
connect  = (coordinate_type == "DLC")    # add inter-fragment bonds
addtr    = (coordinate_type == "TRIC")   # add per-fragment translation+rotation
addcart  = (coordinate_type == "HDLC")   # add Cartesians for all atoms
```

### DLC — Delocalized Internal Coordinates *(recommended)*

- `connect=True` builds a **minimum spanning tree over all atoms and adds the
  missing inter-fragment bonds** to the topology (`primitive_internals.py`
  `makePrimitives`/`newMakePrimitives`). The result: separate molecules are
  stitched into **one connected graph** with a **single unified delocalized
  basis** — bonds, angles, dihedrals only; **no** translation/rotation or
  Cartesian primitives.
- **Why it wins for reactions:** the reacting fragments are coupled through real
  distance/angle coordinates, so the optimizer has a well-conditioned, physically
  damped metric. Fragments cannot drift into each other "for free," which is
  exactly the collapse that produces the force-guard `LoTError`.
- **When to use:** essentially always for ReactIP — single molecules, bimolecular
  reactions, cluster reactions. It is the validated default (992/1032 runs).

### TRIC — Translation-Rotation Internal Coordinates

- `addtr=True` gives **each fragment its own 3 translation + 3 rotation
  coordinates** plus its intramolecular internals; fragments stay **separate** in
  coordinate space (single atoms fall back to Cartesians).
- TRIC (Wang & Song) is designed for multi-molecule systems and is excellent for
  *non-reactive* optimization/MD. **But in this reactive SE-GSM + MLIP setting it
  is empirically fragile:** under a bond-forming driving constraint the free
  translation/rotation DOFs let fragments approach with little restoring force,
  overshoot into collisions, and trip the MLIP out-of-domain guards. The
  per-fragment "block" construction is also the more bug-prone pyGSM path.
- **When to use:** not recommended here. It is the current CLI *default* only for
  historical reasons — override it with `DLC`.

### HDLC — Hybrid Delocalized Internal Coordinates

- `addcart=True` adds **Cartesian x,y,z for every atom** alongside the internal
  coordinates.
- More degrees of freedom and more robust to pathological internal-coordinate
  breakdowns (e.g. near-linear angles), at the cost of efficiency and step
  quality.
- **When to use:** a **fallback** only — if a specific system repeatedly fails
  DLC during coordinate construction or B-matrix inversion (`LinAlgError:
  Singular matrix`, orthonormality errors). Try `HDLC` for that case; don't use
  it by default.

### A shared caveat: `get_hybrid_indices`

`primitive_internals.py::get_hybrid_indices` runs **unconditionally** during
coordinate construction (before the DLC/TRIC/HDLC branch), so **no coordinate type
avoids it.** It partitions atoms by fragment and fails
(`ValueError: x not in list`, masked as a bare `RuntimeError`) when a reactant's
**fragments have interleaved atom numbering** (e.g. the two molecules' atoms are
not in contiguous blocks) and the driving coordinates don't merge them into one
fragment. It is rare (2 distinct cases in the 222-case Zimmerman benchmark, both
under DLC) and is best avoided by **numbering each molecule's atoms contiguously
in the input XYZ** (all of molecule A, then all of molecule B). See
[§8](#8-troubleshooting).

| | **DLC** | **TRIC** | **HDLC** |
|---|---|---|---|
| fragment handling | connected via MST bonds | separate + 6 DOF each | separate + Cartesians |
| primitives | bonds/angles/dihedrals (unified) | intra + translation/rotation | intra + Cartesian(all atoms) |
| best for | **reactions, bimolecular** | non-reactive multi-molecule | robustness fallback |
| ReactIP use | **default (recommended)** | avoid | last resort |

---

## 4. Optimizers and `rtype`

`--optimizer` selects the per-node optimizer; `--rtype` selects how hard the
engine works to find the TS.

### eigenvector_follow *(recommended)*

- A quasi-Newton optimizer that, in TS mode, **follows the lowest Hessian
  eigenvector uphill** (`TS_eigenvector_step`) to climb to a first-order saddle
  while minimizing all other modes.
- **This is the only optimizer that can locate a transition state.** If you want
  `--rtype 1` or `--rtype 2` to yield a real TS, you must use
  `eigenvector_follow`.
- Maintains an **approximate Hessian via background BFGS/Bofill updates**
  (`update_hess_in_bg=True`) — no expensive exact Hessian per step.

### lbfgs

- A limited-memory quasi-Newton **pure minimizer**. Builds an inverse-Hessian
  approximation from step history and always descends. **Cannot climb to a TS.**
- **When to use:** only for `--rtype 0` (path/minimization without TS search), or
  as a robustness experiment when eigenvector_follow behaves badly. For normal TS
  searches, keep `eigenvector_follow`.

### `--rtype` (climbing behavior in `go_gsm`)

| rtype | Behavior | Use when |
|-------|----------|----------|
| `0` | No climbing, no TS search — just grow/optimize the path. | You only want a product/endpoint, fastest. |
| `1` | **Climb** the peak with the climbing-image step, no exact saddle finder. | Quick approximate barrier. |
| `2` | **Find + climb**: climbing image *and* exact eigenvector saddle search. | **Default** — you want a real TS + barrier. |

### Optimizer internals (fixed by the ReactIP wrapper)

`_build_optimizer` overrides these regardless of CLI (`se_gsm.py`):

- **`DMAX = 0.1`** — max step size per optimizer iteration (steps larger than this
  are scaled down). Small = conservative and stable; this deliberately trades
  speed for not blowing up floppy/reactive geometries.
- **`Linesearch = "NoLineSearch"`** — take the quasi-Newton step directly.
- **`conv_gmax = 100`, `conv_Ediff = 100`** — effectively **disabled** so the
  *string-level* convergence (`CONV_TOL`, [§5](#5-growth-controls)) governs, not
  the per-node optimizer's own gmax/energy criteria.
- `SCALE_CLIMB = 1.0` — scales the climbing step (`max_step = 0.05/SCALE_CLIMB`);
  larger = more cautious climbing.

---

## 5. Growth controls

These govern how the string grows toward the product. **Three of them
(`DQMAG_MAX`, `BDIST_RATIO`, `ADD_NODE_TOL`) are fixed in the ReactIP wrapper and
not exposed on the CLI** — listed for understanding/tuning in code.

| Parameter | pyGSM default | ReactIP value | Controls |
|-----------|---------------|---------------|----------|
| `--num-nodes` (`nnodes`) | 1 (capacity) | 20 CLI / **30 rec.** | Max nodes in the string. Too few → string "runs out" before crossing the barrier (`ran_out_*`). 30 is standard. |
| `--max-iters` | 50 | 100 | Outer loop: max growth *and* optimization iterations. |
| `--max-opt-steps` | 10 | 20 | Inner loop: optimizer steps **per node** during growth. |
| `--conv-tol` (`CONV_TOL`) | 5e-4 | 5e-4 | String/TS convergence on node gradient RMS (Hartree/Bohr). Lower = tighter TS, slower. |
| `DQMAG_MAX` / `DQMAG_MIN` | 0.8 / 0.2 | 0.8 / 0.2 *(fixed)* | Step size when adding a node along the driving tangent. Scales between MIN (far from target) and MAX (near target). |
| `BDIST_RATIO` | 0.5 | 0.5 *(fixed)* | Growth stops once the driving-coordinate "bond distance" has shrunk to `(1 − ratio)` of its initial value — i.e. how far to push before declaring the string grown. |
| `ADD_NODE_TOL` | 0.1 | 0.01 *(fixed)* | Frontier node must reach this gradient RMS before the next node is added. Tighter (0.01) → cleaner, more expensive growth. |

**Growth termination** (`se_gsm.py::check_if_grown`): the string is "grown" when it
is **past the TS** (`past_ts()` detects the energy going over the hill) **and** the
bond-distance criterion (`BDIST_RATIO`) is met — or on degenerate profiles
(all-uphill / dissociation), which end the run early.

`max-iters` vs `max-opt-steps`: `max-iters` is the number of *outer* growth/optimize
cycles; `max-opt-steps` is how many optimizer steps each node gets *within* a cycle.
Raising `max-iters` helps a string that keeps running out; raising `max-opt-steps`
helps individual nodes that aren't tightening.

---

## 6. Safety guards (the `LoTError` family)

ReactIP-specific, in `reactip/mlip_calculator.py::_validate`. An MLIP can return
garbage (huge forces, absurd energies, NaN) for geometries outside its training
domain — e.g. two atoms nearly on top of each other. The guards catch that and
abort the node instead of propagating nonsense.

| Flag | Default | Recommended | Fires when |
|------|---------|-------------|-----------|
| `--max-force` | 100 | **500** | Any force **component** `> threshold` (eV/Å). |
| `--max-abs-energy` | 10000 | 10000 | `|energy| > threshold` (eV). |
| (always on) | — | — | Any NaN/Inf in energy or forces. |

A tripped guard raises `CalculationFailed` → `ReactIPLoT` re-raises as `LoTError`
→ the run ends as `runtime_error_after_partial_output`.

**Interpreting it:** a firing guard almost always means *the driving coordinate is
chemically wrong for that geometry* (the string drove atoms into a collision), not
that the threshold is too low. In sampling mode, **many `LoTError`s are normal and
expected** — the guard is filtering out the junk coordinates. Raising `--max-force`
to 1000–2000 helps only the rare case where a *real* steep-but-valid gradient is
being clipped; pushing it much higher just lets bad geometries produce meaningless
products. Default to 500; raise only deliberately.

---

## 7. Recommended configurations (recipes)

### A. Known single reaction (you know which bonds react)

```bash
python run_se_gsm.py \
  --model models/model_e1f9_l2_f32.nequip.zip \
  --xyz     path/to/reactant.xyz \
  --isomers path/to/isomers.txt \
  --coord-type DLC --max-force 500 --num-nodes 30 \
  --optimizer eigenvector_follow --rtype 2 \
  --output-dir runs/my_reaction
```

### B. Product discovery (you don't know the product) — sampling

```bash
python run_se_gsm.py \
  --model models/model_e1f9_l2_f32.nequip.zip \
  --xyz path/to/reactant.xyz \
  --sample-products --sample-count 20 --sample-iterations 1 \
  --coord-type DLC --max-force 500 --num-nodes 30 \
  --optimizer eigenvector_follow --rtype 2 \
  --sample-min-quality completed \
  --output-dir runs/my_discovery
```
See [`sampled_product_search.md`](sampled_product_search.md) for the full sampling
knob set (score mode, temperature, multi-step iterations, TS verification).

### C. Hard / floppy / very-multi-fragment system

- Bump `--max-force 1000` (only if you see force-guard aborts on geometries you
  believe are real).
- Try `--coord-type HDLC` **only** if DLC throws coordinate-construction errors
  (`Singular matrix`, orthonormality, `get_hybrid_indices`).
- Raise `--num-nodes 40` and/or `--max-iters 150` if the string keeps `ran_out`.

### D. Tight TS for publication / verification

- Keep A/B settings, add `--sample-verify-ts` (sampling) which runs an MLIP
  finite-difference Hessian and confirms exactly one imaginary frequency.
- Then re-optimize the promising TS at the intended QM level (see
  [`nequip_integration_notes.md`](nequip_integration_notes.md) and the ORCA
  toolchain) — MLIP barriers are for screening, not final numbers.

---

## 8. Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `LoTError: max force component = … exceeds threshold 100.0` | Wrong/tight settings driving atoms into a collision, or MLIP out-of-domain. | Use `--coord-type DLC --max-force 500`. In sampling, expected for bad coords — ignore individual ones. |
| `LoTError: |energy| = … exceeds threshold 10000.0` | Collapsed/dissociated geometry; MLIP extrapolating wildly. | Same as above; the candidate is genuinely bad. |
| `RuntimeError:` (empty) from `get_hybrid_indices` | Interleaved multi-fragment atom numbering in the input; coordinate construction can't partition fragments. | **Renumber the XYZ so each molecule's atoms are contiguous** (all of A, then all of B). Not fixed by changing coord type. |
| `TypeError: only 0-dimensional arrays can be converted to Python scalars` (`get_constraint_steps`) | pyGSM × NumPy 2 bug in the climbing step (`print(" gts %1.4f" % gts)` with `gts` a length-1 array). | Rare once DLC/500 keep geometries sane; it's an upstream pyGSM bug. Pin the validated pyGSM checkout / NumPy build (see note below). |
| `LinAlgError: Singular matrix` / orthonormality error | Internal-coordinate B-matrix degeneracy (near-linear angle, bad geometry). | Pre-optimize the reactant (don't use `--no-pre-opt`); try `--coord-type HDLC` for that case. |
| Status `ran_out_*`, no TS | `num-nodes` too small or `max-iters` too small. | `--num-nodes 30/40`, `--max-iters 100/150`. |
| Correct TS found but **0 ranked candidates** in sampling | `--sample-min-quality converged` rejects `ran_out_with_ts_candidate`. | Step down to `--sample-min-quality completed` (or `finite` on hard cases). |
| `null` everywhere in a candidate's `candidates[]` entry | That candidate errored; the flat `candidates[]` list is the audit trail. | Read `overall_top_candidates[]` for successful, ranked results. |

> **pyGSM reproducibility.** ReactIP expects a specific local pyGSM checkout on
> `PYTHONPATH` (`export REACTIP_PYGSM_DIR=…/growth_string_method/pyGSM`), **not**
> a stock `pip install`. The two pyGSM bugs above live in fragile code paths that
> differ between builds — always confirm which pyGSM (path/commit) and which NumPy
> are active. `summary.json → runtime_provenance` records the exact pyGSM package
> path used for each run.

---

## 9. What to expect (realistic framing)

SE-GSM + MLIP is a **screening** engine. Even with the validated
`DLC / max_force=500 / num_nodes=30` configuration, the 222-case Zimmerman
benchmark gives:

- **~32%** of driving coordinates yield a TS (`has_ts`),
- **~54%** reach a product whose bond graph matches the reference,
- **~40%** end in a `runtime_error` (mostly force/energy guards on bad coords).

That is **normal**. The workflow is built around it: sample many driving
coordinates, let the bad ones fail fast, and rank the survivors. A single "wrong"
driving coordinate failing tells you nothing about your build — only the
*aggregate* over many candidates, and the *correct* coordinate for a known
reaction, are meaningful. For a known reaction, use direct mode (recipe A) with
the correct `isomers.txt`; for discovery, use sampling (recipe B) and judge by the
ranked `overall_top_candidates`.

---

## 10. Parameter quick-reference (CLI)

Engine parameters exposed by `run_se_gsm.py` (sampling-only flags omitted — see
[`sampled_product_search.md`](sampled_product_search.md)):

| Flag | Default | Recommended | One-line meaning |
|------|---------|-------------|------------------|
| `--coord-type {TRIC,DLC,HDLC}` | TRIC | **DLC** | Internal-coordinate system ([§3](#3-coordinate-systems-dlc-vs-tric-vs-hdlc)). |
| `--optimizer {eigenvector_follow,lbfgs}` | eigenvector_follow | eigenvector_follow | Per-node optimizer; EF required for TS ([§4](#4-optimizers-and-rtype)). |
| `--rtype {0,1,2}` | 2 | 2 | No-climb / climb / find+climb. |
| `--num-nodes` | 20 | **30** | Max string nodes. |
| `--max-iters` | 100 | 100 | Outer growth/optimize iterations. |
| `--max-opt-steps` | 20 | 20 | Optimizer steps per node. |
| `--conv-tol` | 5e-4 | 5e-4 | TS/string gradient-RMS convergence. |
| `--max-force` | 100 | **500** | Force-magnitude safety cutoff (eV/Å). |
| `--max-abs-energy` | 10000 | 10000 | Absolute-energy safety cutoff (eV). |
| `--no-pre-opt` | off | off | Skip reactant pre-optimization (leave OFF). |
| `--charge` / `--multiplicity` / `--adiabatic-state` | 0 / 1 / 0 | — | Bookkeeping only; the MLIP has no spin/charge input (single-state PES). |
| `--device` | auto | `cuda` | Torch device. |

Fixed in the wrapper (edit `reactip/se_gsm.py` to change): `DQMAG_MAX=0.8`,
`DQMAG_MIN=0.2`, `BDIST_RATIO=0.5`, `ADD_NODE_TOL=0.01`, optimizer `DMAX=0.1`,
`Linesearch=NoLineSearch`, `conv_gmax=conv_Ediff=100` (disabled in favor of
`CONV_TOL`).

---

### See also

- [`sampled_product_search.md`](sampled_product_search.md) — sampling mode, ranking, JSON schema.
- [`reaction_path_sampling.md`](reaction_path_sampling.md) — scoring modes, populations, multi-step paths.
- [`sampled_search_workflow.md`](sampled_search_workflow.md) — Slurm arrays and reporting.
- [`nequip_integration_notes.md`](nequip_integration_notes.md) — the MLIP calculator and units.
