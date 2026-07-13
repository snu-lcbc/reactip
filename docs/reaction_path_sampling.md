# Reaction Path Sampling with ReactIP

This document describes the sampled reaction-path search implemented in
`run_se_gsm.py`. The feature uses a pretrained ReactIP MLIP model with SE-GSM to
sample possible product-forming driving coordinates, rank the candidates, and
optionally continue from the top products to build two-step or three-step paths.

For a shorter sendable README-style version, see
[`docs/sampled_product_search.md`](sampled_product_search.md).

For Slurm arrays, CSV tables, analysis plots, structure panels, and GIF reports,
see [`docs/sampled_search_workflow.md`](sampled_search_workflow.md).

## What it does

Given one reactant XYZ structure, sampled mode:

1. Generates candidate pyGSM driving-coordinate files from the current reactant.
2. Runs one SE-GSM calculation for each candidate.
3. Extracts a product endpoint from each successful candidate trajectory.
4. Ranks valid products using Boltzmann weights.
5. Prints the top candidates and writes a machine-readable search summary.
6. Optionally repeats the process from the top products for multi-step searches.

The main entry point is:

```bash
python run_se_gsm.py --sample-products ...
```

Normal single-path SE-GSM behavior is unchanged when `--sample-products` is not
used.

## Recommended command

Example production-style run on the Diels-Alder benchmark:

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

The reliability-critical settings `--coord-type DLC`, `--max-force 500`,
`--num-nodes 30`, and `--sample-min-quality completed` are now the **defaults**
(previously `TRIC` / `100` / `20` / `converged`), so they no longer need to be
passed explicitly. All remain overridable. This samples up to 10 candidate
reactions per frontier reactant. After each
iteration, the top 3 ranked product geometries become the reactants for the next
iteration. With the defaults above, the search can therefore explore up to three
reaction steps.

For a fast smoke run, reduce the search size and optimizer settings:

```bash
python run_se_gsm.py \
    --model models/model_e1f9_l2_f32.nequip.zip \
    --xyz examples/benchmark_cases/butadiene_ethylene_diels_alder__C6H10/reactant.xyz \
    --sample-products \
    --sample-count 1 \
    --sample-iterations 1 \
    --sample-modes two_add \
    --sample-seed 3 \
    --device cpu \
    --optimizer lbfgs \
    --num-nodes 4 \
    --max-iters 1 \
    --max-opt-steps 1 \
    --no-pre-opt \
    --output-dir /tmp/reactip_sampling_smoke
```

## How candidates are generated

Candidate generation is implemented in `reactip/sampling.py`. Two strategies
are available via `--sample-pool-strategy`:

### `rule_based` (default)

Valence-bounded, complexity-bounded, symmetry-reduced generation following the
ARD-GSM rules of the group's Dandelion prior work
(`generate_rule_based_pool`). A bond change is proposed only if:

- every atom whose coordination changes stays within its element connection
  limits (`CONNECTION_LIMITS_CLOSED_SHELL`, taken verbatim from ARD-GSM
  `limits.py`: H 1–1, C 2–4, N 1–3, O 1–2, F 1–1, S 1–4, Cl 1–1, Br 1–1,
  Li 0–1 — enforcing the minimum makes these neutral closed-shell bounds); and
- the step is elementary: at most `--sample-maxbreak` breaks (default 1),
  `--sample-maxform` forms (default 1), `--sample-maxchange` total changes
  (default 2).

Topologically equivalent atoms (methyl H's, phenyl ortho carbons) are found by
Weisfeiler–Lehman graph refinement and collapsed, so symmetry-redundant
coordinate sets are removed. **Hydrogen is included by default here** — the
pruned H pool is small and chemically meaningful (H-abstraction, H-shift).
Across 3–17-heavy-atom reactants this yields 15–118× fewer candidates than the
exhaustive enumerator with hydrogen (median 63×), while removing hypervalent /
under-coordinated products the enumerator would generate and waste an SE-GSM
run discarding.

For radical / combustion chemistry, `--sample-open-shell` switches to a relaxed
connection-limit table (minima dropped to 0) that admits homolysis and
O-centred-radical steps. **Note** the shipped MLIP is neutral-singlet only, so
radical energetics from it are extrapolative.

### `exhaustive`

The legacy geometric enumerator (`generate_driving_coordinate_pool`), kept for
benchmark reproducibility.

- Available sample modes are `add`, `break`, `exchange`, and `two_add`.
- Hydrogen atoms are excluded by default; use `--sample-include-hydrogen` to
  include them.
- The default ADD distance cutoff is 5.0 A. This is a candidate-generation
  heuristic chosen to avoid missing separated-reactant forming bonds; it is not
  itself a proof that the reaction is chemically valid.
- Two-ADD candidates avoid sharing a newly bonded atom by default. Use
  `--sample-allow-shared-add-atoms` only for exploratory searches.

SE-GSM still decides whether a candidate is usable. Invalid, failed, or
low-quality candidates are recorded in the summary and excluded from production
ranking by default.

### Sampled subset vs. exhaustive enumeration

By default the generated pool is **subsampled**: `--sample-count N` draws a
random (seeded) subset of `N` coordinate sets per reactant. This is the
"sampled" in sampled-product search and trades coverage for speed — the printed
populations are then relative to the sampled subset, not the full pool. To
reproduce the original RPS/Halo8 behavior, which enumerates *all* driving
coordinates from the reactant graph and runs every one, pass `--sample-all`
(it ignores `--sample-count` and runs the entire enumerated pool per reactant).
Exhaustive mode is much more expensive, especially with multiple iterations, so
it is best used with `--sample-iterations 1` for a single-step product search.
The full pool size for a given reactant is the number returned by
`generate_driving_coordinate_pool` in `reactip/sampling.py`.

## Ranking and populations

The printed population is normalized over ranked candidates with:

```text
population_i = exp(-dE_i / RT) / sum_j exp(-dE_j / RT)
```

The default score mode is:

```text
--sample-score-mode thermodynamic
```

In this mode, `dE` is the product energy relative to the original root reactant.
When the shared calculator is active (the default), this is computed from
**absolute MLIP energies**: `dE = E(product) - E(root reactant)`, evaluated on a
single consistent energy scale (`ranking_score_source = absolute_product_delta_e_vs_root`).
This is the correct reference for multi-step paths because each product is
re-optimized when it becomes the next reactant; summing per-step `dE` would drop
the relaxation energy at every hand-off. If absolute energies are unavailable
(e.g. `--sample-reload-model-each-candidate`), the code falls back to the
telescoped per-step sum (`cumulative_product_delta_e_stepwise`).

The alternative mode is:

```text
--sample-score-mode kinetic
```

This ranks by the **rate-limiting (largest single-step) TS barrier** along the
path — the rate-determining step governs the overall rate, not the sum of
barriers. It only ranks paths where SE-GSM found a unique TS at *every* step.
It is a screening proxy, not a full rate calculation; for a rigorous multi-step
rate, an energetic-span (Kozuch–Shaik) analysis on DFT-refined intermediates and
TSs is the appropriate follow-up.

### Out-of-domain guard

```text
--sample-max-abs-delta-e 500.0   # kcal/mol; 0 disables
```

A pretrained MLIP can return unphysical energies for geometries outside its
training distribution (collapsed or dissociated structures). Such a candidate
would otherwise have an enormous negative `dE`, capture ~100% of the Boltzmann
weight, and seed the next iteration. Candidates whose step or cumulative `|dE|`
exceeds this bound are excluded with reason `implausible ... dE ... (likely MLIP
out-of-domain)`. The default 500 kcal/mol is far above any real reaction energy
for the in-domain small molecules while still catching extrapolation failures.

### Product de-duplication

Different driving coordinates frequently converge to the **same** product. By
default such candidates are merged before the Boltzmann normalization (the
best-scoring representative is kept; the rest are recorded with reason
`duplicate product of <id>`). Merging uses a **symmetry-invariant canonical
bond signature** (`canonical_bond_signature` in `reactip/sampling.py`): each
product bond is keyed on the Weisfeiler–Lehman symmetry classes of its
endpoints, so two products that are the same species up to atom relabeling
(mirror images) are correctly recognized as identical — an index-based
signature misses these and double-counts them. This prevents a degenerate
product from inflating its apparent population. Disable with
`--sample-no-dedupe-products`.

Each ranked product also carries honest ranking-semantics fields:
`relative_stability_score` (the Boltzmann weight, named to signal it is not an
equilibrium concentration), `kinetics_status`
(`ts_verified` / `ts_geometric` / `no_ts_thermodynamic_only`), and
`is_fragmentation`. The run summary's `ranking_semantics` block documents what
the score omits (ZPE, thermal ΔG/entropy, fragmentation entropy, solvation),
the effective sampling temperature, and the subset normalization.

### Transition-state verification

```text
--sample-verify-ts
```

SE-GSM's `converged_ts` only means the optimized string had a single energy
peak. With `--sample-verify-ts`, each unique TS node is confirmed with an MLIP
finite-difference Hessian: the geometry is a genuine first-order saddle only if
it has exactly one imaginary frequency (above a 50 cm^-1 noise threshold). The
result is stored as `ts_imaginary_mode_count` / `ts_is_first_order_saddle`, and
`--sample-min-quality ts` then requires a verified saddle. This reproduces the
TS criterion used in the RPS/Halo8 reference pipelines at MLIP cost
(~6N gradient evaluations per TS).

Candidate quality is controlled by:

```text
--sample-min-quality {finite,completed,converged,ts}
```

Recommended default:

```text
--sample-min-quality completed
```

`completed` is the current default. It admits `completed_*` and `converged_*`
statuses, including the common `ran_out_with_ts_candidate` (a TS was found but
the string did not fully converge in the node budget) — which `converged` would
reject, and which was the cause of the `0/10` result in the GUI team's test. Use
`converged` or `ts` to be stricter. Use `finite` only for debugging or smoke
tests because it may include early-ended strings that have finite endpoint
energies but are not production-quality
reaction candidates.

Null reactions are excluded from ranking by default. A candidate whose product
bond graph is identical to its reactant (no bonds formed or broken at the
covalent-radius cutoff) is recorded with exclusion reason
`null reaction: product bond graph identical to reactant`. Without this filter,
strings that relax back to the reactant basin have near-zero `dE` and dominate
the Boltzmann populations, so multi-step searches stall in place. Use
`--sample-allow-unchanged-products` to restore the old behavior, e.g. when
conformational (non-bond-changing) steps are of interest. The per-candidate
counts are stored as `product_bond_added` / `product_bond_removed`.

## Console output

Each iteration prints a short population table:

```text
Top sampled candidates after iteration 1
  score mode: cumulative product dE; min quality: converged; T=298.15 K
  ranked candidates: 5/10
   1. cand_0004 parent=root score= 5.850 kcal/mol pop=36.39% status=converged_no_ts coords=[ADD 1 5; ADD 2 4]
   2. cand_0007 parent=root score= 6.010 kcal/mol pop=27.75% status=converged_no_ts coords=[ADD 1 3; ADD 1 5]
```

At the end it also prints an overall top-candidate table and the path to:

```text
candidate_search_summary.json
```

## Output files

The top-level output directory contains:

```text
runs/sample_search/
  candidate_search_summary.json
  iteration_01/
    root/
      cand_0001/
        reactant.xyz
        isomers.txt
        product.xyz
        summary.json
        grown_string_*.xyz
        opt_converged_*.xyz
        scratch/
```

For every candidate:

- `reactant.xyz`: the input structure for that candidate.
- `isomers.txt`: generated driving coordinates.
- `product.xyz`: extracted product endpoint, if a trajectory was available.
- `summary.json`: per-candidate SE-GSM result and metadata.
- pyGSM trajectory/scratch files: raw SE-GSM outputs.

At the search level:

- `candidate_search_summary.json`: complete search metadata, all candidates,
  iteration reports, top candidates, populations, score mode, and filter settings.

Useful fields in `candidate_search_summary.json` include:

- `candidate_count`: total candidates attempted.
- `ranked_candidate_count`: candidates included in final ranking.
- `search_parameters.sample_count_per_reactant`: requested samples per reactant.
- `search_parameters.sample_iterations`: number of recursive iterations.
- `search_parameters.resample_top_k`: number of products carried forward.
- `search_parameters.sample_score_mode`: `thermodynamic` or `kinetic`.
- `search_parameters.sample_min_quality`: quality filter used for ranking.
- `candidates[*].ranking_included`: whether the candidate passed ranking filters.
- `candidates[*].ranking_exclusion_reason`: why a candidate was excluded.
- `candidates[*].ranking_score_delta_e`: score used for population weighting.
- `candidates[*].path_product_delta_e`: cumulative product energy along the path.
- `candidates[*].path_candidate_ids`: candidate IDs forming the multi-step path.
- `iterations[*].top_candidates[*].relative_population`: normalized iteration-level
  population for ranked top candidates.
- `overall_top_candidates[*].relative_population`: normalized population for the
  final overall ranking.

## Technical notes

- The MLIP calculator is loaded once and reused across candidates by default.
  This avoids reloading the model for every candidate.
- Use `--sample-reload-model-each-candidate` only for debugging calculator state.
- `--sample-export-artifacts` exports SDF/GIF artifacts for every sampled
  candidate, but it adds overhead. It is disabled by default for sampled mode.
- Failed candidates are not fatal for the whole search. They are recorded and
  excluded from ranking unless the selected quality mode allows them.
- The population table is a ranking over sampled candidates, not an exhaustive
  product distribution. The result depends on the candidate pool, the MLIP model,
  SE-GSM convergence, and the quality filter.

## Scientific interpretation

The default thermodynamic score is best interpreted as a relative product
stability ranking over the candidates that were actually sampled and passed the
quality filter.

It is not a direct reaction-rate prediction. For rate-like screening, use
`--sample-score-mode kinetic`, but remember that it is still only a TS-barrier
proxy from SE-GSM and not a full kinetic model with entropy, tunneling, solvent,
or conformer averaging.

For publication-quality DFT-level claims, promising MLIP/SE-GSM candidates should
be recomputed and refined with the intended quantum-chemistry level of theory.
