# Reaction Path Sampling with ReactIP

This document describes the sampled reaction-path search implemented in
`run_se_gsm.py`. The feature uses a pretrained ReactIP MLIP model with SE-GSM to
sample possible product-forming driving coordinates, rank the candidates, and
optionally continue from the top products to build two-step or three-step paths.

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
    --sample-min-quality converged \
    --device cuda \
    --output-dir runs/sample_search
```

This samples up to 10 candidate reactions per frontier reactant. After each
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

Candidate generation is implemented in `reactip/sampling.py`.

- Existing bonds are inferred from covalent radii and the current XYZ geometry.
- Candidate operations are written as pyGSM-style driving coordinates:
  `ADD i j`, `BREAK i j`, or small coordinate sets containing multiple entries.
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

## Ranking and populations

The printed population is normalized over ranked candidates with:

```text
population_i = exp(-dE_i / RT) / sum_j exp(-dE_j / RT)
```

The default score mode is:

```text
--sample-score-mode thermodynamic
```

In this mode, `dE` is the cumulative product energy relative to the original
root reactant. Cumulative scoring is important for two-step and three-step
sampling because candidates from different parent products do not share the same
local energy reference.

The alternative mode is:

```text
--sample-score-mode kinetic
```

This uses a TST-like proxy based on cumulative TS barriers. It only ranks
candidates where SE-GSM identifies a unique TS. It should be treated as a
screening proxy, not a full rate calculation.

Candidate quality is controlled by:

```text
--sample-min-quality {finite,completed,converged,ts}
```

Recommended default:

```text
--sample-min-quality converged
```

Use `finite` only for debugging or smoke tests because it may include early-ended
strings that have finite endpoint energies but are not production-quality
reaction candidates.

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
