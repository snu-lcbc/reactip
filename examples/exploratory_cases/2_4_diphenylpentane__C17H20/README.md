# 2,4-Diphenylpentane (`C17H20`)

This case is an exploratory ReactIP/SE-GSM example, not a benchmark. The goal
is to probe how the MLIP behaves on a larger alkyl-substituted biaryl system.

## Provenance

- requested name: `2,4-diphenylpentane`
- PubChem CID: `244014`
- PubChem title: `4-Phenylpentan-2-ylbenzene`
- molecular formula: `C17H20`
- connectivity SMILES: `CC(CC(C)C1=CC=CC=C1)C2=CC=CC=C2`
- 3D source: PubChem 3D conformer via
  `https://pubchem.ncbi.nlm.nih.gov/compound/244014`

`reactant.xyz` was transcribed from the PubChem 3D SDF record. It is a useful
starting geometry for calculator tests and exploratory path searches.

## Atom map

Heavy-atom numbering in `reactant.xyz`:

- `1`: central methylene bridge between the two benzylic carbons
- `2`: left benzylic carbon
- `3`: right benzylic carbon
- `4-8-12-16-14-10`: left phenyl ring
- `5-9-13-17-15-11`: right phenyl ring
- `6`: left methyl group
- `7`: right methyl group

## Driving-coordinate files

Three hypotheses are included:

- `isomers_torsion_backbone_anti.txt`
  - conformational exploration
  - rotates the `4-2-1-3` dihedral toward an anti arrangement
  - lowest-risk starting point for testing whether the MLIP gives a smooth
    torsional string
- `isomers_aryl_shift_left.txt`
  - exploratory 1,2-aryl migration hypothesis
  - breaks `2-4` and forms `1-4`
- `isomers_methyl_shift_left.txt`
  - exploratory 1,2-methyl migration hypothesis
  - breaks `2-6` and forms `1-6`

The aryl- and methyl-shift files should be treated as stress tests of the
model/path-search workflow, not as validated chemistry.

## Helper scripts

- `run_calculator.sh`
  - single-point energy/force evaluation with `reactip_calculator.py`
- `run_se_gsm_torsion.sh`
  - safer exploratory SE-GSM launch on the torsional path
- `run_se_gsm_aryl_shift.sh`
  - higher-risk SE-GSM launch on the aryl-shift hypothesis
- `run_se_gsm_methyl_shift.sh`
  - higher-risk SE-GSM launch on the methyl-shift hypothesis

All scripts expect a model path as the first argument or via `REACTIP_MODEL`.
The repository does not assume that model weights are committed in Git.
