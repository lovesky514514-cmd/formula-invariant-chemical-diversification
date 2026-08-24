# Formula-Invariant Chemical Diversification (FICD)

Reproducibility repository for the study:

**Formula-Invariant Chemical Diversification for Robust Candidate Shortlisting in Oxide Materials Databases**

FICD is a post-filter materials-shortlisting procedure designed for fixed experimental or higher-level-computation budgets when the final ranking can change with the scoring formulation. The frozen method first builds a multi-formula scoring fingerprint, requires cross-family support, and then performs chemical-diversity-improving swaps under a hard worst-rule-regret constraint.

## Authors

- **Lu Zhang** — lead and corresponding author; School of Materials Science and Engineering, Changchun University of Science and Technology
- **Gaoding Zhou** — supporting co-author; School of Artificial Intelligence, Changchun University of Science and Technology

**Supervision / project guidance:** Yunlong Jiang, School of Materials Science and Engineering, Changchun University of Science and Technology.

## Repository contents

```text
formula-invariant-chemical-diversification/
├── data/
│   ├── mp_transition_candidates.csv
│   └── wbm_transition_candidates.csv
├── results/
│   ├── distance_metric_validation_all.csv
│   ├── LOFO_external_formula_validation_all.csv
│   ├── family_support_sensitivity_FIXED_all.csv
│   ├── frozen_v03_core_reproduction.csv
│   ├── mp_external_structural_validation.csv
│   └── ... additional shortlist and diagnostic outputs
├── cache/
│   └── official ElMD distance matrices and backend metadata
├── figures/
│   ├── figure1_method.*
│   └── figure2_validation.*
├── manuscript/
│   ├── main.tex
│   ├── main.pdf
│   └── Elsevier elsarticle support files
├── docs/
│   ├── Abstract.txt
│   └── Highlights.txt
├── run_ficd_v03_benchmark.py
├── run_ficd_validation.py
├── requirements.txt
├── CITATION.cff
├── .zenodo.json
└── RELEASE_NOTES.md
```

## Frozen FICD procedure

The main FICD v0.3 algorithm is frozen. The later v0.4 workflow adds validation only.

1. Normalize stability and band-gap preference to `[0,1]`.
2. Evaluate four scalarization families:
   - arithmetic,
   - geometric,
   - harmonic,
   - Chebyshev.
3. Evaluate each family at stability weights `0.4, 0.5, 0.6, 0.7`, giving 16 scoring rules.
4. Convert each rule to rank desirability and form the scoring fingerprint.
5. Compute the scoring-invariance index (SII).
6. Require Top-K support from at least two distinct scalarization families.
7. Form the quality score `Q = sqrt(SII × family_support_fraction)`.
8. Select the Top-K quality seed.
9. Freeze the seed shortlist's worst-rule regret as a hard quality budget.
10. Accept only one-for-one swaps that improve nearest-neighbor chemical diversity without exceeding that regret budget.

The default shortlist budget is `K=25`.

## Data

Two processed oxide candidate pools are included:

- Materials Project-derived Li-Mn-O transition pool.
- WBM-derived oxygen-containing transition pool.

The repository contains processed/derived tables used for reproducibility. Users should also cite the original databases and comply with their applicable terms and licenses.

Key source references used by the manuscript include:

- A. Jain et al., *APL Materials* 1 (2013) 011002. DOI: 10.1063/1.4812323.
- J. Riebesell et al., *Nature Machine Intelligence* 7 (2025) 836-847. DOI: 10.1038/s42256-025-01055-1.
- C. J. Hargreaves et al., *Chemistry of Materials* 32 (2020) 10610-10620. DOI: 10.1021/acs.chemmater.0c03381.

## Reproduce the frozen benchmark

```bash
python -m pip install -r requirements.txt
python run_ficd_v03_benchmark.py
```

## Run the full validation

The full validation includes:

- corrected family-support sensitivity bookkeeping,
- official ElMD chemical-distance validation,
- leave-one-scoring-family-out external formula validation,
- Materials Project crystal-system / space-group validation.

```bash
python -m pip install -r requirements.txt
python -m pip install ElMD --no-deps
python run_ficd_validation.py
```

If a compatible official ElMD implementation is already installed, the second command is unnecessary. Cached ElMD matrices are included to facilitate exact reproduction.

## Main validation result

Across both oxide pools, frozen FICD increases chemical separation relative to its cross-family quality seed without increasing the seed's worst-rule regret. Leave-one-family-out and ElMD tests are included to quantify dependence on scoring formulation and chemical-distance choice rather than assuming those dependencies are absent.

## Manuscript note

The manuscript copy in this repository is the **pre-Zenodo-DOI archival version**. Its Data Availability section intentionally contains a DOI placeholder. After the first GitHub release is archived by Zenodo, the DOI should be inserted into the submission manuscript before journal submission.

## License and third-party data

The MIT License in this repository applies to the original FICD code and repository documentation authored for this project. It does **not** relicense third-party source data. See `DATA_LICENSE_NOTICE.md`.

## Citation

A `CITATION.cff` file is provided for GitHub citation metadata. [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22081000.svg)](https://doi.org/10.5281/zenodo.22081000)
