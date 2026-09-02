# Joint threshold sweep + clustering alternative

Grid: KL [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]; perc_coherent [1.5, 1.6, 1.7, 1.8, 1.9]; perc_degen [1.9, 2.0, 2.1, 2.2]; ent_lo [0.1, 0.15, 0.2, 0.25, 0.3]; ent_hi [1.5, 1.6, 1.7, 1.8, 1.9]; 2850 valid combinations (paper values: KL=1.0, 1.7, 2.0, 0.2, 1.8).

## Label stability (uncertainty over labels)

- figure1_iter200: n=60 agents; median label stability 0.83; 45% of agents keep their paper label in >=90% of the grid.
    - coherent_low_kl: n=38, median 1.00, min 0.47
    - committed: n=11, median 0.58, min 0.29
    - mid_range: n=4, median 0.63, min 0.63
    - policy_collapse: n=6, median 0.80, min 0.60
    - text_degenerate: n=1, median 1.00, min 1.00
- seed_replicates: n=160 agents; median label stability 0.80; 39% of agents keep their paper label in >=90% of the grid.
    - coherent_low_kl: n=119, median 1.00, min 0.48
    - committed: n=14, median 0.58, min 0.23
    - mid_range: n=23, median 0.79, min 0.63
    - policy_collapse: n=1, median 0.80, min 0.80
    - text_degenerate: n=3, median 0.47, min 0.47

## Regime-composition robustness per condition

| condition | n | alive min/med/max | committed min/med/max |
|---|---|---|---|
| ashbourne-16 | 16 | 1.00 / 1.00 / 1.00 | 0.00 / 0.19 / 0.44 |
| ashbourne-2 | 2 | 0.50 / 1.00 / 1.00 | 0.00 / 0.00 / 0.50 |
| ashbourne-4 | 4 | 0.50 / 0.75 / 0.75 | 0.00 / 0.00 / 0.00 |
| ashbourne-8 | 8 | 0.88 / 0.88 / 0.88 | 0.00 / 0.00 / 0.38 |
| ashbourne16 | 80 | 0.89 / 0.97 / 1.00 | 0.00 / 0.07 / 0.24 |
| inner_loop8 | 40 | 1.00 / 1.00 / 1.00 | 0.03 / 0.05 / 0.15 |
| margin_notes8 | 40 | 0.80 / 1.00 / 1.00 | 0.00 / 0.05 / 0.12 |
| strathearn-16 | 16 | 0.75 / 0.88 / 1.00 | 0.00 / 0.12 / 0.31 |
| strathearn-2 | 2 | 0.00 / 0.00 / 0.50 | 0.00 / 0.00 / 0.00 |
| strathearn-4 | 4 | 0.50 / 0.75 / 1.00 | 0.00 / 0.00 / 0.00 |
| strathearn-8 | 8 | 0.88 / 1.00 / 1.00 | 0.12 / 0.12 / 0.25 |

## Committed-vs-coherent PMI contrast across the grid

- cross: paper-threshold difference +0.199 nats/token; across 2850 grid points with both groups non-empty (of 2850; 0 had an empty group): 100.0% positive, median +0.185, IQR [+0.130, +0.237], min +0.028.
    - committed group size across grid: min 2, median 6, max 17.
- seeds: paper-threshold difference +0.160 nats/token; across 2850 grid points with both groups non-empty (of 2850; 0 had an empty group): 100.0% positive, median +0.160, IQR [+0.160, +0.198], min +0.130.
    - committed group size across grid: min 4, median 6, max 7.

## policy_collapse: low- vs high-entropy branch

- Of 7 policy_collapse agents (Figure-1 set + replicates) at paper thresholds: 7 via entropy < 0.2 (near-deterministic), 0 via entropy > 1.8 (high-entropy divergence).

## Inner Loop committed-agent accounting

- End-of-training committed in the 5 inner-loop seeds: 6 (run, character) instances, 5 unique characters (Henrietta 'Henri' Voss, Kai Dempsey, Mei-Lin Park, Ravi Krishnamurthy, Renata Solberg).
- Per seed: inner_loop: 3; inner_loop_seed3: 1; inner_loop_seed4: 1; inner_loop_seed5: 1.

## Unsupervised clustering alternative


### Clustering: fig1_plus_replicates (n=220)

- KMeans silhouette selects k=8 (ARI vs 5-way labels 0.12); forced k=5 ARI 0.16. GMM BIC selects k=4.
- cross-tab written to clustering_crosstab_fig1_plus_replicates.csv

### Clustering: endpoints_cross (n=56)

- KMeans silhouette selects k=4 (ARI vs 5-way labels 0.52); forced k=5 ARI 0.55. GMM BIC selects k=1.
- cross-tab written to clustering_crosstab_endpoints_cross.csv
- PMI contrast with cluster labels (no thresholds): high-KL cluster (n=20, mean KL 2.44) shrinkage +0.121 vs low-KL cluster (n=20, mean KL 0.34) +0.058; one-sided Mann-Whitney U=234, p=0.182.

### Clustering: endpoints_seeds (n=40)

- KMeans silhouette selects k=2 (ARI vs 5-way labels 1.00); forced k=5 ARI 0.23. GMM BIC selects k=8.
- cross-tab written to clustering_crosstab_endpoints_seeds.csv
- PMI contrast with cluster labels (no thresholds): high-KL cluster (n=6, mean KL 3.59) shrinkage +0.215 vs low-KL cluster (n=34, mean KL 0.16) +0.055; one-sided Mann-Whitney U=158, p=0.016.

### Clustering: endpoints_pooled (n=96)

- KMeans silhouette selects k=4 (ARI vs 5-way labels 0.54); forced k=5 ARI 0.49. GMM BIC selects k=2.
- cross-tab written to clustering_crosstab_endpoints_pooled.csv
- PMI contrast with cluster labels (no thresholds): high-KL cluster (n=25, mean KL 2.07) shrinkage +0.114 vs low-KL cluster (n=46, mean KL 0.18) +0.057; one-sided Mann-Whitney U=673, p=0.120.

## Dose-response: linear vs step, and KL bimodality

- pooled: linear R2=0.16 (slope +0.029/nat), step R2=0.13; adding a step to the linear model moves R2 to 0.17 (step coef +0.051). LOOCV: linear 0.0181 vs step 0.0186.
- seeds: linear R2=0.34 (slope +0.046/nat), step R2=0.20; adding a step to the linear model moves R2 to 0.34 (step coef +0.006). LOOCV: linear 0.0135 vs step 0.0160.
- cross: linear R2=0.07 (slope +0.018/nat), step R2=0.09; adding a step to the linear model moves R2 to 0.10 (step coef +0.090). LOOCV: linear 0.0283 vs step 0.0255.
- GMM on log10 KL (pooled, n=73): best k=2; 2-comp modes at 0.18 / 2.64 nats, weights [0.76, 0.24].
- GMM on log10 KL (cross, n=33): best k=1; 2-comp modes at 0.34 / 3.56 nats, weights [0.26, 0.74].
- GMM on log10 KL (seeds, n=40): best k=2; 2-comp modes at 0.13 / 2.84 nats, weights [0.85, 0.15].
