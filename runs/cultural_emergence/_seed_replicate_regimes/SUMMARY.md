# Cross-seed regime replication summary (2026-07-31)

Classification thresholds identical to the submitted figures: KL=1.0, perc_coherent=1.7, perc_degen=2.0, entropy band=[0.2, 1.8].


## Study Group (Ashbourne), 16 agents  (5 independent populations, evaluated at iteration 150)

- Alive fraction (coherent+committed+mid-range): mean 0.95 ± 0.08 (sd), 95% CI [0.85, 1.00] across seeds.
- Committed fraction: mean 0.10 ± 0.07 (sd), 95% CI [0.01, 0.19].
- Population-level outcomes (k/n populations, exact 95% CI):
    - >=1 committed agent: 4/5 [0.28, 0.99]
    - >=1 coherent (low-KL) agent: 4/5 [0.28, 0.99]
    - majority alive (>50%): 5/5 [0.48, 1.00]
    - >=1 policy-collapse agent: 1/5 [0.01, 0.72]
    - >=1 text-degenerate agent: 1/5 [0.01, 0.72]
    - whole population collapsed/degenerate: 0/5 [0.00, 0.52]
- Within-population dependence (one-way ICC across seeds): kl ICC=0.00 (design effect 1.0, effective n 80/80), perc_loss ICC=0.84 (design effect 13.7, effective n 6/80), entropy ICC=0.16 (design effect 3.5, effective n 23/80)
- Per-character regime consistency across seeds (roster fixed): Fleiss kappa = -0.09, mean modal agreement = 0.58.
- Threshold sensitivity (KL in [0.5,1.5], perc in [1.6,1.8]): cross-seed alive mean spans [0.95, 0.95].
- Timepoint robustness: common-iter vs final-iter per-agent agreement s1=0.38, s2=1.00, s3=1.00, s4=1.00, s5=1.00.

## Inner Loop, 8 agents  (5 independent populations, evaluated at iteration 78)

- Alive fraction (coherent+committed+mid-range): mean 1.00 ± 0.00 (sd), 95% CI [1.00, 1.00] across seeds.
- Committed fraction: mean 0.07 ± 0.07 (sd), 95% CI [0.00, 0.16].
- Population-level outcomes (k/n populations, exact 95% CI):
    - >=1 committed agent: 3/5 [0.15, 0.95]
    - >=1 coherent (low-KL) agent: 5/5 [0.48, 1.00]
    - majority alive (>50%): 5/5 [0.48, 1.00]
    - >=1 policy-collapse agent: 0/5 [0.00, 0.52]
    - >=1 text-degenerate agent: 0/5 [0.00, 0.52]
    - whole population collapsed/degenerate: 0/5 [0.00, 0.52]
- Within-population dependence (one-way ICC across seeds): kl ICC=0.00 (design effect 1.0, effective n 40/40), perc_loss ICC=0.56 (design effect 4.9, effective n 8/40), entropy ICC=0.00 (design effect 1.0, effective n 39/40)
- Per-character regime consistency across seeds (roster fixed): Fleiss kappa = 0.10, mean modal agreement = 0.93.
- Threshold sensitivity (KL in [0.5,1.5], perc in [1.6,1.8]): cross-seed alive mean spans [1.00, 1.00].
- Timepoint robustness: common-iter vs final-iter per-agent agreement s1=0.75, s2=1.00, s3=0.88, s4=1.00, s5=1.00.
  - at iter 100 (n=4 seeds): alive 1.00 ± 0.00 (sd)

## Margin Notes, 8 agents  (5 independent populations, evaluated at iteration 86)

- Alive fraction (coherent+committed+mid-range): mean 1.00 ± 0.00 (sd), 95% CI [1.00, 1.00] across seeds.
- Committed fraction: mean 0.07 ± 0.11 (sd), 95% CI [0.00, 0.21].
- Population-level outcomes (k/n populations, exact 95% CI):
    - >=1 committed agent: 2/5 [0.05, 0.85]
    - >=1 coherent (low-KL) agent: 5/5 [0.48, 1.00]
    - majority alive (>50%): 5/5 [0.48, 1.00]
    - >=1 policy-collapse agent: 0/5 [0.00, 0.52]
    - >=1 text-degenerate agent: 0/5 [0.00, 0.52]
    - whole population collapsed/degenerate: 0/5 [0.00, 0.52]
- Within-population dependence (one-way ICC across seeds): kl ICC=0.04 (design effect 1.2, effective n 32/40), perc_loss ICC=0.88 (design effect 7.2, effective n 6/40), entropy ICC=0.34 (design effect 3.4, effective n 12/40)
- Per-character regime consistency across seeds (roster fixed): Fleiss kappa = 0.10, mean modal agreement = 0.93.
- Threshold sensitivity (KL in [0.5,1.5], perc in [1.6,1.8]): cross-seed alive mean spans [1.00, 1.00].
- Timepoint robustness: common-iter vs final-iter per-agent agreement s1=0.38, s2=1.00, s3=0.88, s4=1.00, s5=1.00.
  - at iter 150 (n=4 seeds): alive 0.97 ± 0.06 (sd)
