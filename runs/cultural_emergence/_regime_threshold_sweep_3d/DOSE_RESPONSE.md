
## Dose-response: linear vs step, and KL bimodality

- pooled: linear R2=0.16 (slope +0.029/nat), step R2=0.13; adding a step to the linear model moves R2 to 0.17 (step coef +0.051). LOOCV: linear 0.0181 vs step 0.0186.
- seeds: linear R2=0.34 (slope +0.046/nat), step R2=0.20; adding a step to the linear model moves R2 to 0.34 (step coef +0.006). LOOCV: linear 0.0135 vs step 0.0160.
- cross: linear R2=0.07 (slope +0.018/nat), step R2=0.09; adding a step to the linear model moves R2 to 0.10 (step coef +0.090). LOOCV: linear 0.0283 vs step 0.0255.
- GMM on log10 KL (pooled, n=73): best k=2; 2-comp modes at 0.18 / 2.64 nats, weights [0.76, 0.24].
- GMM on log10 KL (cross, n=33): best k=1; 2-comp modes at 0.34 / 3.56 nats, weights [0.26, 0.74].
- GMM on log10 KL (seeds, n=40): best k=2; 2-comp modes at 0.13 / 2.84 nats, weights [0.85, 0.15].
