# Experiment Results Synthesis

This document contains a structured synthesis of the experimental results for LLM analysis.
## 1. SAM Improvement Heatmap Data

Difference in Test Accuracy (SAM - SGD) for each configuration:

| Data Fraction | Noise 0.0 | Noise 0.2 | Noise 0.4 |
|---|---|---|---|
| 0.01 | 3.55% | 2.99% | 1.21% |
| 0.10 | -6.51% | -0.13% | -1.33% |
| 1.00 | 0.95% | 3.78% | 6.34% |

**Raw Test Accuracies (SGD / SAM):**

| Data Fraction | Noise 0.0 (SGD/SAM) | Noise 0.2 (SGD/SAM) | Noise 0.4 (SGD/SAM) |
|---|---|---|---|
| 0.01 | 33.00% / 36.55% | 27.22% / 30.21% | 23.84% / 25.05% |
| 0.10 | 61.77% / 55.26% | 55.57% / 55.44% | 47.66% / 46.33% |
| 1.00 | 88.14% / 89.09% | 81.33% / 85.11% | 74.56% / 80.90% |

## 2. Sharpness vs Generalization Gap Data

Comparison of Loss Landscape Sharpness and Generalization Gap (Train Acc - Test Acc):

| Optimizer | Data Frac | Noise Frac | Sharpness | Gen Gap (%) | Test Acc (%) |
|---|---|---|---|---|---|
| SAM | 0.01 | 0.0 | -0.0085 | 37.45 | 36.55 |
| SAM | 0.01 | 0.2 | 0.0153 | 26.59 | 30.21 |
| SAM | 0.01 | 0.4 | -0.0004 | 22.75 | 25.05 |
| SAM | 0.1 | 0.0 | 0.0361 | 9.68 | 55.26 |
| SAM | 0.1 | 0.2 | 0.0312 | 8.32 | 55.44 |
| SAM | 0.1 | 0.4 | 0.0141 | -9.75 | 46.33 |
| SAM | 1.0 | 0.0 | 0.0473 | 8.28 | 89.09 |
| SAM | 1.0 | 0.2 | 0.1085 | -6.80 | 85.11 |
| SAM | 1.0 | 0.4 | 0.0556 | -21.45 | 80.90 |
| SGD | 0.01 | 0.0 | 0.0198 | 57.60 | 33.00 |
| SGD | 0.01 | 0.2 | 0.0424 | 11.58 | 27.22 |
| SGD | 0.01 | 0.4 | -0.0014 | 66.76 | 23.84 |
| SGD | 0.1 | 0.0 | -0.0008 | 24.43 | 61.77 |
| SGD | 0.1 | 0.2 | -0.0105 | 13.95 | 55.57 |
| SGD | 0.1 | 0.4 | 0.0438 | -9.96 | 47.66 |
| SGD | 1.0 | 0.0 | 0.1995 | 9.51 | 88.14 |
| SGD | 1.0 | 0.2 | 0.1698 | -1.53 | 81.33 |
| SGD | 1.0 | 0.4 | 0.0613 | -12.11 | 74.56 |

## 4. Scaling Behavior (Accuracy vs Data Size)

Test Accuracy (%) as a function of dataset size for different noise levels:

**Noise Level: 0%**

| Data Fraction | SGD Acc | SAM Acc | Delta |
|---|---|---|---|
| 0.01 | 33.00 | 36.55 | +3.55 |
| 0.1 | 61.77 | 55.26 | -6.51 |
| 1.0 | 88.14 | 89.09 | +0.95 |

**Noise Level: 20%**

| Data Fraction | SGD Acc | SAM Acc | Delta |
|---|---|---|---|
| 0.01 | 27.22 | 30.21 | +2.99 |
| 0.1 | 55.57 | 55.44 | -0.13 |
| 1.0 | 81.33 | 85.11 | +3.78 |

**Noise Level: 40%**

| Data Fraction | SGD Acc | SAM Acc | Delta |
|---|---|---|---|
| 0.01 | 23.84 | 25.05 | +1.21 |
| 0.1 | 47.66 | 46.33 | -1.33 |
| 1.0 | 74.56 | 80.90 | +6.34 |


## 3. Training Dynamics (Learning Curve Summary)

Condensed summary of training dynamics. 'Early' = Epoch 10, 'Mid' = Epoch 50, 'Final' = Epoch 100.

| Config (Data/Noise) | Optimizer | Early Test Acc | Mid Test Acc | Final Test Acc | Convergence Speed (Epoch to 90% of Final) |
|---|---|---|---|---|---|
| D=0.01/N=0.0 | SGD | 6.70% | 31.32% | 33.00% | 31 |
| D=0.01/N=0.0 | SAM | 17.03% | 34.71% | 36.55% | 40 |
| D=0.01/N=0.2 | SGD | 11.11% | 24.07% | 27.22% | 52 |
| D=0.01/N=0.2 | SAM | 14.98% | 28.30% | 30.21% | 45 |
| D=0.01/N=0.4 | SGD | 15.85% | 24.08% | 23.84% | 23 |
| D=0.01/N=0.4 | SAM | 16.66% | 22.91% | 25.05% | 46 |
| D=0.1/N=0.0 | SGD | 29.03% | 54.12% | 61.77% | 57 |
| D=0.1/N=0.0 | SAM | 24.65% | 46.69% | 55.26% | 62 |
| D=0.1/N=0.2 | SGD | 23.04% | 48.44% | 55.57% | 53 |
| D=0.1/N=0.2 | SAM | 27.66% | 48.50% | 55.44% | 54 |
| D=0.1/N=0.4 | SGD | 13.67% | 33.71% | 47.66% | 69 |
| D=0.1/N=0.4 | SAM | 17.63% | 36.57% | 46.33% | 61 |
| D=1.0/N=0.0 | SGD | 69.75% | 77.86% | 88.14% | 44 |
| D=1.0/N=0.0 | SAM | 71.51% | 80.30% | 89.09% | 40 |
| D=1.0/N=0.2 | SGD | 64.69% | 75.33% | 81.33% | 21 |
| D=1.0/N=0.2 | SAM | 67.25% | 77.28% | 85.11% | 43 |
| D=1.0/N=0.4 | SGD | 56.27% | 71.85% | 74.56% | 26 |
| D=1.0/N=0.4 | SAM | 52.84% | 70.87% | 80.90% | 42 |
