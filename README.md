# When Do Flat Minima Help? 
## A Study of SAM Across Data Regimes

> **6.7960 Deep Learning Final Project — Due: December 09, 2025**

Davis Lee

This repository investigates when Sharpness-Aware Minimization (SAM) provides a generalization benefit over SGD, specifically focusing on the interaction between **dataset size** and **label noise**.

---

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd sam_analysis

# Install dependencies
pip install -r requirements.txt
```

### 2. Running Experiments

You can run the full suite of experiments (18 runs) or individual configurations.

**Run All Experiments:**
```bash
python src/run_experiments.py --all
```
*Note: This takes approximately 4-5 hours on an NVIDIA A100 GPU.*

**Run Single Experiment:**
```bash
# Example: SAM with 10% data and 20% label noise
python src/run_experiments.py --optimizer sam --data_fraction 0.1 --noise_fraction 0.2
```

### 3. Analysis & Visualization

After experiments complete, generate the analysis figures and summary tables:

```bash
# Generate figures (saved to blog/figures/)
python src/analyze_results.py

# Generate markdown summary for analysis (saved to results_summary.md)
python src/synthesize_results.py
```

---

## Repository Structure

```
sam_analysis/
├── src/
│   ├── run_experiments.py     # Main training loop and experiment runner
│   ├── data_utils.py          # CIFAR-10 loading, stratified subsampling, noise injection
│   ├── sharpness.py           # Loss landscape sharpness estimation
│   ├── analyze_results.py     # Generates plots for the blog post
│   └── synthesize_results.py  # Aggregates results into a summary markdown file
├── sam/                       # Original SAM implementation (forked)
│   └── sam.py                 # SAM optimizer class
├── results/                   # JSON output files from experiments
├── blog/
│   └── figures/               # Generated plots (heatmap, learning curves, etc.)
└── requirements.txt           # Python dependencies
```

## Reproducibility Details

### Hyperparameters
All experiments use **ResNet-18** on **CIFAR-10** with the following fixed settings:
*   **Epochs:** 100
*   **Batch Size:** 128
*   **Learning Rate:** 0.1 (Cosine Annealing schedule)
*   **Momentum:** 0.9
*   **Weight Decay:** 5e-4
*   **SAM Rho:** 0.05
*   **Seed:** 42 (Fixed for data splitting and noise injection)

### Hardware Optimization
*   **SGD:** Uses Automatic Mixed Precision (AMP) via `torch.cuda.amp` for speed.
*   **SAM:** Uses FP32 to ensure stability with the custom two-step update.
*   **Data Loading:** Uses `pin_memory=True` and 2 workers.

### Experiment Matrix
We evaluate 2 optimizers × 3 data fractions × 3 noise levels = 18 total runs.

| Variable | Levels |
|----------|--------|
| **Optimizer** | SGD, SAM |
| **Data Fraction** | 1% (500 samples), 10% (5k samples), 100% (50k samples) |
| **Label Noise** | 0% (Clean), 20%, 40% (Symmetric Flip) |

---

## Citation

This project builds upon the official SAM implementation:

```bibtex
@inproceedings{foret2021sharpnessaware,
  title={Sharpness-aware Minimization for Efficiently Improving Generalization},
  author={Pierre Foret and Ariel Kleiner and Hossein Mobahi and Behnam Neyshabur},
  booktitle={International Conference on Learning Representations},
  year={2021}
}
```
