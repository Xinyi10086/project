# project

> **MSc project** — sparse regression for equation discovery under noise, comparing Bayesian evidence (Bayesian Razor), classical STLSQ, and an Errors‑in‑Variables extension via Orthogonal Distance Regression (ODR).

## Overview

This repository contains minimal, reproducible code and notebooks for discovering sparse polynomial models from noisy data. It focuses on:
- **STLSQ**: sparsity‑promoting regression underlying SINDy (for response‑noise / vertical residuals).
- **STLSQ–ODR**: extending STLSQ to the **Errors‑in‑Variables (EIV)** setting, where both predictors and the response are noisy, by replacing the least‑squares fit with **Orthogonal Distance Regression**.
- **Bayesian Razor**: a (work‑in‑progress) module exploring Bayesian evidence for model selection.

The notebooks generate illustrative figures saved in `output/`.

## Repository Structure

```
project_latest/
└── project-Project/
│   ├── .gitignore  (3.4 KB)
│   ├── LICENSE  (1.0 KB)
│   ├── README.md  (0.0 KB)
│   └── notebook/
│   │   ├── Bayesian_Razor_STLSQ.ipynb  (1.97 MB)
│   │   ├── STLSQ.ipynb  (2.07 MB)
│   │   ├── hello.ipynb  (62.3 KB)
│   └── output/
│   │   ├── compare.png  (27.6 KB)
│   │   ├── error 1.png  (100.5 KB)
│   │   ├── error 2.png  (18.3 KB)
│   │   ├── error 3.png  (84.9 KB)
│   │   ├── error 4.png  (18.3 KB)
│   │   ├── error.png  (1007.4 KB)
│   │   ├── large error 2.png  (18.4 KB)
│   │   ├── large error.png  (100.5 KB)
│   │   ├── model selection.png  (47.4 KB)
│   │   ├── output.png  (519.7 KB)
│   │   ├── small error 2.png  (18.3 KB)
│   │   ├── small error.png  (84.9 KB)
│   │   └── study1/
│   │   └── study2/
│   │   └── study3/
│   └── src/
│   │   ├── Bayesian_razor.py  (1.3 KB)
│   │   ├── README  (0.0 KB)
│   │   ├── STLSQ.py  (1.2 KB)
│   │   ├── STLSQ_ODR.py  (2.9 KB)
```

## Installation

> Requires **Python 3.10+**

```bash
# (optional) create a virtual environment
python -m venv .venv
source .venv/bin/activate   # on Windows: .venv\Scripts\activate

# install core dependencies
pip install numpy scipy matplotlib pandas jupyter
```

## Quick Start

Launch Jupyter and open the demonstration notebooks:
```bash
jupyter lab
```
- `project-Project/notebook/Bayesian_Razor_STLSQ.ipynb`
- `project-Project/notebook/STLSQ.ipynb`
- `project-Project/notebook/hello.ipynb`

## Python Modules

**`project-Project/src/Bayesian_razor.py`**

> y(x) = a₀ xᴺ + a₁ xᴺ⁻¹ + ... + a_N return: y, dy/da, d2y/dada

Functions: solve_poly, bayes_poly_map

**`project-Project/src/STLSQ.py`**

Functions: STLSQ

**`project-Project/src/STLSQ_ODR.py`**

> Returns: beta (deg+1, n_states) — denormalized, can be directly multiplied with x^k

Functions: l2_norm_factors, _odr_fit_powers, poly_model, STLSQ_ODR


## Data and Outputs


**Output / results directories**:
- `project-Project/output`

## Notes

- STLSQ uses vertical residuals for fitting and model selection; for EIV, use ODR to obtain orthogonal residuals and compute BIC from the profiled ODR objective.
- Threshold selection can be grid‑searched with BIC; when switching to ODR, redefine RSS to include both vertical misfit and predictor adjustments.

## License

This project includes a license file:
- `project-Project/LICENSE`

## Acknowledgements

Guidance by **Professor Guy Nason** and **Dr. Lloyd Fung** is gratefully acknowledged.
