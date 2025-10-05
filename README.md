# dfs-analysis
A dashboard for viewing historic NESO demand flexibility service (DFS) data and same-day forecasting.

---

## Overview

This repository contains:

- A **logistic regression** model for **DFS event prediction**.
- A **linear regression (OLS)** model for **maximum accepted DFS price prediction** (conditional on event).
- A **Streamlit dashboard** to visualise:
  - Historical DFS events, LoLP/DRM signals, interconnectors, and system price.
  - Forecast results and model evaluation metrics.

---

### Live Demo
You can try the hosted version here:

[https://dfs-forecast.streamlit.app](https://dfs-forecast.streamlit.app)

---

## Model Summary

| Model | Purpose | Evaluation Metric | Baseline Result |
|--------|----------|-------------------|----------------|
| Logistic Regression | Probability of DFS event | AUC | ≈ 0.74 |
| Linear Regression | Maximum accepted price (conditional on event) | R² | ≈ 0.31 |

- Event prediction hit rate (threshold 0.7): **≈ 33%** for events, **≈ 92%** for non-events
- Model period: **Jan–Sep 2025**
  (earlier months removed due to extreme pricing at year-round DFS scheme launch)

---

## Data Sources

- **NESO Data Portal**
  - DFS Utilisation: accepted/rejected offers, volumes, and prices
  - Interconnector trades and clearing prices
- **Elexon BMRS**
  - Derated Margin (DRM) & Loss of Load Probability (LoLP) forecasts
  - System Price and Net Imbalance Volume (NIV)

---

## Forecast Issue Rule

All inputs must be **available by 10:00** on the day of forecast:

| Input | Timing Rule |
|--------|--------------|
| DRM / LoLP | 8h and 12h forecasts used; after 18:00, 8h values default to 12h |
| Interconnectors | 1-day lagged volumes and clearing prices |
| System Price & NIV | 1-day lagged values |
| DFS Volume | 1-day lagged values |
| Forecast Window | 16:00–21:00 (evening peak only) |

---

## Features Used (10 total)

1. `drm_forecast_12h`
2. `drm_forecast_8h`
3. `lolp_forecast_12h`
4. `lolp_forecast_8h`
5. `interconnector_volume_1d_lag`
6. `interconnector_cp_1d_lag`
7. `interconnector_dispatched_1d_lag` *(binary)*
8. `system_price_1d_lag`
9. `niv_1d_lag`
10. `dfs_volume_1d_lag`

---

## Model Formulation

### 1️. Logistic Regression (DFS Event Probability)

We model the probability of an event as:

$$
P(\text{event}=1 \mid \mathbf{x}) =
\sigma(\beta_0 + \mathbf{x}^\top \boldsymbol{\beta}),
\quad
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

- Trained with **maximum likelihood estimation (MLE)**
- Solver: `liblinear` (stable for small datasets)
- Balanced class weighting
- 5-fold **time-series cross-validation**

**Classification rule:**
Predict event `dfs_event_pred` when `P(event) > 0.7`

---

### 2️. Linear Regression (DFS Max Price)

Conditional on event:

$$
y = \alpha_0 + \mathbf{x}^\top \boldsymbol{\alpha} + \varepsilon,
\quad
\varepsilon \sim \mathcal{N}(0, \sigma^2)
$$

- Trained on periods where `dfs_event_pred == 1`
- Pipeline:
  `SimpleImputer(median)` → `StandardScaler()` → `LinearRegression(fit_intercept=True)`
- Evaluated on **time-ordered 15% test set**

---

## Evaluation Summary

| Metric | Value | Comment |
|---------|-------|----------|
| Mean AUC | 0.74 | Passable separability (AUC = 0.5 → no better than coin-flip) |
| Event hit-rate (p>0.7) | 33% | Conservative threshold used |
| R² (Price model) | 0.31 | Weak fit, ~31% variance explained |
| MAE | ~18.4 £/MWh | Acceptable first-pass |

**Feature Importance (top 3):**
- Event prediction: `drm_forecast_8h`, `dfs_volume_1d_lag`, `drm_forecast_12h`
- Max price prediction: `lolp_forecast_12h`, `lolp_forecast_8h`, `system_price_1d_lag`

---

## Modelling Assumptions

- Price forecast targets the **maximum accepted price**.
- **Evening-only** model; not generalised across all SPs.
- **Regime change:** early months excluded (prices > £700/MWh).
- **No look-ahead bias:** only features known by 10:00 are used.

---

## Installation

```bash
# 1. Clone this repo
git clone https://github.com/<sam-secher>/<dfs-analysis>.git
cd <repo-name>

# 2. Create and activate the environment
conda env create -f environment.yml
conda activate dfs-forecast

# 3. Launch the Streamlit app
streamlit run app.py
