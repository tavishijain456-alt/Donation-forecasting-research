# Donation Forecasting Using Time Series ML Models

A comparative ML research project evaluating five forecasting models for predicting
donation volume in community food assistance programs, with a donation drought
detection framework for proactive NGO procurement planning.

**Author:** Tavishi Jain | CS + Applied Maths | R&D Department, College Tech Society

---

## Key Results

| Model | RMSE ($) | MAE ($) | MAPE (%) |
|---|---|---|---|
| ARIMA | 25,445.61 | 19,981.86 | 26.46 |
| ETS (Holt-Winters) | 12,396.26 | 11,175.29 | 18.52 |
| Prophet + Holidays | 14,811.52 | 11,286.44 | 16.15 |
| **XGBoost** | **16,364.50** | **13,418.58** | **13.81 ✓** |
| LSTM | — | — | — |

**XGBoost achieved the best MAPE of 13.81%**, outperforming the ARIMA baseline by 12.65 percentage points.

---

## What This Project Does

- Compares 5 forecasting model families end-to-end: ARIMA, ETS, Prophet, XGBoost, LSTM
- Tests whether Indian cultural holiday covariates improve forecast accuracy (RQ2)
- Builds a **donation drought detection framework** — flags months where forecast drops below 70% of the 8-month rolling average, giving NGOs advance warning of shortfalls (RQ3)
- Full research paper written in LaTeX (Overleaf)

---

## Notebooks

| # | Notebook | What it covers |
|---|---|---|
| 01 | `01_donation_eda.ipynb` | Data loading, cleaning, STL decomposition, ACF/PACF, descriptive stats |
| 02 | `02_arima_ets.ipynb` | ARIMA (auto_arima), ETS (Holt-Winters), residual analysis, confidence intervals |
| 03 | `03_prophet.ipynb` | Prophet baseline, Indian holiday covariates, cross-validation, custom seasonality |
| 04 | `04_xgboost.ipynb` | Feature engineering, XGBoost, hyperparameter tuning, feature importance |
| 05 | `05_lstm.ipynb` | PyTorch LSTM, sequence modelling, hyperparameter sensitivity, multivariate variant |
| 06 | `06_drought_stats.ipynb` | All-model comparison, drought detection framework, Wilcoxon significance test |

---

## How to Run

```bash
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels pmdarima prophet xgboost torch jupyterlab
```

Open notebooks in order (01 → 06) in VS Code or JupyterLab. Each notebook is self-contained with its own data loading cell.

**Dataset:** Superstore Sales — [Kaggle](https://www.kaggle.com/datasets/vivek468/superstore-dataset-final)

---

## Tech Stack

Python · Pandas · NumPy · Matplotlib · Seaborn · Statsmodels · pmdarima · Prophet · XGBoost · PyTorch · scikit-learn · LaTeX (Overleaf)

