
> A comparative ML research project evaluating five forecasting model families for predicting periodic revenue/donation-proxy trends, with an anomaly-aware drought detection framework for proactive NGO resource planning.

**Author:** Tavishi Jain | CS + Applied Mathematics | R&D Dept., College Tech Society

---

## Overview

Real-world NGO donation data is rarely public and difficult to source. This project uses the **Kaggle Superstore Sales dataset as a structural proxy** — its seasonal demand patterns, irregular spikes, and cyclical behaviour closely mirror donation time-series dynamics. The forecasting framework is designed to generalise directly to real donation data once available; the methodology, not the domain data, is the core contribution.

The research compares five model families end-to-end, builds a donation drought detection system, and tests whether Indian cultural holiday covariates improve forecast accuracy — producing a full LaTeX research paper alongside the code.

---

## Research Questions

- **RQ1:** Which model family (classical statistical vs. ML vs. deep learning) achieves the best forecast accuracy on seasonal, irregular time-series data?
- **RQ2:** Do Indian cultural holiday covariates (Diwali, Holi, etc.) meaningfully improve forecast accuracy when included as exogenous features?
- **RQ3:** Can a rule-based drought detection framework reliably flag upcoming shortfall periods using model forecasts?

---

## Key Results

| Model | RMSE ($) | MAE ($) | MAPE (%) |
|---|---|---|---|
| ARIMA | 25,445.61 | 19,981.86 | 26.46 |
| ETS (Holt-Winters) | 12,396.26 | 11,175.29 | 18.52 |
| Prophet + Holidays | 14,811.52 | 11,286.44 | 16.15 |
| **XGBoost** | **16,364.50** | **13,418.58** | **13.81 ✓** |
| LSTM | 36,231.84 | 27,450.21 | 31.46 |

**XGBoost achieved the best MAPE (13.81%)**, outperforming the ARIMA baseline by 12.65 percentage points. Notably, the deep learning LSTM underperformed simpler models — likely due to the limited dataset size, a finding discussed in the paper's limitations section.

---

## Core Contribution: Drought Detection Framework

The drought detection system flags any forecast month where predicted volume drops below **70% of the 8-month rolling average**. This gives NGOs advance warning of donation shortfalls, enabling proactive procurement decisions before a crisis occurs. Statistical significance of drought period differences was validated using a **Wilcoxon signed-rank test**.

---

## Notebooks

| # | Notebook | What it covers |
|---|---|---|
| 01 | `01_donation_eda.ipynb` | Data loading, cleaning, STL decomposition, ACF/PACF, descriptive statistics |
| 02 | `02_arima_ets.ipynb` | ARIMA (auto_arima), ETS (Holt-Winters), residual analysis, confidence intervals |
| 03 | `03_prophet.ipynb` | Prophet baseline, Indian holiday covariates, cross-validation, custom seasonality |
| 04 | `04_xgboost.ipynb` | Feature engineering, XGBoost, hyperparameter tuning, SHAP feature importance |
| 05 | `05_lstm.ipynb` | PyTorch LSTM, sequence modelling, hyperparameter sensitivity, multivariate variant |
| 06 | `06_drought_stats.ipynb` | All-model comparison, drought detection framework, Wilcoxon significance test |

---

## Dataset & Limitations

**Dataset:** Kaggle Superstore Sales — [source](https://www.kaggle.com/datasets/vivek468/superstore-dataset-final)

This dataset was chosen as a proxy because actual NGO donation records are rarely publicly available and often too small for meaningful ML benchmarking. Superstore sales share key structural properties with donation data: pronounced seasonality, holiday-driven spikes, and irregular demand cycles. This is a known limitation of the study — results on real donation data may differ, particularly for models sensitive to domain-specific distributional features (e.g., LSTM). A full limitations discussion is included in the research paper.

---

## Tech Stack

Python · Pandas · NumPy · Matplotlib · Seaborn · Statsmodels · pmdarima · Prophet · XGBoost · PyTorch · scikit-learn · LaTeX (Overleaf)

---

## How to Run

```bash
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels pmdarima prophet xgboost torch jupyterlab
```

Open notebooks in order (`01 → 06`) in VS Code or JupyterLab. Each notebook is self-contained with its own data loading cell.

---

## Paper

Full research paper written in LaTeX (Overleaf). Covers methodology, results, significance testing, and limitations. Available on request.
