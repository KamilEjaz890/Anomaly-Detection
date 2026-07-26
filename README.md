# Predictive Maintenance & Anomaly Detection — NASA Turbofan Engine Degradation (C-MAPSS FD001)

## Overview
This project builds a full predictive maintenance pipeline on NASA's C-MAPSS turbofan engine degradation dataset (FD001 subset). It combines **Remaining Useful Life (RUL) regression** with **two anomaly detection approaches** — a classical statistical method (multivariate Gaussian, the Andrew Ng ML course approach) and a modern ensemble method (Isolation Forest) — and compares them head-to-head.

## Problem Statement
Given time-series sensor readings from a fleet of turbofan engines running to failure, predict:
1. How many operating cycles remain before failure (RUL)
2. Whether a given engine reading should be flagged as anomalous (needs maintenance attention)

## Dataset
- **Source:** NASA C-MAPSS (Commercial Modular Aero-Propulsion System Simulation), FD001 subset
- 100 training engines (full run-to-failure trajectories), 100 test engines (truncated trajectories)
- 21 sensor measurements + 3 operational settings per cycle
- Single operating condition, single fault mode (simplest of the four FD subsets)

## Pipeline
1. **Data loading & EDA** — inspected 21 sensors, identified 7 flat/constant sensors (no signal under single operating condition) and 14 informative degrading sensors
2. **Feature engineering**
   - RUL label derived from max cycle per engine, piecewise-linear capped at 125 cycles (standard practice — early-life degradation is not linear)
   - Rolling mean/std (window=5) added per sensor to smooth noise → ~42 total features
3. **Train/validation split** — engine-wise split (not row-wise) to prevent data leakage between consecutive cycles of the same engine
4. **RUL Regression**
   - Random Forest baseline
   - XGBoost (best performer)
5. **Anomaly Detection — two approaches compared:**
   - **Multivariate Gaussian (Andrew Ng method):** fit μ, Σ on healthy-engine data, computed p(x) via multivariate normal PDF, selected optimal epsilon threshold by maximizing F1-score on validation set
   - **Isolation Forest:** ensemble of random trees isolating anomalies via shorter path lengths, no distributional assumptions
6. **Visualization** — degradation trajectories with anomaly flags overlaid per engine

## Results

| Method | Precision | Recall | F1-Score |
|---|---|---|---|
| Gaussian (Andrew Ng) | 0.456 | 0.743 | 0.565 |
| Isolation Forest | 0.792 | 0.760 | 0.776 |

**Key finding:** Isolation Forest outperformed the classical Gaussian approach primarily on precision (0.79 vs 0.46), while recall was comparable. This is because real sensor degradation is often non-linear and doesn't strictly follow a multivariate normal distribution — an assumption the Gaussian method depends on but Isolation Forest does not.

**Anomaly label definition:** RUL < 20 cycles flagged as ground-truth anomaly (imminent maintenance need). Recall was prioritized in interpretation over precision, since missed failures carry higher real-world cost than false alarms in a maintenance context.

## Tech Stack
Python, pandas, NumPy, scikit-learn, XGBoost, SciPy (multivariate_normal), Matplotlib

## Key Learnings
- Engine-wise (not row-wise) train/validation splitting is essential for time-series data to avoid leakage
- RUL capping (piecewise linear) improves model stability by preventing the model from over-focusing on early, near-flat degradation cycles
- Classical statistical anomaly detection (Gaussian/probability density) is interpretable and fast but assumes data normality; ensemble methods like Isolation Forest handle non-linear, non-Gaussian degradation patterns better
- Precision/recall trade-offs must be interpreted in business context — in predictive maintenance, recall (catching real failures) often outweighs precision (avoiding false alarms)
