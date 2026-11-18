# Beyond Price Filters: Context-Aware Machine Learning for Prioritizing Suspicious Trade Flows

A simulation-based study of whether transaction context improves price-anomaly ranking in trade-based money laundering (TBML) scenarios.

## Overview

Traditional price filters compare a declared price with a commodity-level reference. This simulation asks whether models can better rank injected price distortions when they account for transaction volume, origin, destination, season, and a freight proxy.

All records are synthetic. Legitimate prices are generated from a known process, then a random subset of test flows is assigned a ground-truth label and given an over- or under-invoicing distortion. This is a controlled experiment, not evidence of real-world detection performance.

## Methods

| Method | Description |
|---|---|
| B1 – Commodity band | Absolute log-price deviation from the training commodity median (global median fallback for unseen commodities) |
| B1z – Commodity z-score | Commodity deviation scaled by a robust spread estimate |
| B2 – Log-linear context | Ridge regression expected price from transaction context |
| M – Gradient boosting | Histogram gradient boosting expected price from the same context |
| U – Isolation Forest | Unusual commodity-relative price, volume, and freight proxy |
| R50 – Fixed ±50% rule | Price more than 50% above or below the training commodity mean |

Regression methods score absolute residuals, scaled by a per-commodity robust calibration spread. The alert threshold is set to the 98th percentile of calibration scores. Because calibration data contains injected contamination and samples are finite, this is a nominal alert target, not a formal false-positive guarantee.

## Main results

For 25% injected price distortion, averaged over 10 seeds in `results_main.csv`:

- Gradient boosting AP: **0.093**, compared with **0.054** for log-linear context and **0.023** for the commodity z-score benchmark.
- At the nominal 2% alert target, gradient boosting recall was **23.7%**, compared with **8.5%** for the commodity z-score benchmark. The observed gradient-boosting false-positive rate was about **1.7%**.

Average precision is useful at the simulated 1% anomaly prevalence, but absolute detection remains limited: most alerts are still false positives. Results describe only this simulation and its assumptions.

The full run writes one per-seed CSV for each scenario and `results_summary.json`, which contains means and sample standard deviations across seeds. Undefined AP values (for the fixed-rule method) are represented as JSON `null`.

## Limitations

- The experiment uses no real trade data and does not establish operational effectiveness.
- Training, calibration, and test flows are sampled from the same generated world. The experiment does not test generalization to unseen commodities, routes, or changing market conditions.
- Price-only distortions are a narrow typology. Correctly priced transactions and other laundering techniques are outside the detection scope.
- Small average precision gains should be read alongside the low anomaly prevalence, false-positive rates, and variation across random seeds.

This project is an academic simulation, not an anti-money-laundering system. It must not be used for compliance decisions, financial-crime accusations, or operational screening.

## Reproducibility

Requires Python 3.9 or later. From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python tbml_sim.py
```

On macOS or Linux, activate the environment with `source .venv/bin/activate` instead.

The checked-in full-run results were generated with Python 3.14.3, NumPy 2.5.5, pandas 3.1.1, and scikit-learn 1.9.1. Different compatible dependency versions may produce slightly different results.

The full run uses 10 seeds across the main, no-freight-proxy, and 5%-training-contamination scenarios. It can take substantially longer than the smoke run. Use `--output-dir` to choose where results are written:

```text
python tbml_sim.py --output-dir results
```

Run a smaller end-to-end check with one seed and 3,000 training, 1,000 calibration, and 2,000 test flows:

```text
python tbml_sim.py --smoke --output-dir smoke-output
```

Smoke outputs use `results_smoke_` filenames and are not comparable to the full experiment results.

## License

This project is licensed under the MIT License; see [LICENSE](LICENSE).
