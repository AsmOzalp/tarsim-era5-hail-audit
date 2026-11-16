# Extracting Environmental Signals of Recorded Agricultural Hail Damage

This repository contains the Python analysis and evaluation scripts used for the manuscript:
**"Extracting Environmental Signals of Recorded Agricultural Hail Damage from Insurance Portfolios Using ERA5 Reanalysis and Explainable Machine Learning"** (submitted to *Stochastic Environmental Research and Risk Assessment - SERRA*).

## Data Confidentiality
In accordance with the data-sharing agreement with the Turkish Agricultural Insurance Pool (TARSIM), the raw settlement-level and policy-level insurance claims data contain proprietary commercial information and cannot be shared publicly. The scripts provided here are designed to run on the standardized `model_ready_data.csv` structure described in Section 2 of the manuscript.

## Script to Manuscript Mapping
The table below maps the specific Python scripts in the `scripts/` folder to the corresponding Tables and Figures in the manuscript:

| Script Name | Manuscript Element Generated |
| :--- | :--- |
| `01_data_pipeline_and_summary.py` | Tables 1, 2, 4; Appendix Tables A1, A2 |
| `02_stage1_mundlak_fe.py` | Tables 5, 6, 7; Appendix Tables A4, A5 |
| `03_stage2_conditional_severity.py` | Table 9 |
| `04_shap_refitted_bootstrap.py` | Table 8 |
| `05_hazard_maps_and_consistency.py` | Figures 1, 2; Table 11; Appendix Table A7 |
| `06_provincial_portfolio_benchmarks.py`| Tables 12, 13; Appendix Table A8 |
| `07_reviewer_sensitivity_CIN_PW.py` | Appendix Table A6 (Panel B) |

## Dependencies
- `pandas`, `numpy`, `scipy`, `statsmodels`, `lightgbm`, `scikit-learn`, `shap`, `matplotlib`
