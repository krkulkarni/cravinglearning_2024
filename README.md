# A computational mechanism linking momentary craving and decision-making in alcohol-drinkers and cannabis-users

Contains analysis code for [https://doi.org/10.1101/2021.05.04.442433](https://doi.org/10.1101/2023.04.24.538109)

Jupyter notebooks for analyses:
* 0_group_demographics.ipynb (Generate group demographic summary)
* 0_model_free_metrics.ipynb (Generate model-agnostic metrics)
* 1_decision_model_comparison.ipynb (Perform computational modeling of decision-making and model comparison)
* 2_decision_model_simulations.ipynb (Generate decision model simulations)
* 3_bycondition_decision_parameter_dist.ipynb (Visualize parameter distributions for decision modeling)
* 4_craving_model_comparison.ipynb (Perform computational modeling of momentary craving and model comparison)
* 5_craving_parameter_dist.ipynb (Visualize parameter distributions for craving modeling)
* 6_parameter_correlations.ipynb (Visualize simple correlations between model parameters and clinical measures)
* 7_6model_bayesian_regression.ipynb (Perform multivariate regression with model parameters to predict clinical measures)

## Installation

Requires: Python 3.9.16\
Python modules to install:
* arviz==0.17.1
* bambi==0.13.0
* ipython==8.15.0
* matplotlib==3.8.4
* numpy==1.25.2
* pandas==2.2.2
* patsy==0.5.3
* pyem==0.0.1
* pymc==5.12.0
* pytensor==2.19.0
* scikit-learn==1.2.2
* scipy==1.10.0
* seaborn==0.13.2
* statsmodels==0.14.0

Tested on: Macbook Pro 2021, M1 Pro, 16GB RAM

To install:
```
conda create -f conda_req.yml
conda activate test_env
python -m pip install --upgrade git+https://github.com/shawnrhoads/pyEM.git

# To run jupyter notebooks
pip install ipykernel
```
Installation time: ~1 min

## Usage

Run jupyter notebooks in order to reproduce paper analyses and main figures.

Run time: ~20 min to reproduce all results

## Author

* **Kaustubh Kulkarni** (email: kaustubh.kulkarni at icahn.mssm.edu)

