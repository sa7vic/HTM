# HTM

This repository was developed as part of Advitiya '26, the annual technical fest of IIT Ropar, for the competition **Predictathon**.

## Overview

The project focuses on predicting material stress using a combination of physics-based models and advanced machine learning techniques. The code integrates LightGBM, XGBoost, and CatBoost regressors, employs physics-informed feature engineering, and leverages robust cross-validation for reliable predictions.

## Dataset

The dataset (`train_clean.csv` and `test_clean.csv`) contains physical measurements:

- **Force (N)**: Force applied to material samples
- **Strain (%)**: Relative deformation
- **Stress (MPa)**: Target to predict in the train set

_Code samples load the dataset as:_
```python
train = pd.read_csv('/kaggle/input/hack-the-model-stress-prediction-challenge/train_clean.csv')
test = pd.read_csv('/kaggle/input/hack-the-model-stress-prediction-challenge/test_clean.csv')
```

## Project Details

- **Physics-based baseline**: Uses sigmoid functions and numerical optimization to fit experimental data
- **Residual learning**: Machine learning models (LightGBM, XGBoost, CatBoost) predict the residual (difference) between the physics model and observed data
- **Cross-validation**: Multiple models, robust KFold splits, and feature selection for performance
- **Scripts**: Main logic is in `htm1.py`, `htm2.py`, `htm3.py`, and `htm4.py` with each script implementing and tuning different model and fold configurations

## How It Works

- The code first fits a physics-based "sigmoid" model for material stress, optimizing its parameters.
- Residuals (difference between prediction and reality) are then learned/predicted using machine learning models.
- All models are trained with cross-validation and finally ensembled.

---

*Built for Predictathon at Advitiya '26 — IIT Ropar's annual technical festival.*
