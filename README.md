
# Multi-Output Machine Learning Models

This repository contains code for training, validation, prediction, and visualization of three multi-output models. The project implements three mainstream gradient boosting algorithms: XGBoost, LightGBM, and CatBoost.

## Overview

This project simultaneously predicts three target variables: `cropland_area`, `built_up_area`, and `pas_area`.

## Application Scenarios

This project is designed for predicting land use changes in Protected Areas, including:
- `cropland_area` 
- `built_up_area` 
- `pas_area`

However, it can also be generalized to other zero-inflated regression tasks.

## File Structure

```
├── catboost.py                    # CatBoost multi-output regression model
├── lightgbm.py                    # LightGBM multi-output regression model
├── xgboost.py                     # XGBoost multi-output regression model
├── val.py                         # 10-fold cross-validation script
├── predict.py                     # Simplified prediction script
├── plot.py                        # Density scatter plot visualization script
└── README.md                      # Project documentation
```

## Core Features

### 1. Three Model Implementations

#### XGBoost Version (`xgboost.py`)
```python
from TwoStageEasyEnsembleXGBoost import TwoStageEasyEnsembleXGBoost

model = TwoStageEasyEnsembleXGBoost(
    easy_ensemble_n_estimators=10,
    cv=5,
    random_state=42,
    use_gpu=True
)
```

#### LightGBM Version (`lightgbm.py`)
```python
from TwoStageEasyEnsembleLightGBM import TwoStageEasyEnsembleLightGBM

model = TwoStageEasyEnsembleLightGBM(
    easy_ensemble_n_estimators=10,
    cv=5,
    random_state=42,
    use_gpu=True
)
```

#### CatBoost Version (`catboost.py`)
```python
from TwoStageEasyEnsembleCatBoost import TwoStageEasyEnsembleCatBoost

model = TwoStageEasyEnsembleCatBoost(
    easy_ensemble_n_estimators=10,
    cv=5,
    random_state=42,
    use_gpu=True
)
```

### 2. Main Features

- Bayesian hyperparameter optimization using scikit-optimize for automatic parameter tuning
- GPU acceleration support for improved training efficiency
- Built-in K-fold cross-validation functionality
- Effective handling of class imbalance problems
- Specifically designed for handling zero-inflated data
- Multi-output regression: simultaneous prediction of multiple target variables
- Model serialization: support for saving and loading models

### 3. Data Preprocessing

All models include complete data preprocessing pipelines:
- Categorical variable encoding (LabelEncoder)
- Missing value imputation (median imputation)
- Automatic feature type identification and processing

## Quick Start

### Requirements

```bash
# Basic dependencies
pip install numpy pandas scikit-learn imbalanced-learn scikit-optimize

# Choose the model you need
pip install xgboost      # XGBoost version
pip install lightgbm     # LightGBM version
pip install catboost     # CatBoost version

# Visualization dependencies
pip install matplotlib scipy
```

### Basic Usage

#### 1. Train Model

```python
import pandas as pd
from two_stage_xgboost_easyensemble import TwoStageEasyEnsembleXGBoost

# Load data
df = pd.read_csv('your_data.csv')

# Create model
model = TwoStageEasyEnsembleXGBoost(
    easy_ensemble_n_estimators=10,  # Number of ensemble learners
    cv=5,
    random_state=42,
    verbose=True,
    constraint_column='pa_total_area',
    use_gpu=True,
    zero_threshold=1e-6
)

# Prepare data
X, y_continuous, y_binary = model.prepare_data(df)

# Bayesian optimization for classifier
classifier_optimization = model.bayesian_optimize_classifier(
    X, y_binary, n_calls=30, cv_folds=5
)

# Bayesian optimization for regressor
regressor_optimization = model.bayesian_optimize_regressor(
    X, y_continuous, y_binary, n_calls=30, cv_folds=5
)

# Train final model
model.fit(X, y_continuous, y_binary)

# Save model
model.save_model("trained_model.pkl")
```

#### 2. Model Prediction

```python
from predict import SimpleEasyEnsemblePredictor

# Create predictor
predictor = SimpleEasyEnsemblePredictor("trained_model.pkl")

# Batch prediction
predictor.predict_csv(
    input_csv="test_data.csv",
    output_csv="predictions.csv",
    threshold=0.5
)
```

#### 3. Cross-Validation

```python
from val import OptimalParamsTenFoldCV

# Perform 10-fold cross-validation using trained model parameters
cv_validator = OptimalParamsTenFoldCV(
    model_file_path="trained_model.pkl",
    random_state=42,
    verbose=True
)

# Execute cross-validation
cv_results = cv_validator.ten_fold_cross_validation(df)

# Print results
cv_validator.print_comprehensive_results(cv_results)
```

#### 4. Results Visualization

```python
# Use plot.py to create density scatter plots
# Prepare CSV file containing true values and predicted values
# File should include columns: cropland_area, built_up_area, pas_area and their predictions
python plot.py
```

## Evaluation Metrics

The models provide comprehensive evaluation metrics:

### Classifier Performance
- Accuracy
- Precision
- Recall
- F1 Score

### Regressor Performance
- R² Score
- Adjusted R²
- Root Mean Square Error (RMSE)
- Mean Absolute Error (MAE)

## Core Classes and Methods

### TwoStageEasyEnsembleXGBoost/LightGBM/CatBoost

**Main Methods:**

- `prepare_data(df)`: Prepare training data
- `bayesian_optimize_classifier(X, y_binary, n_calls, cv_folds)`: Optimize classifier parameters
- `bayesian_optimize_regressor(X, y_continuous, y_binary, n_calls, cv_folds)`: Optimize regressor parameters
- `fit(X, y_continuous, y_binary)`: Train model
- `predict(X)`: Prediction (probability weighted)
- `predict_detailed(X)`: Detailed prediction results (including intermediate steps)
- `cross_validate(X, y_continuous, y_binary, cv_folds)`: Cross-validation
- `save_model(filepath)`: Save model
- `load_model(filepath)`: Load model

### OptimalParamsTenFoldCV

**Main Methods:**

- `ten_fold_cross_validation(df)`: Execute 10-fold cross-validation
- `print_comprehensive_results(cv_results)`: Print detailed results

### SimplePredictor

**Main Methods:**

- `predict_csv(input_csv, output_csv, threshold)`: Predict from CSV and save results

## Model Performance

The models excel in handling zero-inflated data:
- Effective handling of class imbalance problems
- Significantly improved zero-value prediction accuracy
- Bayesian optimization ensures parameter optimization
- GPU acceleration support for high training efficiency

## Advanced Configuration

### Custom Classifier Parameters

```python
classifier_params = {
    'max_depth': 6,
    'learning_rate': 0.1,
    'n_estimators': 100,
    'subsample': 0.8,
    'colsample_bytree': 0.8
}

model = TwoStageEasyEnsembleXGBoost(
    classifier_params=classifier_params
)
```

### Custom Regressor Parameters

```python
regressor_params = {
    'max_depth': 10,
    'learning_rate': 0.07,
    'n_estimators': 600,
    'subsample': 0.88
}

model = TwoStageEasyEnsembleXGBoost(
    regressor_params=regressor_params
)
```

### Bayesian Optimization Search Space

You can customize the search space by modifying the `create_bayesian_search_space_classifier()` and `create_bayesian_search_space_regressor()` methods in each model class.

## Data Format Requirements

Input data should be in CSV format, containing:
- **Feature columns**: Numerical or categorical features
- **Target columns**: Values to be predicted (e.g., cropland_area, built_up_area, pas_area)
- **Categorical features** (optional): desig_eng, desig_type, iucn_cat, gov_type, own_type, iso3

Supported categorical columns will be automatically encoded using LabelEncoder.

## Visualization

`plot.py` provides high-quality density scatter plot visualization:
- Density color mapping
- 1:1 baseline
- Regression line
- Evaluation metrics annotation (R², Adjusted R², MAE, RMSE)
- Intelligent data sampling (handling large datasets)
- Contrast enhancement

## Contributing

Issues and Pull Requests are welcome!


## Acknowledgments

This project uses the following excellent open-source libraries:
- [XGBoost](https://github.com/dmlc/xgboost)
- [LightGBM](https://github.com/microsoft/LightGBM)
- [CatBoost](https://github.com/catboost/catboost)
- [scikit-learn](https://scikit-learn.org/)
- [imbalanced-learn](https://imbalanced-learn.org/)
- [scikit-optimize](https://scikit-optimize.github.io/)

## Contact

For questions or suggestions, please contact via GitHub Issues.

---

**Note**: When using GPU training, please ensure CUDA and the corresponding GPU version libraries are correctly installed.
