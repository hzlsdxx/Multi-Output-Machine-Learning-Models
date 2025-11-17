import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.ensemble import EasyEnsembleClassifier
import xgboost as xgb
import time
import pickle
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')


class TwoStageEasyEnsembleXGBoost:

    def __init__(self, verbose=True):
        self.verbose = verbose
        self.target_names = ['cropland_area', 'built_up_area', 'pas_area']
        self.constraint_column = 'pa_total_area'
        self.zero_threshold = 1e-6
        self.label_encoders = {}
        self.feature_names = None
    
    def _preprocess_data(self, df: pd.DataFrame, fit_encoders: bool = True) -> pd.DataFrame:
        df_processed = df.copy()

        columns_to_drop = []
        if 'value' in df_processed.columns:
            columns_to_drop.append('value')
        if 'pa_total_area' in df_processed.columns and self.constraint_column != 'pa_total_area':
            columns_to_drop.append('pa_total_area')

        if columns_to_drop:
            df_processed = df_processed.drop(columns_to_drop, axis=1)

        categorical_cols = ['desig_eng', 'desig_type', 'iucn_cat', 'gov_type', 'own_type', 'iso3']

        for col in categorical_cols:
            if col in df_processed.columns:
                if fit_encoders:
                    le = LabelEncoder()
                    df_processed[col] = le.fit_transform(df_processed[col].astype(str))
                    self.label_encoders[col] = le
                    if self.verbose:
                        print(f"Encoding {col}: {len(le.classes_)} unique values")

        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
        df_processed[numeric_cols] = df_processed[numeric_cols].fillna(df_processed[numeric_cols].median())

        return df_processed

    def prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        df_processed = self._preprocess_data(df, fit_encoders=True)

        feature_cols = [col for col in df_processed.columns
                        if col not in self.target_names and col != self.constraint_column]

        X = df_processed[feature_cols]
        y_continuous = df_processed[self.target_names].values

        y_binary = (y_continuous > self.zero_threshold).astype(int)

        self.feature_names = X.columns.tolist()

        return X.values, y_continuous, y_binary


class OptimalParamsTenFoldCV:

    def __init__(self, 
                 model_file_path: str = None,
                 classifier_params: dict = None,
                 regressor_params: dict = None,
                 easy_ensemble_n_estimators: int = None,
                 random_state: int = 42,
                 verbose: bool = True,
                 use_gpu: bool = True,
                 zero_threshold: float = 1e-6):

        
        self.random_state = random_state
        self.verbose = verbose
        self.zero_threshold = zero_threshold
        self.use_gpu = use_gpu and self._check_gpu_availability()
        

        self.loaded_from_file = False
        if model_file_path:
            try:
                self._load_optimal_params_from_file(model_file_path)
                self.loaded_from_file = True
                if self.verbose:
                    print(f"Successfully loaded optimal parameters from {model_file_path}")
            except Exception as e:
                if self.verbose:
                    print(f"Failed to load parameters from file: {e}")
                    print("Using manually specified parameters or default parameters instead")

        if not self.loaded_from_file:
            self._set_default_or_provided_params(
                classifier_params, regressor_params, easy_ensemble_n_estimators
            )
        
        self.target_names = ['cropland_area', 'built_up_area', 'pas_area']
    
    def _load_optimal_params_from_file(self, filepath: str):
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.classifier_params = model_data.get('classifier_params', {}).copy()
        self.regressor_params = model_data.get('regressor_params', {}).copy()
        self.easy_ensemble_n_estimators = model_data.get('easy_ensemble_n_estimators', 10)
        
        if 'best_classifier_params' in model_data and model_data['best_classifier_params']:
            self.classifier_params.update(model_data['best_classifier_params'])


        if 'best_regressor_params' in model_data and model_data['best_regressor_params']:
            self.regressor_params.update(model_data['best_regressor_params'])


        if self.use_gpu:
            gpu_settings = {
                'tree_method': 'gpu_hist',
                'gpu_id': 0,
                'predictor': 'gpu_predictor'
            }
            self.classifier_params.update(gpu_settings)
            self.regressor_params.update(gpu_settings)
        else:
            gpu_keys = ['tree_method', 'gpu_id', 'predictor']
            for key in gpu_keys:
                self.classifier_params.pop(key, None)
                self.regressor_params.pop(key, None)
    
    def _set_default_or_provided_params(self, classifier_params, regressor_params, easy_ensemble_n_estimators):
        self.easy_ensemble_n_estimators = easy_ensemble_n_estimators or 20
        
        default_classifier_params = {
            'objective': 'binary:logistic',
            'booster': 'gbtree',
            'n_estimators': 200,      
            'learning_rate': 0.2,     
            'max_depth': 8,           
            'subsample': 0.70,        
            'colsample_bytree': 0.799,  
            'reg_alpha': 0.01,        
            'reg_lambda': 0.01,       
            'random_state': self.random_state,
            'n_jobs': -1,
            'verbosity': 0
        }
        
        default_regressor_params = {
            'objective': 'reg:squarederror',
            'booster': 'gbtree',
            'n_estimators': 1000,      
            'learning_rate': 0.0392,    
            'max_depth': 14,          
            'subsample': 0.7991,        
            'colsample_bytree': 0.7652, 
            'reg_alpha': 0.0193,        
            'reg_lambda': 0.1000,       
            'gamma': 0.1060,             
            'min_child_weight': 8,    
            'random_state': self.random_state,
            'n_jobs': -1,
            'verbosity': 0,
            'importance_type': 'gain'
        }
        
        if self.use_gpu:
            gpu_settings = {
                'tree_method': 'gpu_hist',
                'gpu_id': 0,
                'predictor': 'gpu_predictor'
            }
            default_classifier_params.update(gpu_settings)
            default_regressor_params.update(gpu_settings)
        
        if classifier_params:
            default_classifier_params.update(classifier_params)
        if regressor_params:
            default_regressor_params.update(regressor_params)
            
        self.classifier_params = default_classifier_params
        self.regressor_params = default_regressor_params
    
    def _check_gpu_availability(self) -> bool:
        try:
            import subprocess
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
            if result.returncode != 0:
                return False
            
            test_data = np.random.random((100, 10))
            test_target = np.random.random(100)
            
            model = xgb.XGBRegressor(
                tree_method='gpu_hist',
                gpu_id=0,
                n_estimators=10,
                verbosity=0
            )
            model.fit(test_data, test_target)
            return True
        except Exception:
            return False
    
    def calculate_adjusted_r2(self, y_true: np.ndarray, y_pred: np.ndarray, 
                            n_features: int) -> float:

        n_samples = len(y_true)
        r2 = r2_score(y_true, y_pred)
        
        if n_samples <= n_features + 1:
            return float('nan')  # 避免数值问题
        
        adjusted_r2 = 1 - (1 - r2) * (n_samples - 1) / (n_samples - n_features - 1)
        return adjusted_r2
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                         n_features: int) -> Dict[str, float]:

        r2 = r2_score(y_true, y_pred)
        adjusted_r2 = self.calculate_adjusted_r2(y_true, y_pred, n_features)
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        
        return {
            'R2': r2,
            'Adjusted_R2': adjusted_r2,
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse
        }
    
    def calculate_classification_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        return {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1': f1
        }
    
    def print_loaded_params(self):

        for param, value in self.classifier_params.items():
            if isinstance(value, float):
                print(f"  {param}: {value:.4f}")
            else:
                print(f"  {param}: {value}")
        
        for param, value in self.regressor_params.items():
            if isinstance(value, float):
                print(f"  {param}: {value:.4f}")
            else:
                print(f"  {param}: {value}")
    
    def ten_fold_cross_validation(self, X: np.ndarray, y_continuous: np.ndarray, 
                                y_binary: np.ndarray) -> Tuple[Dict[str, Dict[str, list]], Dict[str, Dict[str, list]], float]:

        if self.verbose:
            gpu_status = "GPU" if self.use_gpu else "CPU"
            print("=" * 80)
            print("Starting ten-fold cross-validation")
            print(f"Using device: {gpu_status}")
            print(f"Number of EasyEnsemble base learners: {self.easy_ensemble_n_estimators}")
            print(f"Number of features: {X.shape[1]}, Number of samples: {X.shape[0]}")
            print("=" * 80)

        results = {
            target: {
                'R2': [], 'Adjusted_R2': [], 'MAE': [], 'MSE': [], 'RMSE': []
            } for target in self.target_names
        }
        
        classification_results = {
            target: {
                'Accuracy': [], 'Precision': [], 'Recall': [], 'F1': []
            } for target in self.target_names
        }
        
        kfold = KFold(n_splits=10, shuffle=True, random_state=self.random_state)
        
        start_time = time.time()
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
            if self.verbose:
                print(f"\n处理第 {fold + 1}/10 折...")
            
            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_cont_train, y_cont_val = y_continuous[train_idx], y_continuous[val_idx]
            y_bin_train, y_bin_val = y_binary[train_idx], y_binary[val_idx]
            
            classifier_fold = {}
            for i, target in enumerate(self.target_names):
                easy_ensemble = EasyEnsembleClassifier(
                    estimator=xgb.XGBClassifier(**self.classifier_params),
                    n_estimators=self.easy_ensemble_n_estimators,
                    random_state=self.random_state,
                    n_jobs=1
                )
                easy_ensemble.fit(X_train_fold, y_bin_train[:, i])
                classifier_fold[target] = easy_ensemble
            
            positive_mask = np.any(y_bin_train > 0, axis=1)
            
            if np.sum(positive_mask) > 0:
                X_pos_train = X_train_fold[positive_mask]
                y_pos_train = y_cont_train[positive_mask]
                
                regressor_fold = MultiOutputRegressor(
                    xgb.XGBRegressor(**self.regressor_params),
                    n_jobs=1
                )
                regressor_fold.fit(X_pos_train, y_pos_train)
            else:
                if self.verbose:
                    print(f"  Warning: No positive samples in fold {fold+1}, skipping")
                continue

            probabilities = np.zeros((X_val_fold.shape[0], len(self.target_names)))
            for i, target in enumerate(self.target_names):
                prob = classifier_fold[target].predict_proba(X_val_fold)
                probabilities[:, i] = prob[:, 1]
            
            classification_pred = np.zeros((X_val_fold.shape[0], len(self.target_names)))
            for i, target in enumerate(self.target_names):
                classification_pred[:, i] = classifier_fold[target].predict(X_val_fold)
            
            reg_pred = regressor_fold.predict(X_val_fold)
            reg_pred = np.maximum(reg_pred, 0.0)
            
            final_pred = probabilities * reg_pred
            
            for i, target in enumerate(self.target_names):
                metrics = self.calculate_metrics(
                    y_cont_val[:, i], 
                    final_pred[:, i], 
                    X.shape[1]
                )
                
                classification_metrics = self.calculate_classification_metrics(
                    y_bin_val[:, i],
                    classification_pred[:, i]
                )
                
                for metric_name, metric_value in metrics.items():
                    results[target][metric_name].append(metric_value)
                
                for metric_name, metric_value in classification_metrics.items():
                    classification_results[target][metric_name].append(metric_value)

                if self.verbose:
                    print(f"  {target}:")
                    print(f"    Regression - R²: {metrics['R2']:.4f}, Adj-R²: {metrics['Adjusted_R2']:.4f}")
                    print(
                        f"    Regression - MAE: {metrics['MAE']:.2f}, MSE: {metrics['MSE']:.2f}, RMSE: {metrics['RMSE']:.2f}")
                    print(
                        f"    Classification - Acc: {classification_metrics['Accuracy']:.4f}, F1: {classification_metrics['F1']:.4f}")
                    print(
                        f"    Classification - Prec: {classification_metrics['Precision']:.4f}, Rec: {classification_metrics['Recall']:.4f}")

        total_time = time.time() - start_time

        if self.verbose:
            print(f"\nTen-fold cross-validation completed! Total time: {total_time:.2f} seconds")

        return results, classification_results, total_time
    
    def summarize_results(self, results: Dict[str, Dict[str, list]], 
                         classification_results: Dict[str, Dict[str, list]],
                         total_time: float) -> Tuple[pd.DataFrame, pd.DataFrame]:

        
        regression_summary_data = []
        
        for target in self.target_names:
            target_results = results[target]
            
            row = {'Target': target}
            
            for metric in ['R2', 'Adjusted_R2', 'MAE', 'MSE', 'RMSE']:
                values = target_results[metric]
                values_clean = [v for v in values if not np.isnan(v)]
                
                if values_clean:
                    mean_val = np.mean(values_clean)
                    std_val = np.std(values_clean)
                    row[f'{metric}_Mean'] = mean_val
                    row[f'{metric}_Std'] = std_val
                else:
                    row[f'{metric}_Mean'] = np.nan
                    row[f'{metric}_Std'] = np.nan
            
            regression_summary_data.append(row)
        
        overall_row = {'Target': 'Overall'}
        for metric in ['R2', 'Adjusted_R2', 'MAE', 'MSE', 'RMSE']:
            all_values = []
            for target in self.target_names:
                values = results[target][metric]
                values_clean = [v for v in values if not np.isnan(v)]
                all_values.extend(values_clean)
            
            if all_values:
                overall_row[f'{metric}_Mean'] = np.mean(all_values)
                overall_row[f'{metric}_Std'] = np.std(all_values)
            else:
                overall_row[f'{metric}_Mean'] = np.nan
                overall_row[f'{metric}_Std'] = np.nan
        
        regression_summary_data.append(overall_row)
        regression_summary_df = pd.DataFrame(regression_summary_data)
        
        classification_summary_data = []
        
        for target in self.target_names:
            target_results = classification_results[target]
            
            row = {'Target': target}
            
            for metric in ['Accuracy', 'Precision', 'Recall', 'F1']:
                values = target_results[metric]
                values_clean = [v for v in values if not np.isnan(v)]
                
                if values_clean:
                    mean_val = np.mean(values_clean)
                    std_val = np.std(values_clean)
                    row[f'{metric}_Mean'] = mean_val
                    row[f'{metric}_Std'] = std_val
                else:
                    row[f'{metric}_Mean'] = np.nan
                    row[f'{metric}_Std'] = np.nan
            
            classification_summary_data.append(row)
        
        overall_classification_row = {'Target': 'Overall'}
        for metric in ['Accuracy', 'Precision', 'Recall', 'F1']:
            all_values = []
            for target in self.target_names:
                values = classification_results[target][metric]
                values_clean = [v for v in values if not np.isnan(v)]
                all_values.extend(values_clean)
            
            if all_values:
                overall_classification_row[f'{metric}_Mean'] = np.mean(all_values)
                overall_classification_row[f'{metric}_Std'] = np.std(all_values)
            else:
                overall_classification_row[f'{metric}_Mean'] = np.nan
                overall_classification_row[f'{metric}_Std'] = np.nan
        
        classification_summary_data.append(overall_classification_row)
        classification_summary_df = pd.DataFrame(classification_summary_data)

        if self.verbose:
            print("\n" + "=" * 120)
            print("Ten-fold cross-validation results summary")
            print("=" * 120)

            print("\nRegression performance metrics:")
            print(regression_summary_df.round(4).to_string(index=False))

            print("\nClassification performance metrics:")
            print(classification_summary_df.round(4).to_string(index=False))

            print(f"\nTotal time: {total_time:.2f} seconds")
            print(f"Number of EasyEnsemble base learners: {self.easy_ensemble_n_estimators}")
            print(f"Using GPU: {'Yes' if self.use_gpu else 'No'}")
            print(
                f"Parameter source: {'Loaded from saved model file' if self.loaded_from_file else 'Manually specified/default parameters'}")

        return regression_summary_df, classification_summary_df

    def detailed_analysis(self, results: Dict[str, Dict[str, list]],
                          classification_results: Dict[str, Dict[str, list]]) -> None:


        for target in self.target_names:
            print(f"\nDetailed analysis for {target}:")
            print("-" * 50)

            # Regression performance analysis
            print("Regression Performance:")
            target_results = results[target]

            for metric in ['R2', 'Adjusted_R2', 'MAE', 'MSE', 'RMSE']:
                values = target_results[metric]
                values_clean = [v for v in values if not np.isnan(v)]

                if values_clean:
                    mean_val = np.mean(values_clean)
                    std_val = np.std(values_clean)
                    min_val = np.min(values_clean)
                    max_val = np.max(values_clean)
                    median_val = np.median(values_clean)

                    print(f"  {metric:12s}: {mean_val:8.4f} ± {std_val:6.4f} "
                          f"[{min_val:8.4f}, {max_val:8.4f}] (Median: {median_val:8.4f})")
                else:
                    print(f"  {metric:12s}: No valid values")

            # Classification performance analysis
            print("\nClassification Performance:")
            classification_target_results = classification_results[target]

            for metric in ['accuracy', 'precision', 'recall', 'f1']:
                values = classification_target_results[metric]
                values_clean = [v for v in values if not np.isnan(v)]

                if values_clean:
                    mean_val = np.mean(values_clean)
                    std_val = np.std(values_clean)
                    min_val = np.min(values_clean)
                    max_val = np.max(values_clean)
                    median_val = np.median(values_clean)

                    print(f"  {metric:12s}: {mean_val:8.4f} ± {std_val:6.4f} "
                          f"[{min_val:8.4f}, {max_val:8.4f}] (Median: {median_val:8.4f})")
                else:
                    print(f"  {metric:12s}: No valid values")

            for metric in ['Accuracy', 'Precision', 'Recall', 'F1']:
                values = classification_target_results[metric]
                values_clean = [v for v in values if not np.isnan(v)]

                if values_clean:
                    mean_val = np.mean(values_clean)
                    std_val = np.std(values_clean)
                    min_val = np.min(values_clean)
                    max_val = np.max(values_clean)
                    median_val = np.median(values_clean)

                    print(f"  {metric:12s}: {mean_val:8.4f} ± {std_val:6.4f} "
                          f"[{min_val:8.4f}, {max_val:8.4f}] (Median: {median_val:8.4f})")
                else:
                    print(f"  {metric:12s}: No valid values")


def main_ten_fold_cv_with_trained_params():
    print("=" * 80)
    print("Two-Stage EasyEnsemble-XGBoost Model Ten-Fold Cross Validation")
    print("Evaluating using the optimal parameters from the trained model")
    print("=" * 80)

    # Load data
    try:
        df = pd.read_csv('PA_1km_drop_AF_new.csv')
        print(f"Data loaded successfully, shape: {df.shape}")
    except FileNotFoundError:
        print("Please replace the CSV file path with the correct one")
        return

    try:
        cv_evaluator = OptimalParamsTenFoldCV(
            model_file_path="final_AF_two_stage_easy_ensemble.pkl",
            random_state=42,
            verbose=True,
            use_gpu=True,
            zero_threshold=1e-6
        )
    except:
        print("Unable to load parameters from the model file, using default optimal parameters...")

        optimal_classifier_params = {
            'n_estimators': 200,      
            'learning_rate': 0.2,     
            'max_depth': 8,           
            'subsample': 0.70,        
            'colsample_bytree': 0.799,  
            'reg_alpha': 0.01,        
            'reg_lambda': 0.01,       
        }
        
        optimal_regressor_params = {
            'n_estimators': 1000,      
            'learning_rate': 0.0392,    
            'max_depth': 14,          
            'subsample': 0.7991,        
            'colsample_bytree': 0.7652, 
            'reg_alpha': 0.0193,        
            'reg_lambda': 0.1000,       
            'gamma': 0.1060,             
            'min_child_weight': 8,    
        }
        
        cv_evaluator = OptimalParamsTenFoldCV(
            classifier_params=optimal_classifier_params,
            regressor_params=optimal_regressor_params,
            easy_ensemble_n_estimators=20,
            random_state=42,
            verbose=True,
            use_gpu=True,
            zero_threshold=1e-6
        )
    
    cv_evaluator.print_loaded_params()
    
    temp_model = TwoStageEasyEnsembleXGBoost(verbose=True)
    X, y_continuous, y_binary = temp_model.prepare_data(df)

    print(f"\nData preprocessing completed:")
    print(f"Number of features: {X.shape[1]}, Number of samples: {X.shape[0]}")
    if temp_model.feature_names and len(temp_model.feature_names) >= 5:
        print(f"Example feature names: {temp_model.feature_names[:5]}...")

    target_names = ['cropland_area', 'built_up_area', 'pas_area']
    for i, target in enumerate(target_names):
        zero_count = np.sum(y_binary[:, i] == 0)
        positive_count = np.sum(y_binary[:, i] == 1)
        print(f"{target}: Zero-value samples {zero_count}, Positive samples {positive_count} "
              f"(Positive sample ratio: {positive_count / (zero_count + positive_count) * 100:.1f}%)")

    results, classification_results, total_time = cv_evaluator.ten_fold_cross_validation(X, y_continuous, y_binary)

    regression_summary_df, classification_summary_df = cv_evaluator.summarize_results(results, classification_results,
                                                                                      total_time)

    cv_evaluator.detailed_analysis(results, classification_results)

    regression_summary_df.to_csv('ten_fold_cv_regression_results.csv', index=False)
    classification_summary_df.to_csv('ten_fold_cv_classification_results.csv', index=False)
    print(f"\nResults have been saved to:")
    print(f"  Regression results: ten_fold_cv_regression_results.csv")
    print(f"  Classification results: ten_fold_cv_classification_results.csv")

    return results, classification_results, regression_summary_df, classification_summary_df


if __name__ == "__main__":
    main_ten_fold_cv_with_trained_params()