import numpy as np
from sklearn.multioutput import MultiOutputRegressor, MultiOutputClassifier
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, accuracy_score, classification_report, f1_score
from imblearn.ensemble import EasyEnsembleClassifier
import lightgbm as lgb
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Any
import warnings
import pickle
import time
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
import pandas as pd
warnings.filterwarnings('ignore')


class TwoStageEasyEnsembleLightGBM:


    def __init__(self,
                 classifier_params: dict = None,
                 regressor_params: dict = None,
                 easy_ensemble_n_estimators: int = 10,
                 cv: int = 5,
                 random_state: int = 42,
                 verbose: bool = True,
                 constraint_column: str = None,
                 use_gpu: bool = True,
                 zero_threshold: float = 1e-6,
                 n_jobs: int = -1):


        self.cv = cv
        self.random_state = random_state
        self.verbose = verbose
        self.constraint_column = constraint_column
        self.zero_threshold = zero_threshold
        self.easy_ensemble_n_estimators = easy_ensemble_n_estimators
        self.n_jobs = n_jobs

        self.use_gpu = use_gpu and self._check_gpu_availability()
        if self.use_gpu and self.verbose:
            print("GPU acceleration has been enabled")
        elif use_gpu and not self.use_gpu and verbose:
            print("GPU not available, using CPU for training")

        default_classifier_params = {
            'objective': 'binary',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.01,
            'reg_lambda': 0.01,
            'random_state': random_state,
            'n_jobs': self.n_jobs,
            'verbosity': -1,
            'force_row_wise': True
        }

        default_regressor_params = {
            'objective': 'regression',
            'boosting_type': 'gbdt',
            'num_leaves': 127,
            'learning_rate': 0.07,
            'n_estimators': 600,
            'subsample': 0.88,
            'colsample_bytree': 0.88,
            'reg_alpha': 0.015,
            'reg_lambda': 0.015,
            'min_child_weight': 3,
            'random_state': random_state,
            'n_jobs': self.n_jobs,
            'verbosity': -1,
            'importance_type': 'gain',
            'force_row_wise': True
        }

        if self.use_gpu:
            gpu_settings = {
                'device': 'gpu',
                'gpu_platform_id': 0,
                'gpu_device_id': 0
            }
            default_classifier_params.update(gpu_settings)
            default_regressor_params.update(gpu_settings)

        if classifier_params:
            default_classifier_params.update(classifier_params)
        if regressor_params:
            default_regressor_params.update(regressor_params)

        self.classifier_params = default_classifier_params
        self.regressor_params = default_regressor_params

        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.classifier = None
        self.regressor = None
        self.feature_names = None
        self.target_names = ['cropland_area', 'built_up_area', 'pas_area']

        self.best_classifier_params = None
        self.best_regressor_params = None
        self.optimization_history = []

    def _check_gpu_availability(self) -> bool:
        try:
            test_data = np.random.random((100, 10))
            test_target = np.random.random(100)

            train_data = lgb.Dataset(test_data, label=test_target)

            params = {
                'objective': 'regression',
                'device': 'gpu',
                'verbosity': -1,
                'num_iterations': 1
            }

            model = lgb.train(params, train_data, valid_sets=[train_data],
                            callbacks=[lgb.early_stopping(1), lgb.log_evaluation(0)])
            return True
        except Exception:
            return False

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
                else:
                    if col in self.label_encoders:
                        unique_vals = set(df_processed[col].astype(str))
                        known_vals = set(self.label_encoders[col].classes_)
                        unknown_vals = unique_vals - known_vals

                        if unknown_vals:
                            if self.verbose:
                                print(f"Warning: Unknown categories in {col}: {unknown_vals}")
                            most_common = self.label_encoders[col].classes_[0]
                            df_processed[col] = df_processed[col].astype(str).replace(
                                list(unknown_vals), most_common
                            )

                        df_processed[col] = self.label_encoders[col].transform(df_processed[col].astype(str))

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

    def create_bayesian_search_space_classifier(self) -> List:
        space = [
            Integer(5, 20, name='easy_ensemble_n_estimators'),
            Integer(50, 200, name='n_estimators'),
            Real(0.05, 0.2, name='learning_rate'),
            Integer(15, 63, name='num_leaves'),
            Real(0.7, 1.0, name='subsample'),
            Real(0.7, 1.0, name='feature_fraction'),
        ]
        return space

    def create_bayesian_search_space_regressor(self) -> List:
        space = [
            Integer(400, 1000, name='n_estimators'),
            Real(0.01, 0.2, name='learning_rate'),
            Integer(63, 255, name='num_leaves'),
            Real(0.7, 1.0, name='subsample'),
            Real(0.7, 1.0, name='feature_fraction'),
            Integer(1, 10, name='min_child_weight'),
        ]
        return space

    def bayesian_optimize_classifier(self, X: np.ndarray, y_binary: np.ndarray,
                                     n_calls: int = 20, cv_folds: int = 5) -> Dict[str, Any]:

        if self.verbose:
            print("Starting Bayesian parameter optimization for EasyEnsemble classifier...")
            print(f"Optimization iterations: {n_calls}, Cross-validation folds: {cv_folds}")
            print("Optimization objective: F1 score")
            print("Optimizing parameters: easy_ensemble_n_estimators, n_estimators, learning_rate, num_leaves")

        space = self.create_bayesian_search_space_classifier()
        classifier_history = []
        kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)

        @use_named_args(space)
        def objective(**params):
            easy_ensemble_n_est = params.pop('easy_ensemble_n_estimators')

            current_params = self.classifier_params.copy()
            current_params.update(params)

            f1_scores = []
            for train_idx, val_idx in kfold.split(X):
                X_train_fold, X_val_fold = X[train_idx], X[val_idx]
                y_train_fold, y_val_fold = y_binary[train_idx], y_binary[val_idx]

                fold_f1_scores = []
                for i in range(y_binary.shape[1]):
                    easy_ensemble = EasyEnsembleClassifier(
                        estimator=lgb.LGBMClassifier(**current_params),
                        n_estimators=easy_ensemble_n_est,
                        random_state=self.random_state,
                        n_jobs=1
                    )

                    easy_ensemble.fit(X_train_fold, y_train_fold[:, i])
                    y_pred_fold = easy_ensemble.predict(X_val_fold)

                    f1 = f1_score(y_val_fold[:, i], y_pred_fold, zero_division=0)
                    fold_f1_scores.append(f1)

                f1_scores.append(np.mean(fold_f1_scores))

            mean_f1 = np.mean(f1_scores)

            classifier_history.append({
                'params': {**params, 'easy_ensemble_n_estimators': easy_ensemble_n_est},
                'f1_score': mean_f1,
                'iteration': len(classifier_history) + 1
            })

            if self.verbose and len(classifier_history) % 5 == 0:
                print(f"EasyEnsemble iteration {len(classifier_history)}: F1 Score: {mean_f1:.4f}")

            return -mean_f1

        start_time = time.time()
        result = gp_minimize(
            func=objective,
            dimensions=space,
            n_calls=n_calls,
            random_state=self.random_state,
            acq_func='EI',
            n_jobs=1
        )

        optimization_time = time.time() - start_time

        best_params_values = result.x
        param_names = [dim.name for dim in space]
        best_params_dict = dict(zip(param_names, best_params_values))

        self.easy_ensemble_n_estimators = best_params_dict.pop('easy_ensemble_n_estimators')
        self.best_classifier_params = best_params_dict

        self.classifier_params.update(self.best_classifier_params)

        optimization_result = {
            'best_params': self.best_classifier_params,
            'best_easy_ensemble_n_estimators': self.easy_ensemble_n_estimators,
            'best_f1_score': -result.fun,
            'optimization_time': optimization_time,
            'n_iterations': n_calls,
            'optimization_history': classifier_history.copy(),
            'convergence_trace': result.func_vals
        }

        if self.verbose:
            print(f"\nBayesian optimization for EasyEnsemble classifier is complete!")
            print(f"Optimization time: {optimization_time:.2f} seconds")
            print(f"Best F1 score: {-result.fun:.4f}")
            print(f"Best number of EasyEnsemble base estimators: {self.easy_ensemble_n_estimators}")
            print(f"Best LightGBM parameters:")

            for param, value in self.best_classifier_params.items():
                if isinstance(value, float):
                    print(f"  {param}: {value:.4f}")
                else:
                    print(f"  {param}: {value}")

        return optimization_result

    def bayesian_optimize_regressor(self, X: np.ndarray, y_continuous: np.ndarray, y_binary: np.ndarray,
                                  n_calls: int = 25, cv_folds: int = 5) -> Dict[str, Any]:
        if self.verbose:
            print("Starting Bayesian parameter optimization for the regressor...")
            print(f"Number of optimization iterations: {n_calls}, number of cross-validation folds: {cv_folds}")
            print("Optimization parameters: n_estimators, learning_rate, num_leaves, subsample")

        space = self.create_bayesian_search_space_regressor()
        regressor_history = []
        kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)

        @use_named_args(space)
        def objective(**params):
            current_params = self.regressor_params.copy()
            current_params.update(params)

            positive_mask = np.any(y_binary > 0, axis=1)
            X_positive = X[positive_mask]
            y_positive = y_continuous[positive_mask]

            if len(X_positive) == 0:
                return 1000

            regressor = MultiOutputRegressor(
                lgb.LGBMRegressor(**current_params),
                n_jobs=1
            )

            r2_scores = []
            kfold_positive = KFold(n_splits=min(cv_folds, len(X_positive)),
                                 shuffle=True, random_state=self.random_state)

            for train_idx, val_idx in kfold_positive.split(X_positive):
                X_train_fold = X_positive[train_idx]
                X_val_fold = X_positive[val_idx]
                y_train_fold = y_positive[train_idx]
                y_val_fold = y_positive[val_idx]

                regressor.fit(X_train_fold, y_train_fold)
                y_pred_fold = regressor.predict(X_val_fold)

                y_pred_fold = np.maximum(y_pred_fold, 0.0)

                fold_r2_scores = []
                for i in range(y_positive.shape[1]):
                    if len(np.unique(y_val_fold[:, i])) > 1:
                        r2 = r2_score(y_val_fold[:, i], y_pred_fold[:, i])
                        fold_r2_scores.append(r2)

                if fold_r2_scores:
                    r2_scores.append(np.mean(fold_r2_scores))

            mean_r2 = np.mean(r2_scores) if r2_scores else -1

            regressor_history.append({
                'params': params.copy(),
                'r2_score': mean_r2,
                'iteration': len(regressor_history) + 1
            })

            if self.verbose and len(regressor_history) % 5 == 0:
                print(f"Regressor iteration {len(regressor_history)}, R²: {mean_r2:.4f}")

            return -mean_r2

        start_time = time.time()
        result = gp_minimize(
            func=objective,
            dimensions=space,
            n_calls=n_calls,
            random_state=self.random_state,
            acq_func='EI',
            n_jobs=1
        )

        optimization_time = time.time() - start_time

        best_params_values = result.x
        param_names = [dim.name for dim in space]
        self.best_regressor_params = dict(zip(param_names, best_params_values))

        self.regressor_params.update(self.best_regressor_params)

        optimization_result = {
            'best_params': self.best_regressor_params,
            'best_r2_score': -result.fun,
            'optimization_time': optimization_time,
            'n_iterations': n_calls,
            'optimization_history': regressor_history.copy(),
            'convergence_trace': result.func_vals
        }

        if self.verbose:
            print(f"\nBayesian optimization for regressor completed!")
            print(f"Optimization time: {optimization_time:.2f} seconds")
            print(f"Best R² score: {-result.fun:.4f}")
            print(f"Best parameters:")

            for param, value in self.best_regressor_params.items():
                if isinstance(value, float):
                    print(f"  {param}: {value:.4f}")
                else:
                    print(f"  {param}: {value}")

        return optimization_result

    def fit(self, X: np.ndarray, y_continuous: np.ndarray, y_binary: np.ndarray):
        if self.verbose:
            gpu_status = "GPU" if self.use_gpu else "CPU"
            print(f"Starting to train the Two-Stage EasyEnsemble model... (Using {gpu_status})")
            print(f"Number of EasyEnsemble base learners: {self.easy_ensemble_n_estimators}")
            print(f"Number of parallel threads: {self.n_jobs}")

        start_time = time.time()

        if self.verbose:
            print("Training the EasyEnsemble classifier...")

        self.classifier = {}
        for i, target in enumerate(self.target_names):
            easy_ensemble = EasyEnsembleClassifier(
                estimator=lgb.LGBMClassifier(**self.classifier_params),
                n_estimators=self.easy_ensemble_n_estimators,
                random_state=self.random_state,
                n_jobs=1
            )
            easy_ensemble.fit(X, y_binary[:, i])
            self.classifier[target] = easy_ensemble

        if self.verbose:
            print("Training the regressor...")

        positive_mask = np.any(y_binary > 0, axis=1)
        X_positive = X[positive_mask]
        y_positive = y_continuous[positive_mask]
        
        self.regressor = MultiOutputRegressor(
            lgb.LGBMRegressor(**self.regressor_params),
            n_jobs=1
        )
        self.regressor.fit(X_positive, y_positive)
        
        training_time = time.time() - start_time
        if self.verbose:
            print(f"EasyEnsemble model training completed! Training time: {training_time:.2f} seconds")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """EasyEnsemble soft classification prediction"""
        if self.classifier is None or self.regressor is None:
            raise ValueError("The model has not been trained yet, please call the fit method first")

        probabilities = np.zeros((X.shape[0], len(self.target_names)))
        for i, target in enumerate(self.target_names):
            prob = self.classifier[target].predict_proba(X)
            probabilities[:, i] = prob[:, 1]

        reg_pred = self.regressor.predict(X)
        reg_pred = np.maximum(reg_pred, 0.0)
        
        final_pred = probabilities * reg_pred
        
        return final_pred

    def predict_detailed(self, X: np.ndarray) -> Dict[str, np.ndarray]:


        probabilities = np.zeros((X.shape[0], len(self.target_names)))
        for i, target in enumerate(self.target_names):
            prob = self.classifier[target].predict_proba(X)
            probabilities[:, i] = prob[:, 1]

        reg_pred = self.regressor.predict(X)
        reg_pred = np.maximum(reg_pred, 0.0)
        
        final_pred = probabilities * reg_pred
        
        return {
            'probabilities': probabilities,
            'regression_pred': reg_pred,
            'final_pred': final_pred
        }

    def cross_validate(self, X: np.ndarray, y_continuous: np.ndarray, y_binary: np.ndarray, 
                       cv_folds: int = 5) -> dict:
        kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        results = {target: {'r2': [], 'mse': [], 'mae': [], 'rmse': []}
                  for target in self.target_names}
        
        classifier_results = {target: {'accuracy': [], 'precision': [], 'recall': [], 'f1': []} 
                            for target in self.target_names}

        if self.verbose:
            gpu_status = "GPU" if self.use_gpu else "CPU"
            print(f"Starting {cv_folds}-fold cross-validation... (Using {gpu_status})")
            print(f"Number of EasyEnsemble base learners: {self.easy_ensemble_n_estimators}")
            print(f"Number of parallel threads: {self.n_jobs}")

        start_time = time.time()
        for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
            if self.verbose:
                print(f"Fold {fold + 1}/{cv_folds}...")

            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_cont_train, y_cont_val = y_continuous[train_idx], y_continuous[val_idx]
            y_bin_train, y_bin_val = y_binary[train_idx], y_binary[val_idx]

            classifier_fold = {}
            for i, target in enumerate(self.target_names):
                easy_ensemble = EasyEnsembleClassifier(
                    estimator=lgb.LGBMClassifier(**self.classifier_params),
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
                    lgb.LGBMRegressor(**self.regressor_params),
                    n_jobs=1
                )
                regressor_fold.fit(X_pos_train, y_pos_train)
            else:
                continue

            probabilities = np.zeros((X_val_fold.shape[0], len(self.target_names)))
            for i, target in enumerate(self.target_names):
                prob = classifier_fold[target].predict_proba(X_val_fold)
                probabilities[:, i] = prob[:, 1]

            reg_pred = regressor_fold.predict(X_val_fold)
            reg_pred = np.maximum(reg_pred, 0.0)
            
            final_pred = probabilities * reg_pred

            for i, target in enumerate(self.target_names):
                r2 = r2_score(y_cont_val[:, i], final_pred[:, i])
                mse = mean_squared_error(y_cont_val[:, i], final_pred[:, i])
                mae = mean_absolute_error(y_cont_val[:, i], final_pred[:, i])
                rmse = np.sqrt(mse)
                
                results[target]['r2'].append(r2)
                results[target]['mse'].append(mse)
                results[target]['mae'].append(mae)
                results[target]['rmse'].append(rmse)

            y_bin_pred = np.zeros_like(y_bin_val)
            for i, target in enumerate(self.target_names):
                y_bin_pred[:, i] = classifier_fold[target].predict(X_val_fold)
            
            for i, target in enumerate(self.target_names):
                from sklearn.metrics import precision_score, recall_score
                acc = accuracy_score(y_bin_val[:, i], y_bin_pred[:, i])
                prec = precision_score(y_bin_val[:, i], y_bin_pred[:, i], zero_division=0)
                rec = recall_score(y_bin_val[:, i], y_bin_pred[:, i], zero_division=0)
                f1 = f1_score(y_bin_val[:, i], y_bin_pred[:, i], zero_division=0)
                
                classifier_results[target]['accuracy'].append(acc)
                classifier_results[target]['precision'].append(prec)
                classifier_results[target]['recall'].append(rec)
                classifier_results[target]['f1'].append(f1)

        cv_time = time.time() - start_time
        if self.verbose:
            print(f"Cross-validation completed! Total time: {cv_time:.2f} seconds")

        cv_results = {}
        for target in self.target_names:
            cv_results[target] = {
                'r2_mean': np.mean(results[target]['r2']),
                'r2_std': np.std(results[target]['r2']),
                'mse_mean': np.mean(results[target]['mse']),
                'mse_std': np.std(results[target]['mse']),
                'mae_mean': np.mean(results[target]['mae']),
                'mae_std': np.std(results[target]['mae']),
                'rmse_mean': np.mean(results[target]['rmse']),
                'rmse_std': np.std(results[target]['rmse']),
                'accuracy_mean': np.mean(classifier_results[target]['accuracy']),
                'precision_mean': np.mean(classifier_results[target]['precision']),
                'recall_mean': np.mean(classifier_results[target]['recall']),
                'f1_mean': np.mean(classifier_results[target]['f1']),
                'f1_std': np.std(classifier_results[target]['f1'])
            }

        overall_r2 = np.mean([cv_results[target]['r2_mean'] for target in self.target_names])
        overall_accuracy = np.mean([cv_results[target]['accuracy_mean'] for target in self.target_names])
        overall_f1 = np.mean([cv_results[target]['f1_mean'] for target in self.target_names])

        cv_results['overall'] = {
            'r2_mean': overall_r2,
            'accuracy_mean': overall_accuracy,
            'f1_mean': overall_f1,
            'cv_time': cv_time,
            'easy_ensemble_n_estimators': self.easy_ensemble_n_estimators
        }

        return cv_results

    def get_feature_importance(self) -> Dict[str, pd.DataFrame]:

        importance_results = {}
        
        classifier_importance = pd.DataFrame()
        for i, target in enumerate(self.target_names):
            easy_ensemble = self.classifier[target]
            
            all_importances = []
            for estimator in easy_ensemble.estimators_:
                if hasattr(estimator, 'feature_importances_'):
                    all_importances.append(estimator.feature_importances_)
            
            if all_importances:
                avg_importance = np.mean(all_importances, axis=0)
                temp_df = pd.DataFrame({
                    'feature': self.feature_names,
                    'importance': avg_importance,
                    'target': target,
                    'model_type': 'easy_ensemble_classifier'
                })
                classifier_importance = pd.concat([classifier_importance, temp_df], ignore_index=True)
        
        regressor_importance = pd.DataFrame()
        for i, (target, estimator) in enumerate(zip(self.target_names, self.regressor.estimators_)):
            if hasattr(estimator, 'feature_importances_'):
                temp_df = pd.DataFrame({
                    'feature': self.feature_names,
                    'importance': estimator.feature_importances_,
                    'target': target,
                    'model_type': 'regressor'
                })
                regressor_importance = pd.concat([regressor_importance, temp_df], ignore_index=True)
        
        importance_results['classifier'] = classifier_importance
        importance_results['regressor'] = regressor_importance
        
        return importance_results

    def save_model(self, filepath: str):
        model_data = {
            'classifier': self.classifier,
            'regressor': self.regressor,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_names': self.feature_names,
            'target_names': self.target_names,
            'classifier_params': self.classifier_params,
            'regressor_params': self.regressor_params,
            'constraint_column': self.constraint_column,
            'use_gpu': self.use_gpu,
            'zero_threshold': self.zero_threshold,
            'best_classifier_params': self.best_classifier_params,
            'best_regressor_params': self.best_regressor_params,
            'optimization_history': self.optimization_history,
            'easy_ensemble_n_estimators': self.easy_ensemble_n_estimators,
            'n_jobs': self.n_jobs
        }

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

        if self.verbose:
            print(f"模型已保存到: {filepath}")

    def load_model(self, filepath: str):
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        self.classifier = model_data['classifier']
        self.regressor = model_data['regressor']
        self.scaler = model_data['scaler']
        self.label_encoders = model_data['label_encoders']
        self.feature_names = model_data['feature_names']
        self.target_names = model_data['target_names']
        self.classifier_params = model_data['classifier_params']
        self.regressor_params = model_data['regressor_params']
        self.constraint_column = model_data['constraint_column']
        self.use_gpu = model_data.get('use_gpu', False)
        self.zero_threshold = model_data.get('zero_threshold', 1e-6)
        self.best_classifier_params = model_data.get('best_classifier_params', None)
        self.best_regressor_params = model_data.get('best_regressor_params', None)
        self.optimization_history = model_data.get('optimization_history', [])
        self.easy_ensemble_n_estimators = model_data.get('easy_ensemble_n_estimators', 10)
        self.n_jobs = model_data.get('n_jobs', -1)

        if self.verbose:
            print(f"Model loaded from {filepath}")

    def print_comprehensive_results(self, cv_results: dict):
        """Print comprehensive results"""
        print("\n" + "=" * 80)
        print("Two-Stage EasyEnsemble-LightGBM Multi-output Regression Model Comprehensive Results")
        print("=" * 80)

        # Classifier performance
        print(f"\nEasyEnsemble Classifier Performance (Number of Base Learners: {self.easy_ensemble_n_estimators}):")
        print(f"{'Target Variable':<15} {'Accuracy':<8} {'Precision':<8} {'Recall':<8} {'F1 Score':<8}")
        print("-" * 60)
        for target in self.target_names:
            results = cv_results[target]
            print(f"{target:<15} {results['accuracy_mean']:.4f}   {results['precision_mean']:.4f}   "
                  f"{results['recall_mean']:.4f}   {results['f1_mean']:.4f}")

        # Regressor performance
        print(f"\nRegressor Performance:")
        print(f"{'Target Variable':<15} {'R² Mean':<8} {'R² Std':<8} {'RMSE Mean':<10} {'MAE Mean':<8}")
        print("-" * 55)
        for target in self.target_names:
            results = cv_results[target]
            print(f"{target:<15} {results['r2_mean']:.4f}   {results['r2_std']:.4f}   "
                  f"{results['rmse_mean']:.4f}     {results['mae_mean']:.4f}")

        # Overall metrics
        print(f"\nOverall Average Metrics:")
        print(f"  Average Classification Accuracy: {cv_results['overall']['accuracy_mean']:.4f}")
        print(f"  Average F1 Score: {cv_results['overall']['f1_mean']:.4f}")
        print(f"  Average R² Score: {cv_results['overall']['r2_mean']:.4f}")
        print(f"  EasyEnsemble Base Learners Count: {cv_results['overall']['easy_ensemble_n_estimators']}")
        print(f"  Total Cross-validation Time: {cv_results['overall']['cv_time']:.2f} seconds")
        print(f"  Parallel Thread Count: {self.n_jobs}")

    def main_easy_ensemble_lightgbm_optimization():
        """
        Main function - Two-Stage EasyEnsemble-LightGBM Model + Optimized Bayesian Search
        """
        # Load data
        try:
            df = pd.read_csv('PA_1km_drop_AF_new.csv')
            print(f"Data loaded successfully, shape: {df.shape}")
        except FileNotFoundError:
            print("Please replace the CSV file path with the correct one")
            return

        print("=" * 80)
        print("Two-Stage EasyEnsemble-LightGBM Regression Model + Optimized Bayesian Search")
        print("Stage 1: EasyEnsemble Classifier predicts zero-value probabilities (LightGBM base learners)")
        print("Stage 2: LightGBM Regressor predicts specific values")
        print("Final prediction: Probability-weighted combination")
        print("Optimization strategy: Only optimize key parameters, others use default values")
        print("=" * 80)

        # Create model instance
        model = TwoStageEasyEnsembleLightGBM(
            easy_ensemble_n_estimators=10,
            cv=5,
            random_state=42,
            verbose=True,
            constraint_column='pa_total_area',
            use_gpu=True,
            zero_threshold=1e-6,
            n_jobs=-1  # Use all CPU cores
        )

        # Prepare data
        X, y_continuous, y_binary = model.prepare_data(df)
        print(f"Data preparation completed, feature count: {X.shape[1]}, sample count: {X.shape[0]}")

        # Display data distribution
        for i, target in enumerate(model.target_names):
            zero_count = np.sum(y_binary[:, i] == 0)
            positive_count = np.sum(y_binary[:, i] == 1)
            print(f"{target}: Zero samples {zero_count}, Positive samples {positive_count} "
                  f"(Positive ratio: {positive_count / (zero_count + positive_count) * 100:.1f}%)")

        # Bayesian optimization of EasyEnsemble classifier (streamlined)
        print("\n=== EasyEnsemble Classifier Optimized Bayesian Search (F1 score optimization) ===")
        print("Optimizing parameters: easy_ensemble_n_estimators, n_estimators, learning_rate, num_leaves")
        classifier_optimization = model.bayesian_optimize_classifier(
            X, y_binary, n_calls=30, cv_folds=5  # Reduced iteration count
        )

        # Bayesian optimization of regressor (streamlined)
        print("\n=== Regressor Optimized Bayesian Search ===")
        print("Optimizing parameters: n_estimators, learning_rate, num_leaves, subsample")
        regressor_optimization = model.bayesian_optimize_regressor(
            X, y_continuous, y_binary, n_calls=30, cv_folds=5  # Reduced iteration count
        )

        # Cross-validation using optimized parameters
        print("\n=== Cross-validation with Optimized Parameters ===")
        cv_results = model.cross_validate(X, y_continuous, y_binary, cv_folds=5)

        # Train final model on full data
        print("\n=== Training Final Model ===")
        model.fit(X, y_continuous, y_binary)

        # Predictions
        predictions = model.predict(X)
        detailed_predictions = model.predict_detailed(X)

        # Print comprehensive results
        model.print_comprehensive_results(cv_results)

        # Save the model
        model.save_model("final_AF_two_stage_easy_ensemble_lightgbm.pkl")

        print(f"\n" + "=" * 80)
        print("Two-Stage EasyEnsemble-LightGBM Model Summary")
        print("=" * 80)
        print(f"Best Classifier F1 Score: {classifier_optimization['best_f1_score']:.4f}")
        print(f"Best Regressor R²: {regressor_optimization['best_r2_score']:.4f}")
        print(f"Final Average F1 Score: {cv_results['overall']['f1_mean']:.4f}")
        print(f"Final Average R²: {cv_results['overall']['r2_mean']:.4f}")
        print(f"Best EasyEnsemble Base Learners: {classifier_optimization['best_easy_ensemble_n_estimators']}")
        print(f"Classifier Optimization Time: {classifier_optimization['optimization_time']:.2f} seconds")
        print(f"Regressor Optimization Time: {regressor_optimization['optimization_time']:.2f} seconds")

        # Print best parameters
        print(f"\nBest EasyEnsemble Classifier Parameters:")
        print(f"  Base learners count: {classifier_optimization['best_easy_ensemble_n_estimators']}")
        for param, value in classifier_optimization['best_params'].items():
            if isinstance(value, float):
                print(f"  {param}: {value:.4f}")
            else:
                print(f"  {param}: {value}")

        print(f"\nBest Regressor Parameters:")
        for param, value in regressor_optimization['best_params'].items():
            if isinstance(value, float):
                print(f"  {param}: {value:.4f}")
            else:
                print(f"  {param}: {value}")

        # Display unoptimized default parameters
        print(f"\nUnoptimized Classifier Parameters (using defaults):")
        unoptimized_classifier = ['subsample', 'feature_fraction', 'lambda_l1', 'lambda_l2', 'boosting_type']
        for param in unoptimized_classifier:
            if param in model.classifier_params:
                print(f"  {param}: {model.classifier_params[param]}")

        print(f"\nUnoptimized Regressor Parameters (using defaults):")
        unoptimized_regressor = ['feature_fraction', 'lambda_l1', 'lambda_l2', 'min_child_weight', 'boosting_type']
        for param in unoptimized_regressor:
            if param in model.regressor_params:
                print(f"  {param}: {model.regressor_params[param]}")

        return model, classifier_optimization, regressor_optimization, cv_results

    def analyze_lightgbm_easy_ensemble_predictions(model, X, y_continuous, sample_size=100):
        """
        Analyze LightGBM EasyEnsemble predictions, with a focus on improvements in handling imbalanced data
        """
        print("\n" + "=" * 80)
        print("LightGBM EasyEnsemble Prediction Analysis")
        print("=" * 80)

        # Random sampling analysis
        if len(X) > sample_size:
            indices = np.random.choice(len(X), sample_size, replace=False)
            X_sample = X[indices]
            y_sample = y_continuous[indices]
        else:
            X_sample = X
            y_sample = y_continuous

        # Get detailed predictions
        detailed_pred = model.predict_detailed(X_sample)
        probabilities = detailed_pred['probabilities']
        regression_pred = detailed_pred['regression_pred']
        final_pred = detailed_pred['final_pred']

        print(f"Analysis sample size: {len(X_sample)}")
        print(f"EasyEnsemble Base Learners Count: {model.easy_ensemble_n_estimators}")
        print(f"Parallel Thread Count: {model.n_jobs}")

        for i, target in enumerate(model.target_names):
            print(f"\n{target} Analysis:")

            # Find historical zero-value samples
            zero_mask = y_sample[:, i] < 1e-6
            zero_count = np.sum(zero_mask)

            if zero_count > 0:
                # Prediction situation for zero-value samples
                zero_probs = probabilities[zero_mask, i]
                zero_reg_pred = regression_pred[zero_mask, i]
                zero_final_pred = final_pred[zero_mask, i]

                print(f"  Historical zero-value samples: {zero_count}")
                print(f"  Zero-value sample average probability: {np.mean(zero_probs):.4f} ± {np.std(zero_probs):.4f}")
                print(f"  Zero-value sample probability median: {np.median(zero_probs):.4f}")
                print(f"  Zero-value sample regression prediction mean: {np.mean(zero_reg_pred):.1f}")
                print(f"  Zero-value sample final prediction mean: {np.mean(zero_final_pred):.1f}")
                print(f"  Zero-value sample maximum final prediction: {np.max(zero_final_pred):.1f}")

                # Check for extreme jumps
                extreme_jumps = zero_final_pred > 50000  # More than 50,000 square meters
                extreme_count = np.sum(extreme_jumps)
                print(
                    f"  Extreme jumps (>50000) sample count: {extreme_count} ({extreme_count / zero_count * 100:.1f}%)")

                # Analyze probability distribution
                low_prob_count = np.sum(zero_probs < 0.1)
                mid_prob_count = np.sum((zero_probs >= 0.1) & (zero_probs < 0.5))
                high_prob_count = np.sum(zero_probs >= 0.5)
                print(
                    f"  Probability distribution: Low(<0.1): {low_prob_count}, Medium(0.1-0.5): {mid_prob_count}, High(>=0.5): {high_prob_count}")

            # Prediction situation for positive samples
            positive_mask = y_sample[:, i] > 1e-6
            positive_count = np.sum(positive_mask)

            if positive_count > 0:
                pos_probs = probabilities[positive_mask, i]
                pos_final_pred = final_pred[positive_mask, i]
                pos_true = y_sample[positive_mask, i]

                print(f"  Historical positive samples: {positive_count}")
                print(f"  Positive sample average probability: {np.mean(pos_probs):.4f} ± {np.std(pos_probs):.4f}")
                print(f"  Positive sample probability median: {np.median(pos_probs):.4f}")
                print(f"  Positive sample R²: {r2_score(pos_true, pos_final_pred):.4f}")

def main_easy_ensemble_lightgbm_optimization():
    """
    Main function - Two-Stage EasyEnsemble-LightGBM Model + Optimized Bayesian Search
    """
    # Load data
    try:
        df = pd.read_csv('PA_1km_drop_AF_new.csv')
        print(f"Data loaded successfully, shape: {df.shape}")
    except FileNotFoundError:
        print("Please replace the CSV file path with the correct one")
        return

    print("=" * 80)
    print("Two-Stage EasyEnsemble-LightGBM Regression Model + Optimized Bayesian Search")
    print("Stage 1: EasyEnsemble Classifier predicts zero-value probabilities (LightGBM base learners)")
    print("Stage 2: LightGBM Regressor predicts specific values")
    print("Final prediction: Probability-weighted combination")
    print("Optimization strategy: Only optimize key parameters, others use default values")
    print("=" * 80)

    # Create model instance
    model = TwoStageEasyEnsembleLightGBM(
        easy_ensemble_n_estimators=10,
        cv=5,
        random_state=42,
        verbose=True,
        constraint_column='pa_total_area',
        use_gpu=True,
        zero_threshold=1e-6,
        n_jobs=-1  # Use all CPU cores
    )

    # Prepare data
    X, y_continuous, y_binary = model.prepare_data(df)
    print(f"Data preparation completed, feature count: {X.shape[1]}, sample count: {X.shape[0]}")

    # Display data distribution
    for i, target in enumerate(model.target_names):
        zero_count = np.sum(y_binary[:, i] == 0)
        positive_count = np.sum(y_binary[:, i] == 1)
        print(f"{target}: Zero samples {zero_count}, Positive samples {positive_count} "
              f"(Positive ratio: {positive_count / (zero_count + positive_count) * 100:.1f}%)")

    # Bayesian optimization of EasyEnsemble classifier (streamlined)
    print("\n=== EasyEnsemble Classifier Optimized Bayesian Search (F1 score optimization) ===")
    print("Optimizing parameters: easy_ensemble_n_estimators, n_estimators, learning_rate, num_leaves")
    classifier_optimization = model.bayesian_optimize_classifier(
        X, y_binary, n_calls=30, cv_folds=5  # Reduced iteration count
    )

    # Bayesian optimization of regressor (streamlined)
    print("\n=== Regressor Optimized Bayesian Search ===")
    print("Optimizing parameters: n_estimators, learning_rate, num_leaves, subsample")
    regressor_optimization = model.bayesian_optimize_regressor(
        X, y_continuous, y_binary, n_calls=30, cv_folds=5  # Reduced iteration count
    )

    # Cross-validation using optimized parameters
    print("\n=== Cross-validation with Optimized Parameters ===")
    cv_results = model.cross_validate(X, y_continuous, y_binary, cv_folds=5)

    # Train final model on full data
    print("\n=== Training Final Model ===")
    model.fit(X, y_continuous, y_binary)

    # Predictions
    predictions = model.predict(X)
    detailed_predictions = model.predict_detailed(X)

    # Print comprehensive results
    model.print_comprehensive_results(cv_results)

    # Save the model
    model.save_model("final_AF_two_stage_easy_ensemble_lightgbm.pkl")

    print(f"\n" + "=" * 80)
    print("Two-Stage EasyEnsemble-LightGBM Model Summary")
    print("=" * 80)
    print(f"Best Classifier F1 Score: {classifier_optimization['best_f1_score']:.4f}")
    print(f"Best Regressor R²: {regressor_optimization['best_r2_score']:.4f}")
    print(f"Final Average F1 Score: {cv_results['overall']['f1_mean']:.4f}")
    print(f"Final Average R²: {cv_results['overall']['r2_mean']:.4f}")
    print(f"Best EasyEnsemble Base Learners: {classifier_optimization['best_easy_ensemble_n_estimators']}")
    print(f"Classifier Optimization Time: {classifier_optimization['optimization_time']:.2f} seconds")
    print(f"Regressor Optimization Time: {regressor_optimization['optimization_time']:.2f} seconds")

    # Print best parameters
    print(f"\nBest EasyEnsemble Classifier Parameters:")
    print(f"  Base learners count: {classifier_optimization['best_easy_ensemble_n_estimators']}")
    for param, value in classifier_optimization['best_params'].items():
        if isinstance(value, float):
            print(f"  {param}: {value:.4f}")
        else:
            print(f"  {param}: {value}")

    print(f"\nBest Regressor Parameters:")
    for param, value in regressor_optimization['best_params'].items():
        if isinstance(value, float):
            print(f"  {param}: {value:.4f}")
        else:
            print(f"  {param}: {value}")

    # Display unoptimized default parameters
    print(f"\nUnoptimized Classifier Parameters (using defaults):")
    unoptimized_classifier = ['subsample', 'feature_fraction', 'lambda_l1', 'lambda_l2', 'boosting_type']
    for param in unoptimized_classifier:
        if param in model.classifier_params:
            print(f"  {param}: {model.classifier_params[param]}")

    print(f"\nUnoptimized Regressor Parameters (using defaults):")
    unoptimized_regressor = ['feature_fraction', 'lambda_l1', 'lambda_l2', 'min_child_weight', 'boosting_type']
    for param in unoptimized_regressor:
        if param in model.regressor_params:
            print(f"  {param}: {model.regressor_params[param]}")

    return model, classifier_optimization, regressor_optimization, cv_results

if __name__ == "__main__":
    print("=" * 80)
    print("Two-Stage EasyEnsemble-LightGBM Multi-output Regression Model")
    print("Configuration: LightGBM + EasyEnsemble for handling imbalanced data + Optimized Bayesian Search")
    print("Optimization strategy: Only optimize key parameters to reduce computational overhead")
    print("Required dependencies:")
    print("pip install scikit-optimize")
    print("pip install lightgbm")
    print("pip install imbalanced-learn")
    print("pip install matplotlib")
    print("=" * 80)

    try:
        # Run the main function
        model, classifier_opt, regressor_opt, cv_results = main_easy_ensemble_lightgbm_optimization()

        # Optional: Run prediction analysis
        print("\nDo you want to run LightGBM EasyEnsemble prediction result analysis? (y/n, default y): ", end="")
        analysis_choice = input().strip().lower()

        if analysis_choice != 'n':
            # Reload data for analysis
            df = pd.read_csv('PA_1km_drop_AF_new.csv')
            X, y_continuous, y_binary = model.prepare_data(df)
            analyze_lightgbm_easy_ensemble_predictions(model, X, y_continuous, sample_size=200)

    except KeyboardInterrupt:
        print("\nUser interrupted execution")
    except Exception as e:
        print(f"Execution error: {e}")
        print("Please check if dependencies are correctly installed")


