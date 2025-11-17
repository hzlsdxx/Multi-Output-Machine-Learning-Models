import numpy as np
import pandas as pd
import time, pickle, warnings
from typing import Tuple, Dict, Any, List
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, accuracy_score, f1_score, precision_score, recall_score
from imblearn.ensemble import EasyEnsembleClassifier
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args
from catboost import CatBoostRegressor, CatBoostClassifier

warnings.filterwarnings('ignore')

class TwoStageEasyEnsembleCatBoost:

    
    def __init__(self, classifier_params=None, regressor_params=None,
                 easy_ensemble_n_estimators=10, cv=5, random_state=42, verbose=True,
                 constraint_column=None, use_gpu=True, zero_threshold=1e-6):
        self.cv = cv
        self.random_state = random_state
        self.verbose = verbose
        self.constraint_column = constraint_column
        self.zero_threshold = zero_threshold
        self.easy_ensemble_n_estimators = easy_ensemble_n_estimators

        self.use_gpu = use_gpu
        if self.use_gpu and self.verbose:
            print("GPU acceleration for CatBoost is enabled")

        default_clf = dict(
            iterations=100, 
            learning_rate=0.1, 
            depth=6,
            subsample=0.8, 
            random_seed=random_state, 
            verbose=False,
            bootstrap_type='Poisson',
            loss_function='Logloss'
        )

        default_reg = dict(
            iterations=600, 
            learning_rate=0.07, 
            depth=10,
            subsample=0.88, 
            random_seed=random_state, 
            verbose=False,
            bootstrap_type='Poisson',
            loss_function='RMSE'
        )

        if self.use_gpu:
            default_clf.update(dict(task_type='GPU'))
            default_reg.update(dict(task_type='GPU'))

        self.classifier_params = {**default_clf, **(classifier_params or {})}
        self.regressor_params = {**default_reg, **(regressor_params or {})}

        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.classifier = None
        self.regressor = None
        self.feature_names = None
        self.target_names = ['cropland_area', 'built_up_area', 'pas_area']
        
        self.best_classifier_params = None
        self.best_regressor_params = None
        self.optimization_history = []

    def _preprocess_data(self, df: pd.DataFrame, fit=True):
        df = df.copy()
        to_drop = []
        if 'value' in df.columns: 
            to_drop.append('value')
        if 'pa_total_area' in df.columns and self.constraint_column != 'pa_total_area':
            to_drop.append('pa_total_area')
        if to_drop: 
            df.drop(columns=to_drop, inplace=True)

        cats = ['desig_eng', 'desig_type', 'iucn_cat', 'gov_type', 'own_type', 'iso3']
        for c in cats:
            if c in df.columns:
                if fit:
                    le = LabelEncoder()
                    df[c] = le.fit_transform(df[c].astype(str))
                    self.label_encoders[c] = le
                else:
                    df[c] = df[c].astype(str).map(
                        lambda x: x if x in self.label_encoders[c].classes_ else self.label_encoders[c].classes_[0])
                    df[c] = self.label_encoders[c].transform(df[c])
        
        nums = df.select_dtypes(include=[np.number]).columns
        df[nums] = df[nums].fillna(df[nums].median())
        return df

    def prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        dfp = self._preprocess_data(df, fit=True)
        X = dfp.drop(columns=self.target_names + ([self.constraint_column] if self.constraint_column else []))
        y_continuous = dfp[self.target_names].values
        
        y_binary = (y_continuous > self.zero_threshold).astype(int)
        
        self.feature_names = X.columns.tolist()
        return X.values, y_continuous, y_binary

    def create_bayesian_search_space_classifier(self):
        return [
            Integer(5, 20, name='easy_ensemble_n_estimators'),
            Integer(50, 200, name='iterations'),
            Real(0.05, 0.2, name='learning_rate'),
            Integer(3, 8, name='depth'),
            Real(0.7, 1.0, name='subsample')
        ]

    def create_bayesian_search_space_regressor(self):
        return [
            Integer(400, 1000, name='iterations'),
            Real(0.01, 0.2, name='learning_rate'),
            Integer(6, 15, name='depth'),
            Real(0.7, 1.0, name='subsample')
        ]

    def bayesian_optimize_classifier(self, X, y_binary, n_calls=30, cv_folds=5):
        if self.verbose:
            print("Starting Bayesian optimization for EasyEnsemble classifier...")
            print(f"Number of optimization iterations: {n_calls}, Cross-validation folds: {cv_folds}")
            print("Optimization goal: F1 score")

        space = self.create_bayesian_search_space_classifier()
        classifier_history = []
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)

        iteration_counter = {'count': 0}

        @use_named_args(space)
        def objective(**params):
            iteration_counter['count'] += 1
            if self.verbose:
                print(f"Iteration {iteration_counter['count']} of EasyEnsemble classifier...")

            easy_ensemble_n_est = params.pop('easy_ensemble_n_estimators')
            
            current_params = self.classifier_params.copy()
            current_params.update(params)
            
            f1_scores = []
            for train_idx, val_idx in kf.split(X):
                X_train_fold, X_val_fold = X[train_idx], X[val_idx]
                y_train_fold, y_val_fold = y_binary[train_idx], y_binary[val_idx]
                
                fold_f1_scores = []
                for i in range(y_binary.shape[1]):
                    try:
                        catboost_clf = CatBoostClassifier(**current_params)
                        
                        easy_ensemble = EasyEnsembleClassifier(
                            estimator=catboost_clf,
                            n_estimators=easy_ensemble_n_est,
                            random_state=self.random_state,
                            n_jobs=1
                        )
                        
                        easy_ensemble.fit(X_train_fold, y_train_fold[:, i])
                        y_pred_fold = easy_ensemble.predict(X_val_fold)
                        
                        f1 = f1_score(y_val_fold[:, i], y_pred_fold, zero_division=0)
                        fold_f1_scores.append(f1)
                    except Exception as e:
                        if self.verbose:
                            print(f"Classifier training failed: {e}")
                        return 1.0
                
                f1_scores.append(np.mean(fold_f1_scores))
            
            mean_f1 = np.mean(f1_scores)
            
            classifier_history.append({
                'params': {**params, 'easy_ensemble_n_estimators': easy_ensemble_n_est},
                'f1_score': mean_f1,
                'iteration': iteration_counter['count']
            })
            
            if self.verbose:
                print(f"   Current classifier F1 = {mean_f1:.4f}")

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
            print(f"\nEasyEnsemble classifier Bayesian optimization complete!")
            print(f"Optimization time: {optimization_time:.2f} seconds")
            print(f"Best F1 score: {-result.fun:.4f}")
            print(f"Best EasyEnsemble base estimator count: {self.easy_ensemble_n_estimators}")
            print(f"Best CatBoost parameters:")
            for param, value in self.best_classifier_params.items():
                if isinstance(value, float):
                    print(f"  {param}: {value:.4f}")
                else:
                    print(f"  {param}: {value}")

        return optimization_result

    def bayesian_optimize_regressor(self, X, y_continuous, y_binary, n_calls=50, cv_folds=5):
        if self.verbose:
            print("Starting Bayesian parameter optimization for the regressor...")
            print(f"Number of optimization iterations: {n_calls}, Number of cross-validation folds: {cv_folds}")

        space = self.create_bayesian_search_space_regressor()
        regressor_history = []
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)

        iteration_counter = {'count': 0}

        @use_named_args(space)
        def objective(**params):
            iteration_counter['count'] += 1
            if self.verbose:
                print(f"Iteration {iteration_counter['count']} of the regressor...")

            current_params = self.regressor_params.copy()
            current_params.update(params)

            positive_mask = np.any(y_binary > 0, axis=1)
            X_positive = X[positive_mask]
            y_positive = y_continuous[positive_mask]

            if len(X_positive) == 0:
                return 1.0

            r2_scores = []
            kf_positive = KFold(n_splits=min(cv_folds, len(X_positive)),
                               shuffle=True, random_state=self.random_state)

            for train_idx, val_idx in kf_positive.split(X_positive):
                X_train_fold = X_positive[train_idx]
                X_val_fold = X_positive[val_idx]
                y_train_fold = y_positive[train_idx]
                y_val_fold = y_positive[val_idx]

                try:
                    regressor = MultiOutputRegressor(
                        CatBoostRegressor(**current_params),
                        n_jobs=1
                    )

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
                except Exception as e:
                    if self.verbose:
                        print(f"Regressor training failed: {e}")
                    return 1.0

            mean_r2 = np.mean(r2_scores) if r2_scores else -1

            regressor_history.append({
                'params': params.copy(),
                'r2_score': mean_r2,
                'iteration': iteration_counter['count']
            })

            if self.verbose:
                print(f"   Current regression R² = {mean_r2:.4f}")

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
            print(f"\nBayesian optimization for the regressor is complete!")
            print(f"Optimization time: {optimization_time:.2f} seconds")
            print(f"Best R² score: {-result.fun:.4f}")
            print(f"Best parameters:")
            for param, value in self.best_regressor_params.items():
                if isinstance(value, float):
                    print(f"  {param}: {value:.4f}")
                else:
                    print(f"  {param}: {value}")

        return optimization_result

    def fit(self, X, y_continuous, y_binary):
        if self.verbose:
            gpu_status = "GPU" if self.use_gpu else "CPU"
            print(f"Starting to train the two-stage EasyEnsemble model... (Using {gpu_status})")
            print(f"Number of base learners in EasyEnsemble: {self.easy_ensemble_n_estimators}")

        start_time = time.time()

        if self.verbose:
            print("Training EasyEnsemble classifier...")

        self.classifier = {}
        for i, target in enumerate(self.target_names):
            catboost_clf = CatBoostClassifier(**self.classifier_params)
            
            easy_ensemble = EasyEnsembleClassifier(
                estimator=catboost_clf,
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
            CatBoostRegressor(**self.regressor_params),
            n_jobs=1
        )
        self.regressor.fit(X_positive, y_positive)
        
        training_time = time.time() - start_time
        if self.verbose:
            print(f"EasyEnsemble model training complete! Training time: {training_time:.2f} seconds")

        return self

    def predict(self, X):
        if self.classifier is None or self.regressor is None:
            raise ValueError("The model has not been trained yet, please call the fit method first.")

        probabilities = np.zeros((X.shape[0], len(self.target_names)))
        for i, target in enumerate(self.target_names):
            prob = self.classifier[target].predict_proba(X)
            probabilities[:, i] = prob[:, 1]

        reg_pred = self.regressor.predict(X)
        reg_pred = np.maximum(reg_pred, 0.0)
        
        final_pred = probabilities * reg_pred
        
        return final_pred

    def predict_detailed(self, X):
        if self.classifier is None or self.regressor is None:
            raise ValueError("The model has not been trained yet. Please call the fit method first.")

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

    def cross_validate(self, X, y_continuous, y_binary, cv_folds=5):
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        results = {target: {'r2': [], 'mse': [], 'mae': [], 'rmse': []}
                  for target in self.target_names}
        
        classifier_results = {target: {'accuracy': [], 'precision': [], 'recall': [], 'f1': []} 
                            for target in self.target_names}

        if self.verbose:
            gpu_status = "GPU" if self.use_gpu else "CPU"
            print(f"Starting {cv_folds}-fold cross-validation... (Using {gpu_status})")
            print(f"EasyEnsemble base learner count: {self.easy_ensemble_n_estimators}")

        start_time = time.time()
        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            if self.verbose:
                print(f"Fold {fold + 1}/{cv_folds}...")

            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_cont_train, y_cont_val = y_continuous[train_idx], y_continuous[val_idx]
            y_bin_train, y_bin_val = y_binary[train_idx], y_binary[val_idx]

            classifier_fold = {}
            for i, target in enumerate(self.target_names):
                catboost_clf = CatBoostClassifier(**self.classifier_params)
                
                easy_ensemble = EasyEnsembleClassifier(
                    estimator=catboost_clf,
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
                    CatBoostRegressor(**self.regressor_params),
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
            print(f"Cross-validation complete! Total time: {cv_time:.2f} seconds")

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

    def get_feature_importance(self):
        if self.classifier is None or self.regressor is None:
            raise ValueError("The model has not been trained yet.")

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
            'easy_ensemble_n_estimators': self.easy_ensemble_n_estimators
        }

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

        if self.verbose:
            print(f"The two-stage EasyEnsemble model has been saved to: {filepath}")

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

        if self.verbose:
            print(f"The two-stage EasyEnsemble model has been loaded from {filepath}")

    def print_comprehensive_results(self, cv_results: dict):
        """Print the comprehensive results"""
        print("\n" + "=" * 80)
        print("Two-stage EasyEnsemble-CatBoost Multi-output Regression Model Results")
        print("=" * 80)

        print(f"\nEasyEnsemble Classifier Performance (Number of Base Learners: {self.easy_ensemble_n_estimators}):")
        print(f"{'Target Variable':<15} {'Accuracy':<8} {'Precision':<8} {'Recall':<8} {'F1 Score':<8}")
        print("-" * 60)
        for target in self.target_names:
            results = cv_results[target]
            print(f"{target:<15} {results['accuracy_mean']:.4f}   {results['precision_mean']:.4f}   "
                  f"{results['recall_mean']:.4f}   {results['f1_mean']:.4f}")

        print(f"\nRegressor Performance:")
        print(f"{'Target Variable':<15} {'R² Mean':<8} {'R² Std Dev':<8} {'RMSE Mean':<10} {'MAE Mean':<8}")
        print("-" * 55)
        for target in self.target_names:
            results = cv_results[target]
            print(f"{target:<15} {results['r2_mean']:.4f}   {results['r2_std']:.4f}   "
                  f"{results['rmse_mean']:.4f}     {results['mae_mean']:.4f}")

        print(f"\nOverall Average Metrics:")
        print(f"  Average Classification Accuracy: {cv_results['overall']['accuracy_mean']:.4f}")
        print(f"  Average F1 Score: {cv_results['overall']['f1_mean']:.4f}")
        print(f"  Average R² Score: {cv_results['overall']['r2_mean']:.4f}")
        print(f"  Number of EasyEnsemble Base Learners: {cv_results['overall']['easy_ensemble_n_estimators']}")
        print(f"  Cross-validation Total Time: {cv_results['overall']['cv_time']:.2f} seconds")


def main_two_stage_easy_ensemble_catboost_optimization():

    print("Loading data and initializing the two-stage EasyEnsemble-CatBoost model...")

    try:
        df = pd.read_csv('PA_1km_drop_EU_new.csv')
        print(f"Data loaded successfully, shape: {df.shape}")
    except FileNotFoundError:
        print("Please replace the CSV file path with the correct one")
        return

    print("=" * 80)
    print("Two-stage EasyEnsemble-CatBoost Regression Model + Bayesian Optimization")
    print("Phase 1: EasyEnsemble Classifier predicts zero-value probabilities")
    print("Phase 2: Regressor predicts exact values")
    print("Final Prediction: Probability-weighted combination")
    print("=" * 80)

    model = TwoStageEasyEnsembleCatBoost(
        easy_ensemble_n_estimators=10,
        cv=5,
        random_state=42,
        verbose=True,
        constraint_column='pa_total_area',
        use_gpu=True,
        zero_threshold=1e-6
    )

    X, y_continuous, y_binary = model.prepare_data(df)
    print(f"Data preparation complete, features: {X.shape[1]}, samples: {X.shape[0]}")

    for i, target in enumerate(model.target_names):
        zero_count = np.sum(y_binary[:, i] == 0)
        positive_count = np.sum(y_binary[:, i] == 1)
        print(f"{target}: Zero samples {zero_count}, Positive samples {positive_count} "
              f"(Positive ratio: {positive_count / (zero_count + positive_count) * 100:.1f}%)")

    print("\n=== EasyEnsemble Classifier Bayesian Optimization (F1 Score optimization) ===")
    classifier_optimization = model.bayesian_optimize_classifier(
        X, y_binary, n_calls=30, cv_folds=5
    )

    print("\n=== Regressor Bayesian Optimization ===")
    regressor_optimization = model.bayesian_optimize_regressor(
        X, y_continuous, y_binary, n_calls=30, cv_folds=5
    )

    print("\n=== Cross-validation with optimized parameters ===")
    cv_results = model.cross_validate(X, y_continuous, y_binary, cv_folds=5)

    print("\n=== Training final model ===")
    model.fit(X, y_continuous, y_binary)

    predictions = model.predict(X)
    detailed_predictions = model.predict_detailed(X)

    model.print_comprehensive_results(cv_results)

    model.save_model("final_two_stage_EU_easy_ensemble_catboost.pkl")

    print(f"\n" + "=" * 80)
    print("Two-stage EasyEnsemble-CatBoost Model Summary")
    print("=" * 80)
    print(f"Classifier Best F1 Score: {classifier_optimization['best_f1_score']:.4f}")
    print(f"Regressor Best R²: {regressor_optimization['best_r2_score']:.4f}")
    print(f"Final Average F1 Score: {cv_results['overall']['f1_mean']:.4f}")
    print(f"Final Average R²: {cv_results['overall']['r2_mean']:.4f}")
    print(f"Best EasyEnsemble Base Learners: {classifier_optimization['best_easy_ensemble_n_estimators']}")
    print(f"Classifier Optimization Time: {classifier_optimization['optimization_time']:.2f} seconds")
    print(f"Regressor Optimization Time: {regressor_optimization['optimization_time']:.2f} seconds")

    print(f"\nEasyEnsemble Classifier Best Parameters:")
    print(f"  Base Learners: {classifier_optimization['best_easy_ensemble_n_estimators']}")
    for param, value in classifier_optimization['best_params'].items():
        if isinstance(value, float):
            print(f"  {param}: {value:.4f}")
        else:
            print(f"  {param}: {value}")

    print(f"\nRegressor Best Parameters:")
    for param, value in regressor_optimization['best_params'].items():
        if isinstance(value, float):
            print(f"  {param}: {value:.4f}")
        else:
            print(f"  {param}: {value}")

    print("\nModel saved to final_two_stage_EU_easy_ensemble_catboost.pkl")

    return model, classifier_optimization, regressor_optimization, cv_results


if __name__ == "__main__":
    print("=" * 80)
    print("Two-stage EasyEnsemble-CatBoost Model Dependency Libraries:")
    print("pip install catboost")
    print("pip install scikit-optimize")
    print("pip install imbalanced-learn")
    print("=" * 80)

    try:
        model, classifier_opt, regressor_opt, cv_results = main_two_stage_easy_ensemble_catboost_optimization()

    except KeyboardInterrupt:
        print("\nUser interrupted the execution")
    except Exception as e:
        print(f"An error occurred: {e}")
        print("Please ensure that the dependencies are correctly installed")
