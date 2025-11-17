import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings('ignore')


class SimpleEasyEnsemblePredictor:
    """
    Simplified EasyEnsemble Hard Classification Predictor
    Core functionality: Load model → Hard classification prediction → Output to CSV
    """

    def __init__(self, model_path: str):
        """Initialize and load the model"""
        self.load_model(model_path)

    def load_model(self, filepath: str):
        """Load the pre-trained model"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        self.classifier = model_data['classifier']
        self.regressor = model_data['regressor']
        self.label_encoders = model_data['label_encoders']
        self.feature_names = model_data['feature_names']
        self.target_names = model_data['target_names']
        print(f"Model loaded successfully: {len(self.feature_names)} features")

    def _preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Data preprocessing"""
        df_processed = df.copy()

        # Remove unnecessary columns
        columns_to_drop = []
        if 'value' in df_processed.columns:
            columns_to_drop.append('value')
        if 'pa_total_area' in df_processed.columns:
            columns_to_drop.append('pa_total_area')

        if columns_to_drop:
            df_processed = df_processed.drop(columns_to_drop, axis=1)
            print(f"Removed columns: {columns_to_drop}")

        # Process categorical variables
        categorical_cols = ['desig_eng', 'desig_type', 'iucn_cat', 'gov_type', 'own_type', 'iso3']

        for col in categorical_cols:
            if col in df_processed.columns and col in self.label_encoders:
                # Handle unknown categories
                unknown_mask = ~df_processed[col].astype(str).isin(self.label_encoders[col].classes_)
                if unknown_mask.any():
                    df_processed.loc[unknown_mask, col] = self.label_encoders[col].classes_[0]

                df_processed[col] = self.label_encoders[col].transform(df_processed[col].astype(str))

        # Fill missing values
        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
        df_processed[numeric_cols] = df_processed[numeric_cols].fillna(df_processed[numeric_cols].median())

        return df_processed

    def predict_csv(self, input_csv: str, output_csv: str, threshold: float = 0.5):
        """Predict from CSV and output results"""
        print(f"Reading data: {input_csv}")
        df = pd.read_csv(input_csv)
        print(f"Data shape: {df.shape}")

        # Preprocess
        df_processed = self._preprocess_data(df)
        X = df_processed[self.feature_names].values

        # Get classification probabilities
        probabilities = np.zeros((X.shape[0], len(self.target_names)))
        for i, target in enumerate(self.target_names):
            prob = self.classifier[target].predict_proba(X)
            probabilities[:, i] = prob[:, 1]

        # Hard classification (probability >= threshold is 1, otherwise 0)
        classifications = (probabilities >= threshold).astype(int)

        # Regression prediction
        reg_pred = self.regressor.predict(X)
        reg_pred = np.maximum(reg_pred, 0.0)

        # Final prediction = hard classification × regression values
        final_pred = classifications * reg_pred

        # Add predictions to the original data
        result_df = df.copy()
        result_df['predicted_cropland_area'] = final_pred[:, 0]
        result_df['predicted_built_up_area'] = final_pred[:, 1]
        result_df['predicted_pas_area'] = final_pred[:, 2]

        # Save results
        result_df.to_csv(output_csv, index=False)
        print(f"Prediction completed, results saved to: {output_csv}")

        # Simple statistics
        for i, target in enumerate(self.target_names):
            positive_count = np.sum(classifications[:, i] == 1)
            print(
                f"{target}: {positive_count}/{len(X)} samples predicted as positive ({positive_count / len(X) * 100:.1f}%)")


def main():
    """Main function"""

    # File path configuration
    MODEL_PATH = "final_AF_two_stage_easy_ensemble.pkl"
    INPUT_CSV = "PA_1km_drop_AF_SSP1.csv"
    OUTPUT_CSV = "PA_1km_drop_AF_SSP1_predictions.csv"

    try:
        # Create predictor and make predictions
        predictor = SimpleEasyEnsemblePredictor(MODEL_PATH)
        predictor.predict_csv(INPUT_CSV, OUTPUT_CSV, threshold=0.5)

    except Exception as e:
        print(f"Prediction failed: {e}")


if __name__ == "__main__":
    main()
