import os
import sys
import tarfile
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import config
from src.data_preprocessor import DataPreprocessor
from src.model_pipeline import ModelPipeline

sys.path.insert(0, "src")


df = pd.read_csv("data_D.csv") 
for col in config.ordinal_cols + config.cat_cols:
    if col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            print(f"PERINGATAN: Kolom '{col}' adalah NUMERIK tapi ada di list KATEGORIKAL!")

artifact_dir = "model_artifact"
model_filename = "model.joblib"
tarball_path = os.path.join(artifact_dir, "model.tar.gz")

def main() -> None:
    os.makedirs(artifact_dir, exist_ok=True)

    print("Loading dataset...")
    df = pd.read_csv("data_D.csv")
    print(f"Dataset shape: {df.shape}")

    print("Cleaning and feature engineering via OOP modules...")
    preprocessor = DataPreprocessor()
    df_cleaned = preprocessor.clean_data(df)
    df_featured = preprocessor.feature_engineering(df_cleaned)

    y = df_featured['Credit_Score'].map(config.target_mapping)
    X = df_featured.drop(columns=['Credit_Score'])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train size: {X_train.shape}, Test size: {X_test.shape}")

    print("Starting Hyperparameter Tuning with Optuna...")
    pipeline_manager = ModelPipeline()
    best_params, best_score = pipeline_manager.tune_parameters(X_train, y_train, n_trials=60, patience=5)
    print(f"Best F1-Weighted Score: {best_score}")

    print("Training final model pipeline...")
    final_pipeline = pipeline_manager.build_and_fit_final_pipeline(X_train, y_train, best_params)

    print("Evaluating model on test set...")
    y_pred = final_pipeline.predict(X_test)
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred, target_names=config.target_names))

    model_path = os.path.join(artifact_dir, model_filename)
    joblib.dump(final_pipeline, model_path)
    print(f"Saved model binary to: {model_path}")

    print("Packaging model binary AND src folder into tar.gz...")
    with tarfile.open(tarball_path, "w:gz") as tar:
        tar.add(model_path, arcname=model_filename)
        tar.add("src", arcname="code")
    print(f"Packaged successfully: {tarball_path}")

    print("\nNext steps:")
    print("  1. Upload model.tar.gz to S3 bucket")
    print("  2. Run deploy_endpoint.py")

if __name__ == "__main__":
    main()