import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import mlflow

import config
from data_preprocessor import DataPreprocessor
from model_pipeline import ModelPipeline

def main():
    model_pkl_path = "notebook/best_model_pipeline.pkl" 
    
    pipeline_manager = ModelPipeline(experiment_name="Churn_Prediction_Comparison")
    preprocessor = DataPreprocessor()
    
    if os.path.exists(model_pkl_path):
        print(f"Menemukan model pre-trained '{model_pkl_path}'. Langsung memuat model...")
        final_pipeline = pipeline_manager.load_model(model_pkl_path)
        
        print("Loading data baru untuk prediksi...")
        df = pd.read_csv("data_D.csv")
        
        df_cleaned = preprocessor.clean_data(df)
        df_featured = preprocessor.feature_engineering(df_cleaned)
        
        y_test = df_featured['Credit_Score'].map(config.target_mapping)
        X_test = df_featured.drop(columns=['Credit_Score'])
        
        print("Melakukan evaluasi prediksi pada data baru...")
        mlflow_run_context = mlflow.start_run(run_name="Evaluation_PreTrained_Model")
        
    else:
        print("File .pkl tidak ditemukan. Menjalankan proses training dari awal...")
        df = pd.read_csv("data_D.csv")
        
        df_cleaned = preprocessor.clean_data(df)
        df_featured = preprocessor.feature_engineering(df_cleaned)
        
        y = df_featured['Credit_Score'].map(config.target_mapping)
        X = df_featured.drop(columns=['Credit_Score'])
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print("Menjalankan pencarian parameter Optuna (XGBoost vs RF vs LightGBM)...")
        best_params, _ = pipeline_manager.tune_parameters(X_train, y_train, n_trials=50, patience=15)
        
        print("Training final model menggunakan parameter terbaik...")
        final_pipeline = pipeline_manager.build_and_fit_final_pipeline(X_train, y_train, best_params)
        
        os.makedirs(os.path.dirname(model_pkl_path), exist_ok=True)
        pipeline_manager.save_model(model_pkl_path)
        
        print("Melakukan evaluasi prediksi...")
        mlflow_run_context = mlflow.start_run(run_name="Evaluation_Test_Set", nested=True)

    with mlflow_run_context:
        y_pred = final_pipeline.predict(X_test)
        
        report_dict = classification_report(y_test, y_pred, target_names=config.target_names, output_dict=True)
        print("\nClassification Report:\n")
        print(classification_report(y_test, y_pred, target_names=config.target_names))
        
        mlflow.log_metric("test_accuracy", report_dict["accuracy"])
        mlflow.log_metric("test_f1_weighted", report_dict["weighted avg"]["f1-score"])
        mlflow.log_metric("test_precision_weighted", report_dict["weighted avg"]["precision"])
        mlflow.log_metric("test_recall_weighted", report_dict["weighted avg"]["recall"])
        
        mlflow.log_metric("test_f1_macro", report_dict["macro avg"]["f1-score"])
        mlflow.log_metric("test_precision_macro", report_dict["macro avg"]["precision"])
        mlflow.log_metric("test_recall_macro", report_dict["macro avg"]["recall"])
        
        for label in config.target_names:
            clean_label = label.lower().replace(" ", "_")
            mlflow.log_metric(f"class_{clean_label}_f1", report_dict[label]["f1-score"])
            mlflow.log_metric(f"class_{clean_label}_precision", report_dict[label]["precision"])
            mlflow.log_metric(f"class_{clean_label}_recall", report_dict[label]["recall"])
        
        plt.figure(figsize=(6, 4))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=config.target_names, yticklabels=config.target_names)
        plt.title('Confusion Matrix - Memakai Model Pipeline')
        plt.xlabel('Prediksi Model')
        plt.ylabel('Kenyataan (Aktual)')
        plt.tight_layout()
        
        # Simpan plot secara lokal sementara dan upload ke MLflow Artifacts
        plot_path = "confusion_matrix.png"
        plt.savefig(plot_path)
        mlflow.log_artifact(plot_path)
        
        # Hapus file gambar lokal setelah di-upload agar bersih
        if os.path.exists(plot_path):
            os.remove(plot_path)
            
        plt.show()

if __name__ == "__main__":
    main()