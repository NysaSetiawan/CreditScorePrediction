import optuna
import joblib
import mlflow
import mlflow.sklearn
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler, OrdinalEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import config

class EarlyStoppingCallback:
    def __init__(self, patience: int, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = None
        self.stagnant_trials = 0

    def __call__(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> None:
        if len(study.trials) == 0 or study.best_value is None:
            return
        current_score = study.best_value
        if self.best_score is None:
            self.best_score = current_score
        elif current_score > self.best_score + self.min_delta:
            self.best_score = current_score
            self.stagnant_trials = 0  
        else:
            self.stagnant_trials += 1  
        if self.stagnant_trials >= self.patience:
            print(f"\n[Early Stopping] Dihentikan otomatis pada Trial ke-{trial.number} karena tidak ada peningkatan setelah {self.patience} trials.")
            study.stop()

class ModelPipeline:
    def __init__(self, experiment_name: str = "Model_Comparison_Experiment"):
        mlflow.set_experiment(experiment_name)
        
        num_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler())
        ])
        
        ord_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("encoder", OrdinalEncoder(categories=[config.payment_order], handle_unknown="use_encoded_value", unknown_value=-1))
        ])
        
        cat_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ])

        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", num_transformer, config.num_cols),
                ("ord", ord_transformer, config.ordinal_cols),
                ("cat", cat_transformer, config.cat_cols)
            ]
        )
        self.final_pipeline = None

    def _objective(self, trial, X_train, y_train):
        with mlflow.start_run(run_name=f"trial_{trial.number}", nested=True):
            
            model_type = trial.suggest_categorical("model_type", ["xgboost", "random_forest", "lightgbm"])
            smote_k_neighbors = trial.suggest_int("smote_k_neighbors", 3, 7)
            smote = SMOTE(k_neighbors=smote_k_neighbors, random_state=42)

            # Log parameter dasar trial ke MLflow
            mlflow.log_param("model_type", model_type)
            mlflow.log_param("smote_k_neighbors", smote_k_neighbors)

            if model_type == "xgboost":
                params = {
                    "n_estimators": trial.suggest_int("xgb_n_estimators", 100, 500),
                    "max_depth": trial.suggest_int("xgb_max_depth", 3, 12),
                    "learning_rate": trial.suggest_float("xgb_learning_rate", 0.01, 0.3),
                    "subsample": trial.suggest_float("xgb_subsample", 0.5, 1.0),
                    "colsample_bytree": trial.suggest_float("xgb_colsample_bytree", 0.5, 1.0),
                    "gamma": trial.suggest_float("xgb_gamma", 0, 5),
                    "min_child_weight": trial.suggest_int("xgb_min_child_weight", 1, 10),
                    "objective": "multi:softprob",
                    "eval_metric": "mlogloss",
                    "random_state": 42,
                    "n_jobs": -1
                }
                model = XGBClassifier(**params)
                
            elif model_type == "random_forest":
                params = {
                    "n_estimators": trial.suggest_int("rf_n_estimators", 100, 500),
                    "max_depth": trial.suggest_int("rf_max_depth", 5, 20),
                    "min_samples_split": trial.suggest_int("rf_min_samples_split", 2, 10),
                    "min_samples_leaf": trial.suggest_int("rf_min_samples_leaf", 1, 10),
                    "criterion": trial.suggest_categorical("rf_criterion", ["gini", "entropy"]),
                    "max_features": trial.suggest_categorical("rf_max_features", ["sqrt", "log2", None]),
                    "class_weight": trial.suggest_categorical("rf_class_weight", [None, "balanced"]),
                    "random_state": 42,
                    "n_jobs": -1
                }
                model = RandomForestClassifier(**params)
                
            elif model_type == "lightgbm":
                params = {
                    "n_estimators": trial.suggest_int("lgb_n_estimators", 100, 500),
                    "max_depth": trial.suggest_int("lgb_max_depth", 3, 12),
                    "learning_rate": trial.suggest_float("lgb_learning_rate", 0.01, 0.3),
                    "num_leaves": trial.suggest_int("lgb_num_leaves", 20, 150),
                    "subsample": trial.suggest_float("lgb_subsample", 0.5, 1.0),
                    "objective": "multiclass",
                    "random_state": 42,
                    "n_jobs": -1,
                    "verbose": -1
                }
                model = LGBMClassifier(**params)

            mlflow.log_params(params)

            pipeline = Pipeline(steps=[
                ("preprocessor", self.preprocessor),
                ("smote", smote),
                ("model", model)
            ])

            score = cross_val_score(
                pipeline, X_train, y_train, cv=5, scoring="f1_weighted", n_jobs=-1
            ).mean()

            mlflow.log_metric("f1_weighted_cv", score)
            
            return score

    def tune_parameters(self, X_train, y_train, n_trials=50, patience=15):
        with mlflow.start_run(run_name="Optuna_Hyperparameter_Tuning") as parent_run:
            study = optuna.create_study(direction="maximize")
            early_stopping = EarlyStoppingCallback(patience=patience, min_delta=0.001)
            
            study.optimize(
                lambda trial: self._objective(trial, X_train, y_train), 
                n_trials=n_trials, 
                callbacks=[early_stopping]
            )
            
            mlflow.log_metric("best_f1_weighted", study.best_value)
            for param_key, param_val in study.best_params.items():
                mlflow.log_param(f"best_{param_key}", param_val)
                
            print(f"\nTuning Selesai. Model Terbaik: {study.best_params['model_type']} dengan F1-Score: {study.best_value:.4f}")
            return study.best_params, study.best_value

    def build_and_fit_final_pipeline(self, X_train, y_train, best_params):
        params_copy = best_params.copy()
        
        model_type = params_copy.pop("model_type")
        smote_k = params_copy.pop("smote_k_neighbors", 5)
        smote = SMOTE(k_neighbors=smote_k, random_state=42)

        if model_type == "xgboost":
            cleaned_params = {k.replace("xgb_", ""): v for k, v in params_copy.items()}
            cleaned_params.update({"objective": "multi:softprob", "eval_metric": "mlogloss", "random_state": 42, "n_jobs": -1})
            best_model = XGBClassifier(**cleaned_params)
            
        elif model_type == "random_forest":
            cleaned_params = {k.replace("rf_", ""): v for k, v in params_copy.items()}
            cleaned_params.update({"random_state": 42, "n_jobs": -1})
            best_model = RandomForestClassifier(**cleaned_params)
            
        elif model_type == "lightgbm":
            cleaned_params = {k.replace("lgb_", ""): v for k, v in params_copy.items()}
            cleaned_params.update({"objective": "multiclass", "random_state": 42, "n_jobs": -1, "verbose": -1})
            best_model = LGBMClassifier(**cleaned_params)

        self.final_pipeline = Pipeline(steps=[
            ("preprocessor", self.preprocessor),
            ("smote", smote),
            ("model", best_model)
        ])
        
        with mlflow.start_run(run_name=f"Final_Model_{model_type}"):
            self.final_pipeline.fit(X_train, y_train)
            mlflow.sklearn.log_model(self.final_pipeline, artifact_path="final_pipeline_model")
            print(f"-> Final pipeline ({model_type}) berhasil di-fit dan dicatat ke MLflow Artifacts.")
            
        return self.final_pipeline

    def save_model(self, file_path: str):
        if self.final_pipeline is not None:
            joblib.dump(self.final_pipeline, file_path)
            print(f"-> Model pipeline berhasil disimpan secara lokal ke '{file_path}'")
        else:
            print("Error: Belum ada pipeline yang di-training untuk disimpan!")

    def load_model(self, file_path: str):
        self.final_pipeline = joblib.load(file_path)
        print(f"-> Berhasil memuat model pipeline dari '{file_path}'")
        return self.final_pipeline