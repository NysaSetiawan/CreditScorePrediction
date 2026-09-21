import optuna
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler, OrdinalEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
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
    def __init__(self):
            # 1. Definisikan Transformer
            num_transformer = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")), 
                ("scaler", RobustScaler())
            ])
            
            ord_transformer = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="constant", fill_value=-1)),
                ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
            ])
            
            cat_transformer = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                ("encoder", OneHotEncoder(handle_unknown="ignore"))
            ])
    
            self.preprocessor = ColumnTransformer(
                transformers=[
                    ("num", num_transformer, config.num_cols),
                    ("ord", ord_transformer, config.ordinal_cols),
                    ("cat", cat_transformer, config.cat_cols)
                ],
                remainder='drop' 
            )
            
            self.final_pipeline = None

    def _objective(self, trial, X_train, y_train):
        smote_k_neighbors = trial.suggest_int("smote_k_neighbors", 3, 7)
        smote = SMOTE(k_neighbors=smote_k_neighbors, random_state=42)

        lgb_params = {
            "objective": "multiclass",
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1,
            "n_estimators": trial.suggest_int("lgb_n_estimators", 100, 300),
            "max_depth": trial.suggest_int("lgb_max_depth", 3, 20),
            "num_leaves": trial.suggest_int("lgb_num_leaves", 20, 200),
            "learning_rate": trial.suggest_float("lgb_learning_rate", 0.005, 0.3, log=True),
            "subsample": trial.suggest_float("lgb_subsample", 0.5, 1.0),
            "bagging_freq": trial.suggest_int("lgb_bagging_freq", 1, 7), 
            "colsample_bytree": trial.suggest_float("lgb_colsample_bytree", 0.5, 1.0),
            "min_child_samples": trial.suggest_int("lgb_min_child_samples", 10, 100),
            "reg_alpha": trial.suggest_float("lgb_reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("lgb_reg_lambda", 1e-8, 10.0, log=True)
        }
        
        model = LGBMClassifier(**lgb_params)
        pipeline = Pipeline(steps=[
            ("preprocessor", self.preprocessor),
            ("smote", smote),
            ("model", model)
        ])

        score = cross_val_score(
            pipeline, X_train, y_train, cv=3, scoring="f1_weighted", n_jobs=1
        ).mean()
        return score

    def tune_parameters(self, X_train, y_train, n_trials=60, patience=5):
        study = optuna.create_study(direction="maximize")
        early_stopping = EarlyStoppingCallback(patience=patience, min_delta=0.001)
        study.optimize(
            lambda trial: self._objective(trial, X_train, y_train), 
            n_trials=n_trials, 
            callbacks=[early_stopping]
        )
        return study.best_params, study.best_value

    def build_and_fit_final_pipeline(self, X_train, y_train, best_params):
        params_copy = best_params.copy()
        smote_k = params_copy.pop("smote_k_neighbors", 5) 
        smote = SMOTE(k_neighbors=smote_k, random_state=42)

        cleaned_lgb_params = {k.replace("lgb_", ""): v for k, v in params_copy.items()}
        cleaned_lgb_params.update({"random_state": 42, "n_jobs": -1})

        best_model = LGBMClassifier(**cleaned_lgb_params)

        self.final_pipeline = Pipeline(steps=[
            ("preprocessor", self.preprocessor),
            ("smote", smote),
            ("model", best_model)
        ])
        
        self.final_pipeline.fit(X_train, y_train)
        return self.final_pipeline

    def save_model(self, file_path: str):
        if self.final_pipeline is not None:
            joblib.dump(self.final_pipeline, file_path)
            print(f"-> Model pipeline berhasil disimpan ke '{file_path}'")
        else:
            print("Error: Belum ada pipeline yang di-training untuk disimpan!")

    def load_model(self, file_path: str):
        self.final_pipeline = joblib.load(file_path)
        print(f"-> Berhasil memuat model pipeline dari '{file_path}'")
        return self.final_pipeline