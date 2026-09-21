import json
import os
import sys
import logging
import joblib
import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - [ENDPOINT_LOG] - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

sys.path.insert(0, os.path.dirname(__file__))
import config

json_content_type = "application/json"
csv_content_type = "text/csv"

feature_names = [
    'Unnamed: 0', 'ID', 'Customer_ID', 'Name', 'SSN',
    'Month', 'Age', 'Occupation', 
    'Annual_Income', 'Monthly_Inhand_Salary', 'Num_Bank_Accounts', 
    'Num_Credit_Card', 'Interest_Rate', 'Num_of_Loan', 'Type_of_Loan', 
    'Delay_from_due_date', 'Num_of_Delayed_Payment', 'Changed_Credit_Limit', 
    'Num_Credit_Inquiries', 'Credit_Mix', 'Outstanding_Debt', 
    'Credit_Utilization_Ratio', 'Credit_History_Age', 'Payment_of_Min_Amount', 
    'Total_EMI_per_month', 'Amount_invested_monthly', 'Payment_Behaviour', 
    'Monthly_Balance'
]

def model_fn(model_dir):
    logger.info("[CHECKPOINT 1] Memulai model_fn (Pemuatan Model)")
    try:
        model_path = os.path.join(model_dir, "model.joblib")
        logger.info(f"Mencari biner model di jalur: {model_path}")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"File model.joblib tidak ditemukan di folder {model_dir}")
            
        model = joblib.load(model_path)
        logger.info("[CHECKPOINT 1 SUCCESS] Model pipeline sukses dimuat ke memori!")
        return model
    except Exception as e:
        logger.error(f"!!! [CHECKPOINT 1 FAILED] Gagal memuat model. Error: {str(e)}")
        raise e

def input_fn(request_body, request_content_type: str) -> pd.DataFrame:
    logger.info("[CHECKPOINT 2] Memulai input_fn (Parsing Payload Kiriman)")
    logger.info(f"Menerima Content-Type: {request_content_type}")
    
    try:
        if request_content_type == json_content_type:
            logger.info("Memproses payload format JSON...")
            payload = json.loads(request_body)
            
            if "instances" not in payload:
                raise KeyError("Payload JSON wajib mengandung key utama 'instances'")
                
            instances = payload["instances"]
            
            # Cek jika panjang baris tidak sama dengan jumlah feature_names
            if len(instances[0]) != len(feature_names):
                logger.error(f"Mismatch kolom! Diharapkan: {len(feature_names)}, Diterima: {len(instances[0])}")
                
                # Identifikasi kolom yang berlebih
                if len(instances[0]) > len(feature_names):
                    logger.error(f"Data tambahan (fitur ke-24 dst): {instances[0][len(feature_names):]}")
                else:
                    logger.error(f"Kolom yang kurang: {feature_names[len(instances[0]):]}")

            logger.info(f"Feature names count: {len(feature_names)}")
            logger.info(f"Received row count: {len(instances[0])}")
            logger.info(f"Received row data: {instances[0]}")
            df = pd.DataFrame(instances, columns=feature_names)
            logger.info(f"[CHECKPOINT 2 SUCCESS] JSON sukses diubah ke DataFrame. Dimensi: {df.shape}")
            return df

        raise ValueError(f"Unsupported content type: {request_content_type}")
        
    except Exception as e:
        logger.error(f"!!! [CHECKPOINT 2 FAILED] Gagal memparsing input data. Error: {str(e)}")
        raise e

def predict_fn(input_data: pd.DataFrame, pipeline) -> dict:
    logger.info("[CHECKPOINT 3] Memulai predict_fn (Preprocessing & Inference)")
    try:
        from data_preprocessor import DataPreprocessor
        preprocessor = DataPreprocessor()
        
        #1. clean data
        logger.info("Mengeksekusi preprocessor.clean_data()...")
        cleaned_data = preprocessor.clean_data(input_data)
        
        # 2. Feature Engineering
        logger.info("Mengeksekusi preprocessor.feature_engineering()...")
        featured_data = preprocessor.feature_engineering(cleaned_data)
        
        # Pastikan kolom target dibuang
        if 'Credit_Score' in featured_data.columns:
            featured_data = featured_data.drop(columns=['Credit_Score'])
            
        # Cek kolom yang diharapkan model vs kolom yang tersedia
        if hasattr(pipeline, "feature_names_in_"):
            expected = list(pipeline.feature_names_in_)
            actual = list(featured_data.columns)
            missing = [c for c in expected if c not in actual]
            if missing:
                logger.error(f"Kolom hilang dari input: {missing}")
                logger.info(f"Kolom yang ada di data: {actual}")
                raise ValueError(f"Kolom hilang: {missing}")
        
        logger.info(f"Daftar kolom final yang dikirim ke model: {list(featured_data.columns)}")
        
       
        logger.info("Mengeksekusi pipeline.predict_proba()...")
        probs = pipeline.predict_proba(featured_data)
        
        class_ids = np.argmax(probs, axis=1)
        labels = [config.target_names[int(i)] for i in class_ids]
        
        logger.info("[CHECKPOINT 3 SUCCESS] Seluruh proses inferensi berhasil")
        return {
            "probabilities": probs.tolist(),
            "predictions": class_ids.tolist(),
            "labels": labels,
        }
        
    except Exception as e:
        logger.error(f"!!! [CHECKPOINT 3 FAILED] Terjadi kesalahan: {str(e)}")
        raise e

def output_fn(prediction: dict, accept_content_type: str):
    logger.info("[CHECKPOINT 4] Memulai output_fn (Format Respon Balikan)")
    try:
        if accept_content_type == json_content_type:
            response_body = json.dumps(prediction)
            logger.info("[CHECKPOINT 4 SUCCESS] Respon JSON siap dikirim ke client.")
            return response_body, json_content_type
            
        raise ValueError(f"Unsupported accept type: {accept_content_type}")
    except Exception as e:
        logger.error(f"!!! [CHECKPOINT 4 FAILED] Gagal mengonversi output prediksi. Error: {str(e)}")
        raise e
