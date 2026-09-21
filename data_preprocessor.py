import pandas as pd
import numpy as np
import re
import config

class DataPreprocessor:
    def __init__(self):
        pass

    @staticmethod
    def _convert_credit_history_age(x):
        if pd.isna(x):
            return np.nan
        x = str(x)
        years = 0
        months = 0
        if "Years" in x:
            years = int(x.split("Years")[0].strip())
        if "Months" in x:
            try:
                months = int(x.split("Months")[0].split()[-1].strip())
            except (IndexError, ValueError):
                months = 0
        return years * 12 + months

    @staticmethod
    def _count_loans(type_str):
        if pd.isna(type_str) or str(type_str).strip() == "":
            return 0
        s = str(type_str).strip()
        s = re.sub(r"\s+and\s+", ",", s)
        items = [x.strip() for x in s.split(",") if x.strip() != ""]
        return len(items)

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        drop_cols = ["Unnamed: 0", "ID", "Customer_ID", "Name", "SSN"]
        df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
        
        for col in config.not_numeric_columns:
            if col in df.columns:
                df[col] = df[col].astype(str).str.replace("_", "", regex=False)
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        if "Occupation" in df.columns:
            df["Occupation"] = df["Occupation"].replace("_______", "Unknown")
        if "Credit_Mix" in df.columns:
            df["Credit_Mix"] = df["Credit_Mix"].replace("_", "Unknown")
        if "Payment_of_Min_Amount" in df.columns:
            df["Payment_of_Min_Amount"] = df["Payment_of_Min_Amount"].replace("NM", "No")
        if "Payment_Behaviour" in df.columns:
            df["Payment_Behaviour"] = df["Payment_Behaviour"].replace("!@9#%8", "Unknown")
            
        neg_cols = ["Age", "Num_Bank_Accounts", "Num_of_Loan", "Delay_from_due_date", 
                    "Num_of_Delayed_Payment", "Changed_Credit_Limit", "Monthly_Balance"]
        for col in neg_cols:
            if col in df.columns:
                df.loc[df[col] < 0, col] = np.nan
                
        cat_fill_cols = ["Credit_Mix", "Occupation", "Payment_Behaviour", "Payment_of_Min_Amount"]
        for col in cat_fill_cols:
            if col in df.columns:
                df[col] = df[col].fillna("Unknown")
                
        return df

    def feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        epsilon = 1e-5
        
        if "Credit_History_Age" in df.columns:
            df["Credit_History_Age"] = df["Credit_History_Age"].apply(self._convert_credit_history_age)
            
        if 'Monthly_Balance' in df.columns and 'Monthly_Inhand_Salary' in df.columns:
            df['Savings_Rate'] = df['Monthly_Balance'] / (df['Monthly_Inhand_Salary'] + epsilon)

        
        return df