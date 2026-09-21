import streamlit as tf
import pandas as pd
import numpy as np
import joblib
import os

import config
from data_preprocessor import DataPreprocessor

tf.set_page_config(page_title="Credit Approval Prediction", layout="wide")

def load_model_pipeline():
    model_path = "notebook/best_model_pipeline.pkl"
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

pipeline = load_model_pipeline()

tf.title("Credit Approval Prediction")
tf.markdown("Please fill out the customer metrics below to evaluate their creditworthiness instantly.")

if pipeline is None:
    tf.error("Model pipeline file not found. Please ensure 'notebook/best_lgbm_pipeline.pkl' exists.")
else:
    preprocessor = DataPreprocessor()

    with tf.form("credit_form"):
        col1, col2, col3 = tf.columns(3)

        with col1:
            tf.subheader("Personal Information")
            customer_id = tf.text_input("Customer ID", value="CUS_0x697f")
            id_val = tf.text_input("ID", value="0x45c7")
            name = tf.text_input("Customer Name", value="Nysa Setiawan")
            ssn = tf.text_input("SSN (Social Security Number)", value="312-56-4208")
            month = tf.selectbox("Evaluation Month", ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August','September', 'October', 'November', 'December'])
            age = tf.number_input("Age", min_value=18, max_value=100, value=30) 
            occupation = tf.selectbox("Occupation", ['Scientist', 'Teacher', 'Engineer', 'Entrepreneur', 'Developer', 'Doctor', 'Media_Manager', 'Manager', 'Insurance_Changer', 'Mechanic', 'Accountant', 'Architect', 'Writer', 'Musician', 'Lawyer', 'Unknown'])
            annual_income = tf.number_input("Annual Income ($)", min_value=0.0, value=50000.0)
            monthly_inhand_salary = tf.number_input("Monthly In-hand Salary ($)", min_value=0.0, value=4000.0)

        with col2:
            tf.subheader("Financial Information")
            num_bank_accounts = tf.number_input("Number of Bank Accounts", min_value=0, value=2)
            num_credit_card = tf.number_input("Number of Active Credit Cards", min_value=0, value=3)
            interest_rate = tf.number_input("Credit Interest Rate (%)", min_value=0.0, value=12.0)
            num_of_loan = tf.number_input("Number of Active Loans", min_value=0, value=1)
            type_of_loan = tf.text_input("Types of Loans (Separate with commas or 'and')", value="Personal Loan, Home Loan")
            delay_from_due_date = tf.number_input("Average Delay from Due Date (Days)", min_value=0, value=5)
            num_of_delayed_payment = tf.number_input("Number of Delayed Payments", min_value=0, value=2)

        with col3:
            tf.subheader("Loan & Behavioral Profile")
            changed_credit_limit = tf.number_input("Changed Credit Limit ($)", min_value=0.0, value=10.0)
            num_credit_inquiries = tf.number_input("Number of Credit Inquiries", min_value=0, value=1)
            credit_mix = tf.selectbox("Credit Mix", ['Bad', 'Standard', 'Good', 'Unknown'])
            outstanding_debt = tf.number_input("Outstanding Debt ($)", min_value=0.0, value=1500.0)
            credit_utilization_ratio = tf.number_input("Credit Utilization Ratio (%)", min_value=0.0, max_value=100.0, value=30.0)
            credit_history_age_text = tf.text_input("Credit History Age (Text format)", value="22 Years and 4 Months")
            payment_of_min_amount = tf.selectbox("Paying Minimum Amount Only?", ['No', 'Yes'])
            total_emi_per_month = tf.number_input("Monthly EMI Assessment ($)", min_value=0.0, value=300.0)
            amount_invested_monthly = tf.number_input("Amount Invested Monthly ($)", min_value=0.0, value=100.0)
            payment_behaviour = tf.selectbox("Payment Behaviour", config.payment_order[1:])
            monthly_balance = tf.number_input("Monthly Balance ($)", min_value=0.0, value=500.0)

        submitted = tf.form_submit_button("Calculate Customer Credit Score")

    if submitted:
        input_dict = {
            'Month': [month], 'Age': [age], 'Occupation': [occupation], 'Annual_Income': [annual_income],
            'Monthly_Inhand_Salary': [monthly_inhand_salary], 'Num_Bank_Accounts': [num_bank_accounts],
            'Num_Credit_Card': [num_credit_card], 'Interest_Rate': [interest_rate], 'Num_of_Loan': [num_of_loan],
            'Type_of_Loan': [type_of_loan], 'Delay_from_due_date': [delay_from_due_date],
            'Num_of_Delayed_Payment': [num_of_delayed_payment], 'Changed_Credit_Limit': [changed_credit_limit],
            'Num_Credit_Inquiries': [num_credit_inquiries], 'Credit_Mix': [credit_mix], 'Outstanding_Debt': [outstanding_debt],
            'Credit_Utilization_Ratio': [credit_utilization_ratio], 'Credit_History_Age': [credit_history_age_text],
            'Payment_of_Min_Amount': [payment_of_min_amount], 'Total_EMI_per_month': [total_emi_per_month],
            'Amount_invested_monthly': [amount_invested_monthly], 'Payment_Behaviour': [payment_behaviour],
            'Monthly_Balance': [monthly_balance]
        }
        
        raw_df = pd.DataFrame(input_dict)
        
        try:
            cleaned_df = preprocessor.clean_data(raw_df)
            featured_df = preprocessor.feature_engineering(cleaned_df)
            
            prediction_id = pipeline.predict(featured_df)[0]
            probabilities = pipeline.predict_proba(featured_df)[0]
            
            result_label = config.target_names[prediction_id]
            
            tf.subheader("Prediction Evaluation Summary:")
            
            if result_label == "Good":
                tf.success(f"Customer Credit Score Rank: **GOOD** (High Trust Category)")
            elif result_label == "Standard":
                tf.info(f"Customer Credit Score Rank: **STANDARD** (Moderate Risk Category)")
            else:
                tf.error(f"Customer Credit Score Rank: **POOR** (High Default Risk Category)")
                
            tf.markdown("**Classification Probabilities:**")
            prob_df = pd.DataFrame([probabilities], columns=config.target_names)
            tf.dataframe(prob_df.style.format("{:.2%}"))
            
        except Exception as e:
            tf.error(f"An error occurred while compiling input features: {str(e)}")