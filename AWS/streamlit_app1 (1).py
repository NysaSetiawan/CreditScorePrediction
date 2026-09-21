import json
import os
import boto3
import streamlit as st
import pandas as pd
from botocore.exceptions import ClientError, NoCredentialsError

# Configuration
ENDPOINT_NAME = os.environ.get("ENDPOINT_NAME", "credit-score-endpointv28")
REGION = os.environ.get("AWS_REGION", "us-east-1")

@st.cache_resource
def get_runtime_client():
    return boto3.client("sagemaker-runtime", region_name=REGION)

def invoke_endpoint(features: list) -> dict:
    runtime = get_runtime_client()
    payload = {"instances": [features]}
    response = runtime.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType="application/json",
        Accept="application/json",
        Body=json.dumps(payload),
    )
    return json.loads(response["Body"].read().decode("utf-8"))

st.set_page_config(page_title="Credit Approval Prediction", layout="wide")
st.title("Credit Approval Prediction")

with st.form("credit_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Personal Information")
        unnamed = st.number_input(
            "Record Number (Unnamed: 0)",
            min_value=0,
            value=1
        )
    
        record_id = st.text_input(
            "ID",
            value="0x2b4a"
        )
            
        customer_id = st.text_input(
            "Customer ID",
            value="CUST-9999"
        )
    
        customer_name = st.text_input(
            "Customer Name",
            value="Unknown"
        )
    
        ssn = st.text_input(
            "SSN",
            value="000-00-0000"
        )
        month = st.selectbox("Evaluation Month", ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'])
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
        occupation = st.selectbox("Occupation", ['Scientist', 'Teacher', 'Engineer', 'Entrepreneur', 'Developer', 'Doctor', 'Media_Manager', 'Manager', 'Insurance_Changer', 'Mechanic', 'Accountant', 'Architect', 'Writer', 'Musician', 'Lawyer', 'Unknown'])
        annual_income = st.number_input("Annual Income ($)", min_value=0.0, value=50000.0)
        monthly_inhand_salary = st.number_input("Monthly In-hand Salary ($)", min_value=0.0, value=4000.0)

    with col2:
        st.subheader("Financial Information")
        num_bank_accounts = st.number_input("Number of Bank Accounts", min_value=0, value=2)
        num_credit_card = st.number_input("Number of Active Credit Cards", min_value=0, value=3)
        interest_rate = st.number_input("Credit Interest Rate (%)", min_value=0.0, value=12.0)
        num_of_loan = st.number_input("Number of Active Loans", min_value=0, value=1)
        type_of_loan = st.text_input("Types of Loans", value="Personal Loan, Home Loan")
        delay_from_due_date = st.number_input("Average Delay from Due Date (Days)", min_value=0, value=5)
        num_of_delayed_payment = st.number_input("Number of Delayed Payments", min_value=0, value=2)

    with col3:
        st.subheader("Loan & Behavioral Profile")
        changed_credit_limit = st.number_input("Changed Credit Limit ($)", min_value=0.0, value=10.0)
        num_credit_inquiries = st.number_input("Number of Credit Inquiries", min_value=0, value=1)
        credit_mix = st.selectbox("Credit Mix", ['Bad', 'Standard', 'Good', 'Unknown'])
        outstanding_debt = st.number_input("Outstanding Debt ($)", min_value=0.0, value=1500.0)
        credit_utilization_ratio = st.number_input("Credit Utilization Ratio (%)", min_value=0.0, max_value=100.0, value=30.0)
        credit_history_age_text = st.text_input("Credit History Age (Text format)", value="22 Years and 4 Months")
        payment_of_min_amount = st.selectbox("Paying Minimum Amount Only?", ['No', 'Yes'])
        total_emi_per_month = st.number_input("Monthly EMI Assessment ($)", min_value=0.0, value=300.0)
        amount_invested_monthly = st.number_input("Amount Invested Monthly ($)", min_value=0.0, value=100.0)
        payment_behaviour = st.selectbox("Payment Behaviour", ['Low_spent_Small_value_payments', 'High_spent_Medium_value_payments', 'Low_spent_Medium_value_payments', 'High_spent_Large_value_payments', 'Low_spent_Large_value_payments', 'High_spent_Small_value_payments'])
        monthly_balance = st.number_input("Monthly Balance ($)", min_value=0.0, value=500.0)

    submitted = st.form_submit_button("Calculate Customer Credit Score")

if submitted:

    features = [
        unnamed,
        record_id,
        customer_id,
        customer_name,
        ssn,

        month,
        age,
        occupation,
        annual_income,
        monthly_inhand_salary,
        num_bank_accounts,
        num_credit_card,
        interest_rate,
        num_of_loan,
        type_of_loan,
        delay_from_due_date,
        num_of_delayed_payment,
        changed_credit_limit,
        num_credit_inquiries,
        credit_mix,
        outstanding_debt,
        credit_utilization_ratio,
        credit_history_age_text,
        payment_of_min_amount,
        total_emi_per_month,
        amount_invested_monthly,
        payment_behaviour,
        monthly_balance
    ]

    # Validasi jumlah fitur
    if len(features) != 28:
        st.error(
            f"Feature count mismatch. Expected 28 features, got {len(features)}"
        )
        st.stop()

    try:
        result = invoke_endpoint(features)

        label = result["labels"][0]
        probs = result["probabilities"][0]

        st.subheader("Prediction Evaluation Summary")

        if label == "Good":
            st.success("Customer Credit Score Rank: GOOD")
        elif label == "Standard":
            st.info("Customer Credit Score Rank: STANDARD")
        else:
            st.error("Customer Credit Score Rank: POOR")

        prob_df = pd.DataFrame({
            "Credit Score": ["Poor", "Standard", "Good"],
            "Probability": probs
        })

        st.subheader("Classification Probabilities")

        st.dataframe(
            prob_df,
            use_container_width=True
        )

    except NoCredentialsError:
        st.error(
            "AWS credentials not found. Configure AWS credentials or attach an IAM Role."
        )

    except ClientError as e:
        st.error(
            f"AWS Error: {e.response['Error'].get('Message', str(e))}"
        )

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")