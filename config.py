import numpy as np

not_numeric_columns = [
    "Age", "Annual_Income", "Num_of_Loan", "Num_of_Delayed_Payment", 
    "Changed_Credit_Limit", "Outstanding_Debt", "Amount_invested_monthly", "Monthly_Balance"
]

missing_cat_cols = ["Credit_History_Age", "Payment_Behaviour", "Occupation"]

target_mapping = {'Poor': 0, 'Standard': 1, 'Good': 2}
target_names = ['Poor', 'Standard', 'Good']

payment_order = [
    'Unknown',
    'Low_spent_Small_value_payments', 'Low_spent_Medium_value_payments', 'Low_spent_Large_value_payments',
    'High_spent_Small_value_payments', 'High_spent_Medium_value_payments', 'High_spent_Large_value_payments'
]
num_cols = [
    'Age', 'Annual_Income', 'Monthly_Inhand_Salary', 'Num_Bank_Accounts', 
    'Num_Credit_Card', 'Interest_Rate', 'Num_of_Loan', 'Delay_from_due_date', 
    'Num_of_Delayed_Payment', 'Changed_Credit_Limit', 'Num_Credit_Inquiries', 
    'Outstanding_Debt', 'Credit_Utilization_Ratio', 'Credit_History_Age', 
    'Total_EMI_per_month', 'Amount_invested_monthly', 'Monthly_Balance', 
    'Savings_Rate' ]

ordinal_cols = ['Payment_Behaviour']

cat_cols = ['Month', 'Occupation', 'Payment_of_Min_Amount', 'Credit_Mix']