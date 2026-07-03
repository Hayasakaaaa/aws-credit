"""
Streamlit UI for the Wine Quality classifier hosted on SageMaker.

Reads endpoint name and region from environment variables.
boto3 picks up AWS credentials from:
  - the EC2 instance profile (when running on EC2 with LabInstanceProfile), OR
  - ~/.aws/credentials (when running locally)
"""

import json
import os
import pandas as pd
import boto3
import streamlit as st
from botocore.exceptions import ClientError, NoCredentialsError


ENDPOINT_NAME = os.environ.get("ENDPOINT_NAME", "credit-endpoint")
REGION = os.environ.get("AWS_REGION", "us-east-1")


@st.cache_resource
def get_runtime_client():
    return boto3.client("sagemaker-runtime", region_name=REGION)


def invoke_endpoint(features: list[float]) -> dict:
    runtime = get_runtime_client()
    payload = {"instances": [features]}
    response = runtime.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType="application/json",
        Accept="application/json",
        Body=json.dumps(payload),
    )
    return json.loads(response["Body"].read().decode("utf-8"))


st.title('Financial Profile & Credit Score Analysis')
st.write("Masukkan data finansial nasabah untuk memprediksi Credit Score dan Status Finansial.")

st.divider()
    
col1, col2 = st.columns(2)
with col1:
    st.subheader("Profil Dasar & Rekening")
    month = st.selectbox("Month (Bulan)", ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August'])
    age = st.number_input("Age (Usia)", min_value=18, max_value=100, value=30)
    annual_income = st.number_input("Annual Income", min_value=0.0, value=50000.0, step=1000.0)
    monthly_inhand_salary = st.number_input("Monthly Inhand Salary", min_value=0.0, value=4000.0, step=100.0)
    num_bank_accounts = st.number_input("Number of Bank Accounts", min_value=0, max_value=20, value=3)
    num_credit_card = st.number_input("Number of Credit Cards", min_value=0, max_value=20, value=2)
    interest_rate = st.number_input("Interest Rate (%)", min_value=0.0, max_value=100.0, value=12.0)
    num_of_loan = st.number_input("Number of Loans", min_value=0, max_value=20, value=1)
        
with col2:
    st.subheader("Riwayat Kredit & Hutang")
    delay_from_due_date = st.number_input("Delay from Due Date (Days)", min_value=0, value=5)
    num_of_delayed_payment = st.number_input("Number of Delayed Payments", min_value=0, value=1)
    changed_credit_limit = st.number_input("Changed Credit Limit", min_value=-50.0, max_value=100.0, value=10.0, step=0.5)
    num_credit_inquiries = st.number_input("Number of Credit Inquiries", min_value=0, max_value=50, value=2)
    credit_mix = st.selectbox("Credit Mix", ["Bad", "Standard", "Good"]) # Sesuai dengan encoder kita sebelumnya
    credit_score = st.selectbox("Credit Mix", ["Poor", "Standard", "Good"])
    outstanding_debt = st.number_input("Outstanding Debt", min_value=0.0, value=1500.0, step=100.0)
    credit_utilization_ratio = st.slider("Credit Utilization Ratio (%)", 0.0, 100.0, 30.0)
        
st.divider()
st.subheader("Pengeluaran & Investasi Bulanan")
col3, col4 = st.columns(2)
    
with col3:
    payment_of_min_amount = st.selectbox("Payment of Minimum Amount", ["No", "Yes"])
    total_emi_per_month = st.number_input("Total EMI per Month", min_value=0.0, value=200.0, step=10.0)
        
with col4:
    amount_invested_monthly = st.number_input("Amount Invested Monthly", min_value=0.0, value=500.0, step=50.0)
    monthly_balance = st.number_input("Monthly Balance", min_value=-5000.0, value=1000.0, step=100.0)

data = {
        'Month': month,
        'Age': age,
        'Annual_Income': annual_income,
        'Monthly_Inhand_Salary': monthly_inhand_salary,
        'Num_Bank_Accounts': num_bank_accounts,
        'Num_Credit_Card': num_credit_card,
        'Interest_Rate': interest_rate,
        'Num_of_Loan': num_of_loan,
        'Delay_from_due_date': delay_from_due_date,
        'Num_of_Delayed_Payment': num_of_delayed_payment,
        'Changed_Credit_Limit': changed_credit_limit,
        'Num_Credit_Inquiries': num_credit_inquiries,
        'Credit_Mix': credit_mix,
        'Credit_Score': credit_score,
        'Outstanding_Debt': outstanding_debt,
        'Credit_Utilization_Ratio': credit_utilization_ratio,
        'Payment_of_Min_Amount': payment_of_min_amount,
        'Total_EMI_per_month': total_emi_per_month,
        'Amount_invested_monthly': amount_invested_monthly,
        'Monthly_Balance': monthly_balance
}
    
df = pd.DataFrame([data])

model_cls = load_models()

st.divider()
    
    # Tombol Aksi Prediksi
if st.button("Analyze Financial Profile", type="primary", use_container_width=True):
        
    hasil_kategori = model_cls.predict(df)[0]
        
    st.subheader("Hasil Analisis Model:")
    col_hasil1 = st.columns(1)
        
    with col_hasil1:
        st.metric(label="Prediksi Payment of Minimum Amount", value=str(hasil_kategori))


