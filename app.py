import streamlit as st

from predictor import load_bundle, predict

st.set_page_config(page_title="Customer churn demo", page_icon="📉")
st.title("Will this customer churn?")
st.caption("A demo of the balanced XGBoost model from demo.ipynb")

@st.cache_resource
def get_bundle():
    return load_bundle()

try:
    bundle = get_bundle()
except FileNotFoundError:
    st.error("Model artifact not found. Run `python train_model.py` first.")
    st.stop()

with st.form("customer"):
    left, right = st.columns(2)
    with left:
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior = st.selectbox("Senior citizen", [0, 1])
        partner = st.selectbox("Has partner", ["Yes", "No"])
        dependents = st.selectbox("Has dependents", ["Yes", "No"])
        tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
        phone = st.selectbox("Phone service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple lines", ["No", "Yes", "No phone service"])
        internet = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
    with right:
        security = st.selectbox("Online security", ["No", "Yes", "No internet service"])
        backup = st.selectbox("Online backup", ["No", "Yes", "No internet service"])
        device = st.selectbox("Device protection", ["No", "Yes", "No internet service"])
        support = st.selectbox("Tech support", ["No", "Yes", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        streaming_movies = st.selectbox("Streaming movies", ["No", "Yes", "No internet service"])
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless = st.selectbox("Paperless billing", ["Yes", "No"])
        payment = st.selectbox("Payment method", ["Bank transfer (automatic)", "Credit card", "Electronic check", "Mailed check"])
        monthly = st.number_input("Monthly charges", min_value=0.0, value=70.0)
        total = st.number_input("Total charges", min_value=0.0, value=840.0)

    submitted = st.form_submit_button("Predict churn")

if submitted:
    customer = {
        "gender": gender, "SeniorCitizen": senior, "Partner": partner, "Dependents": dependents,
        "tenure": tenure, "PhoneService": phone, "MultipleLines": multiple_lines,
        "InternetService": internet, "OnlineSecurity": security, "OnlineBackup": backup,
        "DeviceProtection": device, "TechSupport": support, "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies, "Contract": contract, "PaperlessBilling": paperless,
        "PaymentMethod": payment, "MonthlyCharges": monthly, "TotalCharges": total,
    }
    result = predict(customer, bundle)
    if result["will_churn"]:
        st.error(f"Likely to churn: {result['churn_probability']:.1%} probability")
    else:
        st.success(f"Likely to stay: {1 - result['churn_probability']:.1%} probability")
    st.progress(result["churn_probability"], text="Churn probability")
    st.caption(f"Model: {result['model_version']} | Decision threshold: {result['threshold']:.2f}")
