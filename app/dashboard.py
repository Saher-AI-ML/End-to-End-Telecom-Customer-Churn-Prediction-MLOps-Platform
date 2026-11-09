"""Streamlit dashboard for the WE churn model.

It never loads the model itself: it only sends requests to the FastAPI service.
Start the API first, then run from the project root:
    streamlit run app/dashboard.py
"""
import os

import pandas as pd
import requests
import streamlit as st

# Where the API lives. Locally this is the default; in Docker Compose we will set API_URL=http://api:8000
API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="WE Churn Predictor")
st.title("WE Customer Churn Predictor")


@st.cache_data(ttl=60)  # remember the answer for 60 seconds so the page does not call the API on every click
def get_json(path):
    response = requests.get(f"{API_URL}{path}", timeout=10)
    response.raise_for_status()
    return response.json()


# Ask the API what the model expects (dropdown values, threshold). If the API is off, say so and stop.
try:
    info = get_json("/model-info")
except requests.exceptions.RequestException:
    st.error(f"Cannot reach the API at {API_URL}. Start it first with: uvicorn api.main:app --reload")
    st.stop()

categories = info["categories"]
ranges = info["numeric_ranges"]


def pick(options, preferred):
    """Index of the preferred option in a dropdown (first option if it is missing)."""
    return options.index(preferred) if preferred in options else 0


def range_help(column):
    return f"Typical range in the training data: {ranges[column]['min']:g} to {ranges[column]['max']:g}"


tab_predict, tab_performance = st.tabs(["Predict", "Model performance"])

# ---------------------------------------------------------------- Tab 1: prediction form
with tab_predict:
    st.write("Enter a customer's details to estimate how likely they are to churn.")

    with st.form("customer_form"):  # a form sends everything at once when the button is pressed
        left, right = st.columns(2)
        with left:
            governorate = st.selectbox("Governorate", categories["governorate"],
                                       index=pick(categories["governorate"], "Cairo"))
            customer_type = st.selectbox("Customer type", categories["customer_type"],
                                         index=pick(categories["customer_type"], "Individual"))
            subscription_plan = st.selectbox("Subscription plan", categories["subscription_plan"],
                                             index=pick(categories["subscription_plan"], "WE Plus 40Mbps"))
            payment_method = st.selectbox("Payment method", categories["payment_method"],
                                          index=pick(categories["payment_method"], "Credit Card"))
        with right:
            monthly_bill = st.number_input("Monthly bill (EGP)", min_value=0.0, value=400.0, step=10.0,
                                           help=range_help("monthly_bill_egp"))
            tenure = st.number_input("Tenure (months)", min_value=0, value=24, step=1,
                                     help=range_help("tenure_months"))
            complaints = st.number_input("Complaints", min_value=0, value=1, step=1,
                                         help=range_help("complaint_count"))
            downtime = st.number_input("Monthly downtime (hours)", min_value=0.0, value=2.0, step=0.5,
                                       help=range_help("downtime_hours_monthly"))
        submitted = st.form_submit_button("Predict churn")

    if submitted:
        # The keys must match the API's Customer schema exactly.
        payload = {
            "governorate": governorate, "customer_type": customer_type,
            "subscription_plan": subscription_plan, "payment_method": payment_method,
            "monthly_bill_egp": monthly_bill, "tenure_months": tenure,
            "complaint_count": complaints, "downtime_hours_monthly": downtime,
        }
        try:
            response = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
        except requests.exceptions.RequestException:
            st.error("The API did not respond. Is it still running?")
        else:
            if response.status_code == 200:
                result = response.json()
                probability = result["churn_probability"]
                st.metric("Churn probability", f"{probability:.0%}")
                st.progress(min(max(probability, 0.0), 1.0))
                if result["will_churn"]:
                    st.error("Likely to churn")
                else:
                    st.success("Likely to stay")
                st.caption(f"A customer is flagged as likely to churn when the probability is "
                           f"at least {result['threshold']:.1%}.")
            else:
                st.error("The API rejected this input.")
                st.json(response.json())

# ---------------------------------------------------------------- Tab 2: model scores
with tab_performance:
    metrics = get_json("/metrics")
    test = metrics["test"]

    st.subheader(f"Final test results ({metrics['chosen_model']})")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("ROC-AUC", f"{test['roc_auc']:.3f}")
    c2.metric("PR-AUC", f"{test['pr_auc']:.3f}")
    c3.metric("Recall", f"{test['recall']:.3f}")
    c4.metric("F1", f"{test['f1']:.3f}")
    st.caption(f"Scores on customers the model never saw, using a threshold of {metrics['threshold']:.3f}. "
               f"Recall is the share of real churners the model catches.")

    (tn, fp), (fn, tp) = test["confusion_matrix"]
    confusion = pd.DataFrame([[tn, fp], [fn, tp]], index=["Actually stayed", "Actually churned"],
                             columns=["Predicted stay", "Predicted churn"])
    st.write("Confusion matrix (number of customers)")
    st.dataframe(confusion)

    st.subheader("Model comparison (cross-validation on the training set)")
    comparison = pd.DataFrame({name: {metric: scores["mean"] for metric, scores in values.items()}
                               for name, values in metrics["cv_comparison"].items()}).T
    comparison = comparison.rename(columns={"roc_auc": "ROC-AUC", "pr_auc": "PR-AUC", "recall": "Recall",
                                            "f1": "F1", "accuracy": "Accuracy"})
    st.dataframe(comparison.round(3))

st.divider()
st.caption("The training data is synthetic, so these predictions and scores do not describe real customers.")
