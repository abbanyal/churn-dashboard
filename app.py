import streamlit as st
import pandas as pd
import joblib
import numpy as np
from pathlib import Path

st.set_page_config(page_title="Churn Risk Dashboard", layout="wide")

BASE = Path(__file__).parent


@st.cache_resource
def load_model():
    return joblib.load(BASE / "model.joblib")


model = load_model()


def build(d):
    d = d.copy()
    d["BalanceSalaryRatio"] = d["Balance"] / (d["EstimatedSalary"] + 1)
    d["ProductDensity"] = d["NumOfProducts"] / (d["Tenure"] + 1)
    d["EngagementProduct"] = d["IsActiveMember"] * d["NumOfProducts"]
    d["AgeTenure"] = d["Age"] * d["Tenure"]
    return d


def predict(d):
    return float(model.predict_proba(build(d))[0, 1])


def risk_label(p):
    return "🔴 High" if p > 0.5 else "🟡 Medium" if p > 0.25 else "🟢 Low"


st.title("Customer Churn Risk Dashboard")
tab1, tab2, tab3 = st.tabs(["Risk Calculator", "What-If Simulator", "Model Insights"])

with st.sidebar:
    st.header("Customer Profile")
    geo = st.selectbox("Geography", ["France", "Spain", "Germany"])
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.slider("Age", 18, 92, 40)
    credit = st.slider("Credit Score", 300, 850, 650)
    tenure = st.slider("Tenure (years)", 0, 10, 5)
    balance = st.number_input("Balance", 0.0, 300000.0, 50000.0)
    products = st.slider("Number of Products", 1, 4, 2)
    card = st.selectbox("Has Credit Card", [1, 0])
    active = st.selectbox("Active Member", [1, 0])
    salary = st.number_input("Estimated Salary", 0.0, 250000.0, 100000.0)

base = pd.DataFrame([{
    "CreditScore": credit, "Geography": geo, "Gender": gender, "Age": age,
    "Tenure": tenure, "Balance": balance, "NumOfProducts": products,
    "HasCrCard": card, "IsActiveMember": active, "EstimatedSalary": salary}])
p = predict(base)

with tab1:
    st.subheader("Churn Risk Calculator")
    st.metric("Churn Probability", f"{p:.1%}")
    st.write("Risk level:", risk_label(p))
    st.progress(min(max(p, 0.0), 1.0))

with tab2:
    st.subheader("What-If Scenario Simulator")
    st.write("Change engagement/product values and watch the probability move.")
    new_prod = st.slider("Products (scenario)", 1, 4, products, key="np")
    new_act = st.selectbox("Active (scenario)", [1, 0], index=[1, 0].index(active), key="na")
    scen = base.copy()
    scen["NumOfProducts"] = new_prod
    scen["IsActiveMember"] = new_act
    p2 = predict(scen)
    c1, c2 = st.columns(2)
    c1.metric("Current", f"{p:.1%}")
    c2.metric("Scenario", f"{p2:.1%}", f"{(p2 - p):+.1%}", delta_color="inverse")

with tab3:
        st.subheader("How Age Changes This Customer's Risk")
    try:
        ages = list(range(18, 93, 2))
        curve = pd.concat([base.assign(Age=a) for a in ages], ignore_index=True)
        probs = model.predict_proba(build(curve))[:, 1]
        st.line_chart(pd.DataFrame({"Age": ages, "Churn probability": probs}),
                      x="Age", y="Churn probability")
    except Exception as e:
        st.error(f"Could not draw the age curve: {e}")
    st.subheader("Feature Importance")
    try:
        names = model.named_steps["prep"].get_feature_names_out()
        names = [n.split("__", 1)[-1] for n in names]
        imp = pd.Series(model.named_steps["model"].feature_importances_, index=names)
        imp = imp.sort_values(ascending=False).head(12)
        imp_df = imp.rename("Importance").rename_axis("Feature").reset_index()
        st.bar_chart(imp_df, x="Feature", y="Importance", horizontal=True, sort="-Importance")
    except Exception as e:
        st.error(f"Could not draw feature importance: {e}")

    st.subheader("Distribution of Predicted Churn Probabilities")
    try:
        tp = pd.read_csv(BASE / "test_probs.csv")
        edges = np.linspace(0, 1, 21)
        labels = [f"{edges[i]:.2f}-{edges[i + 1]:.2f}" for i in range(20)]
        stayed, _ = np.histogram(tp[tp["actual"] == 0]["prob"], bins=edges)
        churned, _ = np.histogram(tp[tp["actual"] == 1]["prob"], bins=edges)
        hist = pd.DataFrame({"Stayed": stayed, "Churned": churned}, index=labels)
        st.bar_chart(hist, x_label="Predicted churn probability", y_label="Customers")
    except Exception as e:
        st.error(f"Could not draw probability distribution: {e}")
