import streamlit as st, pandas as pd, joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="Churn Risk Dashboard", layout="wide")
model = joblib.load("model.joblib")

def build(d):
    d = d.copy()
    d["BalanceSalaryRatio"] = d["Balance"] / (d["EstimatedSalary"] + 1)
    d["ProductDensity"] = d["NumOfProducts"] / (d["Tenure"] + 1)
    d["EngagementProduct"] = d["IsActiveMember"] * d["NumOfProducts"]
    d["AgeTenure"] = d["Age"] * d["Tenure"]
    return d

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

base = pd.DataFrame([{"CreditScore": credit, "Geography": geo, "Gender": gender,
    "Age": age, "Tenure": tenure, "Balance": balance, "NumOfProducts": products,
    "HasCrCard": card, "IsActiveMember": active, "EstimatedSalary": salary}])

def predict(d): return model.predict_proba(build(d))[0, 1]

with tab1:
    p = predict(base)
    st.metric("Churn Probability", f"{p:.1%}")
    st.write("Risk level:", "🔴 High" if p > 0.5 else "🟡 Medium" if p > 0.25 else "🟢 Low")
    st.progress(float(p))

with tab2:
    st.write("Change engagement/product values and watch the probability move.")
    new_prod = st.slider("Products (scenario)", 1, 4, products, key="np")
    new_act = st.selectbox("Active (scenario)", [1, 0], index=[1, 0].index(active), key="na")
    scen = base.copy(); scen["NumOfProducts"] = new_prod; scen["IsActiveMember"] = new_act
    p2 = predict(scen)
    c1, c2 = st.columns(2)
    c1.metric("Current", f"{p:.1%}")
    c2.metric("Scenario", f"{p2:.1%}", f"{(p2-p):+.1%}", delta_color="inverse")

with tab3:
    names = model.named_steps["prep"].get_feature_names_out()
    imp = pd.Series(model.named_steps["model"].feature_importances_, index=names).sort_values().tail(12)
    fig, ax = plt.subplots(); imp.plot(kind="barh", ax=ax); st.pyplot(fig)