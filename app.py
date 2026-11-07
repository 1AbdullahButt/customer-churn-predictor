import pandas as pd
import streamlit as st
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📉", layout="wide")

DATA_FILE = Path(__file__).parent / "telco_churn.csv"
NUMBER_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]


# ---------- 1. Load data and train the model (runs once, then is remembered) ----------
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_FILE)
    # Same cleaning as the notebook
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
    df = df.drop(columns="customerID")
    return df


@st.cache_resource
def train_model():
    df = load_data()
    X = df.drop(columns="Churn")
    y = df["Churn"].map({"No": 0, "Yes": 1})

    category_cols = [c for c in X.columns if c not in NUMBER_COLS]
    preprocess = ColumnTransformer([
        ("numbers", StandardScaler(), NUMBER_COLS),
        ("categories", OneHotEncoder(handle_unknown="ignore"), category_cols),
    ])
    model = Pipeline([
        ("prep", preprocess),
        ("model", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])

    # Same 80/20 split as the notebook, so the app uses the same training data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    model.fit(X_train, y_train)
    return model, X.columns.tolist()


df = load_data()
model, feature_order = train_model()


# ---------- 2. Page title ----------
st.title("📉 Customer Churn Predictor")
st.write("Enter a customer's details on the left. The app shows how likely they are to leave "
         "and suggests what to do to keep them.")

# ---------- 3. Inputs (sidebar) ----------
st.sidebar.header("Customer details")

tenure = st.sidebar.slider("Months with the company (tenure)", 0, 72, 6)
monthly = st.sidebar.slider("Monthly bill ($)", 18.0, 120.0, 75.0, step=0.5)
contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
payment = st.sidebar.selectbox("Payment method", [
    "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
internet = st.sidebar.selectbox("Internet service", ["Fiber optic", "DSL", "No"])

if internet == "No":
    # Customers with no internet cannot have internet add-ons
    tech_support = online_security = streaming_tv = streaming_movies = "No internet service"
    online_backup = device_protection = "No internet service"
    st.sidebar.caption("No internet: add-on services are set to 'No internet service'.")
else:
    tech_support = st.sidebar.selectbox("Tech support", ["No", "Yes"])
    online_security = st.sidebar.selectbox("Online security", ["No", "Yes"])
    streaming_tv = st.sidebar.selectbox("Streaming TV", ["No", "Yes"])
    streaming_movies = st.sidebar.selectbox("Streaming movies", ["No", "Yes"])
    online_backup = st.sidebar.selectbox("Online backup", ["No", "Yes"])
    device_protection = st.sidebar.selectbox("Device protection", ["No", "Yes"])

with st.sidebar.expander("More details"):
    gender = st.selectbox("Gender", ["Female", "Male"])
    senior = st.selectbox("Senior citizen", ["No", "Yes"])
    partner = st.selectbox("Has a partner", ["No", "Yes"])
    dependents = st.selectbox("Has dependents", ["No", "Yes"])
    phone = st.selectbox("Phone service", ["Yes", "No"])
    if phone == "Yes":
        multiple_lines = st.selectbox("Multiple lines", ["No", "Yes"])
    else:
        multiple_lines = "No phone service"
    paperless = st.selectbox("Paperless billing", ["Yes", "No"])

# ---------- 4. Predict ----------
customer = pd.DataFrame([{
    "gender": gender,
    "SeniorCitizen": 1 if senior == "Yes" else 0,
    "Partner": partner,
    "Dependents": dependents,
    "tenure": tenure,
    "PhoneService": phone,
    "MultipleLines": multiple_lines,
    "InternetService": internet,
    "OnlineSecurity": online_security,
    "OnlineBackup": online_backup,
    "DeviceProtection": device_protection,
    "TechSupport": tech_support,
    "StreamingTV": streaming_tv,
    "StreamingMovies": streaming_movies,
    "Contract": contract,
    "PaperlessBilling": paperless,
    "PaymentMethod": payment,
    "MonthlyCharges": monthly,
    "TotalCharges": tenure * monthly,   # estimate: months x monthly bill
}])[feature_order]

probability = model.predict_proba(customer)[0, 1]

# Same cut-offs as the notebook
if probability >= 0.60:
    risk, icon = "High", "🔴"
elif probability >= 0.30:
    risk, icon = "Medium", "🟠"
else:
    risk, icon = "Low", "🟢"

# ---------- 5. Show the result ----------
left, right = st.columns([1, 1])

with left:
    st.subheader("Churn risk")
    st.metric("Risk level", f"{icon} {risk}")
    st.metric("Chance this customer leaves", f"{probability:.0%}")
    st.progress(float(probability))

    st.subheader("Why?")
    reasons = []
    if contract == "Month-to-month":
        reasons.append("Month-to-month contract (43% of these customers churn, vs 3% on two-year)")
    if tenure <= 12:
        reasons.append("In their first year (47% of first-year customers churn)")
    if internet == "Fiber optic":
        reasons.append("Fiber optic internet (42% churn, vs 19% on DSL)")
    if payment == "Electronic check":
        reasons.append("Pays by electronic check (45% churn)")
    if tech_support == "No" and internet != "No":
        reasons.append("No tech support (42% churn, vs 15% with it)")
    if online_security == "No" and internet != "No":
        reasons.append("No online security (42% churn, vs 15% with it)")
    if monthly >= 80:
        reasons.append("High monthly bill (churners pay about $74 on average, others $61)")

    if reasons:
        for r in reasons:
            st.write("• " + r)
    else:
        st.write("• No common warning signs. This customer looks stable.")

with right:
    st.subheader("Suggested actions")
    actions = []
    if contract == "Month-to-month":
        actions.append("Offer a discount or free add-on to move to a 1 or 2-year contract.")
    if tenure <= 12:
        actions.append("Schedule a friendly check-in call during the first year.")
    if payment == "Electronic check":
        actions.append("Offer a small bill credit to switch to automatic payment.")
    if internet != "No" and tech_support == "No":
        actions.append("Offer a free trial of tech support.")
    if internet != "No" and online_security == "No":
        actions.append("Offer a free trial of online security.")
    if internet == "Fiber optic":
        actions.append("Check the customer's service quality and price for fiber.")

    if risk == "High":
        st.error("High risk: contact this customer this week.")
    elif risk == "Medium":
        st.warning("Medium risk: add them to the next retention campaign.")
    else:
        st.success("Low risk: no action needed. Keep the service good.")

    if actions:
        for a in actions:
            st.write("✅ " + a)
    else:
        st.write("✅ Keep doing what you are doing.")

    # Chart: churn rate by contract, from the real data
    st.subheader("Churn rate by contract (all customers)")
    chart_data = df.groupby("Contract")["Churn"].apply(lambda s: (s == "Yes").mean() * 100)
    st.bar_chart(chart_data.round(1), y_label="Churn rate (%)")
    st.caption(f"This customer is on: {contract}")

# ---------- 6. About ----------
with st.expander("About this model"):
    st.write(
        "**Model:** Logistic Regression with `class_weight='balanced'`, trained on the Telco "
        "Customer Churn dataset (7,043 customers).\n\n"
        "**Result on unseen test data:** it catches about 78% of customers who really leave "
        "(recall). About half of the customers it flags will not leave, which is an acceptable "
        "cost because a wasted retention call is cheaper than a lost customer.\n\n"
        "**Note:** Total charges are estimated as months x monthly bill. "
        "The patterns are links in the data, not proven causes.")
