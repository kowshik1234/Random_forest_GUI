# ============================================================
# Credit Card Default Prediction using Random Forest
# ============================================================
# This project predicts whether a credit card customer will
# default on their next payment using a Random Forest model.
#
# Dataset: UCI Credit Card Default dataset (UCI_Credit_Card.csv)
# Algorithm: Random Forest Classifier (scikit-learn)
# UI Framework: Streamlit
# ============================================================

import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ============================================================
# PAGE CONFIGURATION
# ============================================================
# Set the browser tab title, icon, and use wide layout
st.set_page_config(
    page_title="Credit Card Default Predictor",
    page_icon="💳",
    layout="wide"
)


# ============================================================
# STEP 1: LOAD DATA AND TRAIN THE RANDOM FOREST MODEL
# ============================================================
# @st.cache_resource makes sure training only happens ONCE.
# After that, the trained model is reused on every interaction.
@st.cache_resource
def load_data_and_train_model():

    # --- Load the dataset from the same folder ---
    df = pd.read_csv("UCI_Credit_Card.csv")

    # --- Drop the ID column (just a row number, not useful) ---
    df = df.drop("ID", axis=1)

    # --- Separate features (X) and target variable (y) ---
    # X = all columns except the target
    # y = target column: 1 means default, 0 means no default
    X = df.drop("default.payment.next.month", axis=1)
    y = df["default.payment.next.month"]

    # --- Split into 80% training and 20% testing ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # =============================================
    # CREATE THE RANDOM FOREST CLASSIFIER
    # =============================================
    # n_estimators = number of decision trees in the forest
    # max_depth    = maximum depth of each tree (limits complexity)
    # random_state = seed so results are the same every run
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )

    # --- Train (fit) the Random Forest on training data ---
    rf_model.fit(X_train, y_train)

    # --- Evaluate: predict on test data and measure accuracy ---
    y_pred = rf_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # --- Get feature importance scores (built into Random Forest) ---
    feature_importance = pd.Series(
        rf_model.feature_importances_,
        index=X.columns
    ).sort_values(ascending=True)  # ascending for horizontal bar chart

    # Return everything we need
    return rf_model, accuracy, list(X.columns), feature_importance


# --- Train the model (cached — only runs on first load) ---
with st.spinner("Training the Random Forest model... please wait."):
    model, accuracy, feature_names, feature_importance = load_data_and_train_model()


# ============================================================
# STEP 2: SIDEBAR — MODEL INFORMATION
# ============================================================
st.sidebar.title("🌲 Model Info")
st.sidebar.markdown(f"**Algorithm:** Random Forest")
st.sidebar.markdown(f"**Number of Trees:** 100")
st.sidebar.markdown(f"**Max Tree Depth:** 10")
st.sidebar.markdown(f"**Test Accuracy:** {accuracy:.1%}")

# Show feature importance chart (great for viva explanation)
st.sidebar.markdown("---")
st.sidebar.subheader("📊 Feature Importance")
st.sidebar.caption("Which features matter most to the model:")
st.sidebar.bar_chart(feature_importance, horizontal=True, height=400)


# ============================================================
# STEP 3: PAGE HEADER
# ============================================================
st.title("💳 Credit Card Default Predictor")
st.markdown(
    "Enter customer information below and click **Predict** "
    "to check if they are likely to **default** on their credit card."
)
st.markdown("---")


# ============================================================
# STEP 4: DROPDOWN OPTIONS (human-readable labels)
# ============================================================
# These map friendly labels to the numeric codes used in the dataset

# Repayment status codes
pay_status_options = {
    "Paid in full": -1,
    "No consumption": -2,
    "Minimum payment (revolving credit)": 0,
    "1 month late": 1,
    "2 months late": 2,
    "3 months late": 3,
    "4 months late": 4,
    "5 months late": 5,
    "6 months late": 6,
    "7 months late": 7,
    "8 months late": 8,
    "9+ months late": 9,
}


# ============================================================
# STEP 5: USER INPUT FORM
# ============================================================

# ------ Section A: Personal Information ------
st.subheader("👤 Personal Information")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    # Credit limit in dollars
    limit_bal = st.number_input(
        "Credit Limit ($)", min_value=0, max_value=1_000_000,
        value=50_000, step=10_000
    )

with col2:
    # Gender selection
    sex = st.selectbox("Gender", ["Male", "Female"])

with col3:
    # Education level
    education = st.selectbox(
        "Education", ["Graduate School", "University", "High School", "Other"]
    )

with col4:
    # Marital status
    marriage = st.selectbox("Marital Status", ["Married", "Single", "Other"])

with col5:
    # Age in years
    age = st.number_input("Age", min_value=18, max_value=100, value=30)

# --- Convert personal info to numeric codes matching the dataset ---
sex_code = 1 if sex == "Male" else 2
education_code = {"Graduate School": 1, "University": 2, "High School": 3, "Other": 4}[education]
marriage_code = {"Married": 1, "Single": 2, "Other": 3}[marriage]


# ------ Section B: Repayment Status (6 months) ------
st.markdown("---")
st.subheader("📋 Repayment Status (Last 6 Months)")
st.caption("Select the payment status for each month, starting from the most recent.")

pay_cols = st.columns(6)
pay_labels = [
    "Month 1 (Latest)", "Month 2", "Month 3",
    "Month 4", "Month 5", "Month 6 (Oldest)"
]
pay_values = []

for i, col in enumerate(pay_cols):
    with col:
        # Each dropdown lets user pick a repayment status
        selected = st.selectbox(
            pay_labels[i], list(pay_status_options.keys()), key=f"pay_{i}"
        )
        pay_values.append(pay_status_options[selected])


# ------ Section C: Bill Statement Amounts (6 months) ------
st.markdown("---")
st.subheader("💰 Bill Statement Amounts (Last 6 Months)")
st.caption("Enter the bill amount for each month (can be negative if overpaid).")

bill_cols = st.columns(6)
bill_values = []

for i, col in enumerate(bill_cols):
    with col:
        val = st.number_input(
            f"Month {i+1} ($)", min_value=-500_000, max_value=1_000_000,
            value=5_000, step=1_000, key=f"bill_{i}"
        )
        bill_values.append(val)


# ------ Section D: Payment Amounts (6 months) ------
st.markdown("---")
st.subheader("💸 Payment Amounts (Last 6 Months)")
st.caption("Enter how much was paid each month.")

payamt_cols = st.columns(6)
payamt_values = []

for i, col in enumerate(payamt_cols):
    with col:
        val = st.number_input(
            f"Month {i+1} ($)", min_value=0, max_value=1_000_000,
            value=2_000, step=500, key=f"payamt_{i}"
        )
        payamt_values.append(val)


# ============================================================
# STEP 6: COMBINE ALL USER INPUTS INTO ONE ROW
# ============================================================
st.markdown("---")

# Build one row of data in the EXACT same column order as the dataset:
# LIMIT_BAL, SEX, EDUCATION, MARRIAGE, AGE,
# PAY_0, PAY_2, PAY_3, PAY_4, PAY_5, PAY_6,
# BILL_AMT1, BILL_AMT2, BILL_AMT3, BILL_AMT4, BILL_AMT5, BILL_AMT6,
# PAY_AMT1, PAY_AMT2, PAY_AMT3, PAY_AMT4, PAY_AMT5, PAY_AMT6
user_input = [limit_bal, sex_code, education_code, marriage_code, age]
user_input += pay_values       # 6 repayment status values
user_input += bill_values      # 6 bill statement amounts
user_input += payamt_values    # 6 payment amounts

# Convert to a DataFrame with correct column names so the model can use it
input_df = pd.DataFrame([user_input], columns=feature_names)


# ============================================================
# STEP 7: PREDICT USING THE RANDOM FOREST MODEL
# ============================================================

# Center the prediction button
_, center_col, _ = st.columns([1, 2, 1])
with center_col:
    predict_clicked = st.button(
        "🔮 Predict Default", use_container_width=True, type="primary"
    )

# --- When the button is clicked, make a prediction ---
if predict_clicked:

    # Use the Random Forest model to predict (returns 0 or 1)
    prediction = model.predict(input_df)[0]

    # Get probability scores from Random Forest
    # predict_proba returns [probability_no_default, probability_default]
    probabilities = model.predict_proba(input_df)[0]

    st.markdown("---")

    # --- Display the result ---
    if prediction == 1:
        # Customer IS predicted to default
        st.error(f"⚠️ **Likely to Default**")
        st.markdown(f"The model predicts this customer **will default** "
                    f"with **{probabilities[1] * 100:.1f}%** confidence.")
    else:
        # Customer is NOT predicted to default
        st.success(f"✅ **Not Likely to Default**")
        st.markdown(f"The model predicts this customer **will not default** "
                    f"with **{probabilities[0] * 100:.1f}%** confidence.")

    # Show the probability breakdown side by side
    result_col1, result_col2 = st.columns(2)
    with result_col1:
        st.metric("No Default Probability", f"{probabilities[0] * 100:.1f}%")
    with result_col2:
        st.metric("Default Probability", f"{probabilities[1] * 100:.1f}%")
