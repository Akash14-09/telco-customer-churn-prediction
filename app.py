import os
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve
)
from xgboost import XGBClassifier

st.set_page_config(
    page_title="Telco Customer Churn Prediction",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_data(show_spinner="Loading dataset...")
def load_dataset():
    local_file = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    if os.path.exists(local_file):
        df = pd.read_csv(local_file)
    else:
        try:
            import kagglehub
            path = kagglehub.dataset_download("blastchar/telco-customer-churn")
            csv_path = os.path.join(path, "WA_Fn-UseC_-Telco-Customer-Churn.csv")
            df = pd.read_csv(csv_path)
        except Exception:
            fallback_url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
            df = pd.read_csv(fallback_url)

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.dropna(subset=["TotalCharges"], inplace=True)
    if "customerID" in df.columns:
        df.drop(columns=["customerID"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def preprocess_features(df):
    data = df.copy()
    data["Churn"] = data["Churn"].map({"Yes": 1, "No": 0})

    # Feature Engineering: Average monthly spend ratio and tenure cohorts
    data["Charge_to_Tenure_Ratio"] = data["TotalCharges"] / (data["tenure"] + 1)

    categorical_cols = data.select_dtypes(include=["object"]).columns.tolist()
    data_encoded = pd.get_dummies(data, columns=categorical_cols, drop_first=True)

    X = data_encoded.drop(columns=["Churn"])
    y = data_encoded["Churn"]
    return X, y


@st.cache_resource(show_spinner="Training and evaluating models...")
def train_models(df):
    X, y = preprocess_features(df)
    feature_columns = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    num_cols = ["tenure", "MonthlyCharges", "TotalCharges", "Charge_to_Tenure_Ratio"]

    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test_scaled[num_cols] = scaler.transform(X_test[num_cols])

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, C=0.5, random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=10, min_samples_split=5, random_state=42
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=150, learning_rate=0.05, max_depth=4, random_state=42
        ),
        "XGBoost": XGBClassifier(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42
        ),
        "Artificial Neural Network (MLP)": MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            alpha=0.01,
            early_stopping=True,
            max_iter=300,
            random_state=42
        )
    }

    results = []
    trained_models = {}
    roc_data = {}
    conf_matrices = {}

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]

        cv_auc = cross_val_score(
            model, X_train_scaled, y_train, cv=cv, scoring="roc_auc"
        ).mean()

        results.append({
            "Model": name,
            "Accuracy": round(accuracy_score(y_test, y_pred), 4),
            "Precision": round(precision_score(y_test, y_pred), 4),
            "Recall": round(recall_score(y_test, y_pred), 4),
            "F1-Score": round(f1_score(y_test, y_pred), 4),
            "Test ROC-AUC": round(roc_auc_score(y_test, y_prob), 4),
            "5-Fold CV AUC": round(cv_auc, 4)
        })

        trained_models[name] = model
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_data[name] = (fpr, tpr)
        conf_matrices[name] = confusion_matrix(y_test, y_pred)

    results_df = pd.DataFrame(results).sort_values(by="Test ROC-AUC", ascending=False)
    return trained_models, scaler, feature_columns, num_cols, results_df, roc_data, conf_matrices


df = load_dataset()
trained_models, scaler, feature_columns, num_cols, results_df, roc_data, conf_matrices = train_models(df)

st.title("Customer Churn Prediction and Retention Analytics")
st.markdown(
    "Supervised machine learning pipeline comparing linear, ensemble, and neural network architectures "
    "on the Telco Customer Churn dataset."
)

tab_pred, tab_eval, tab_eda = st.tabs([
    "Churn Risk Predictor",
    "Model Evaluation & Comparison",
    "Exploratory Data Analysis"
])

with tab_pred:
    st.subheader("Single Customer Risk Assessment")

    col_input1, col_input2, col_input3 = st.columns(3)

    with col_input1:
        selected_model_name = st.selectbox("Select Model Architecture", list(trained_models.keys()), index=3)
        tenure = st.slider("Tenure (Months)", min_value=0, max_value=72, value=12)
        monthly_charges = st.number_input("Monthly Charges ($)", min_value=18.0, max_value=120.0, value=70.0, step=1.0)
        total_charges = st.number_input(
            "Total Charges ($)",
            min_value=0.0,
            max_value=9000.0,
            value=float(tenure * monthly_charges),
            step=10.0
        )
        senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])

    with col_input2:
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        internet_service = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
        payment_method = st.selectbox(
            "Payment Method",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
        )
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])

    with col_input3:
        online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
        partner = st.selectbox("Has Partner", ["No", "Yes"])
        dependents = st.selectbox("Has Dependents", ["No", "Yes"])

    if st.button("Predict Churn Probability"):
        input_dict = {
            "gender": "Male",
            "SeniorCitizen": 1 if senior_citizen == "Yes" else 0,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": "Yes",
            "MultipleLines": "No",
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": "No",
            "TechSupport": tech_support,
            "StreamingTV": "No",
            "StreamingMovies": "No",
            "Contract": contract,
            "PaperlessBilling": paperless_billing,
            "PaymentMethod": payment_method,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
            "Charge_to_Tenure_Ratio": total_charges / (tenure + 1)
        }

        input_df = pd.DataFrame([input_dict])
        input_encoded = pd.get_dummies(input_df)
        input_encoded = input_encoded.reindex(columns=feature_columns, fill_value=0)
        input_encoded[num_cols] = scaler.transform(input_encoded[num_cols])

        model = trained_models[selected_model_name]
        prob = model.predict_proba(input_encoded)[0][1]
        pred_label = "High Churn Risk" if prob >= 0.5 else "Low Churn Risk"

        st.markdown("---")
        res_col1, res_col2 = st.columns([1, 2])
        with res_col1:
            st.metric(label="Predicted Churn Probability", value=f"{prob * 100:.2f}%")
            if prob >= 0.5:
                st.error(f"Classification: {pred_label}")
            else:
                st.success(f"Classification: {pred_label}")

        with res_col2:
            st.markdown("**Key Retention Insights:**")
            drivers = []
            if contract == "Month-to-month":
                drivers.append("Month-to-month contracts exhibit the highest historical attrition rate.")
            if internet_service == "Fiber optic" and online_security == "No":
                drivers.append("Fiber optic users without Online Security or Tech Support show elevated churn risk.")
            if payment_method == "Electronic check":
                drivers.append("Electronic check billing correlates with higher churn compared to automatic transfers.")
            if tenure < 12:
                drivers.append("Customer is in the early onboarding phase (< 12 months tenure).")
            if not drivers:
                drivers.append("Customer profile shows strong retention indicators across contract and service tiers.")
            for d in drivers:
                st.write(f"- {d}")

with tab_eval:
    st.subheader("Cross-Validated Performance Metrics")
    st.dataframe(results_df, use_container_width=True, hide_index=True)

    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        st.markdown("**ROC Curves Across Models**")
        fig_roc, ax_roc = plt.subplots(figsize=(6, 4.5))
        for name, (fpr, tpr) in roc_data.items():
            ax_roc.plot(fpr, tpr, label=name)
        ax_roc.plot([0, 1], [0, 1], "k--", lw=1)
        ax_roc.set_xlabel("False Positive Rate")
        ax_roc.set_ylabel("True Positive Rate")
        ax_roc.legend(loc="lower right", fontsize=8)
        ax_roc.grid(alpha=0.3)
        st.pyplot(fig_roc)

    with col_chart2:
        cm_model = st.selectbox("Select Model for Confusion Matrix", list(conf_matrices.keys()), index=3)
        fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
        sns.heatmap(
            conf_matrices[cm_model],
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Retained (0)", "Churned (1)"],
            yticklabels=["Retained (0)", "Churned (1)"],
            ax=ax_cm
        )
        ax_cm.set_xlabel("Predicted Label")
        ax_cm.set_ylabel("Actual Label")
        st.pyplot(fig_cm)

with tab_eda:
    st.subheader("Dataset Overview & Feature Distributions")
    m1, m2, m3 = st.columns(3)
    m1.metric("Total Customer Records", f"{len(df):,}")
    m2.metric("Overall Churn Rate", f"{(df['Churn'] == 'Yes').mean() * 100:.2f}%")
    m3.metric("Median Monthly Charge", f"${df['MonthlyCharges'].median():.2f}")

    eda_col1, eda_col2 = st.columns(2)
    with eda_col1:
        st.markdown("**Churn Distribution by Contract Type**")
        fig_c, ax_c = plt.subplots(figsize=(6, 4))
        sns.countplot(data=df, x="Contract", hue="Churn", palette="Blues", ax=ax_c)
        ax_c.set_ylabel("Customer Count")
        st.pyplot(fig_c)

    with eda_col2:
        st.markdown("**Tenure Distribution by Churn Status**")
        fig_t, ax_t = plt.subplots(figsize=(6, 4))
        sns.kdeplot(data=df, x="tenure", hue="Churn", fill=True, common_norm=False, palette="crest", ax=ax_t)
        ax_t.set_xlabel("Tenure (Months)")
        st.pyplot(fig_t)
