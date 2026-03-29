import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Safe XGBoost
try:
    from xgboost import XGBClassifier
    xgb_available = True
except:
    xgb_available = False

# -----------------------
# PAGE CONFIG
# -----------------------
st.set_page_config(page_title="Customer Churn ML Dashboard", layout="wide")

# -----------------------
# LOAD CSS
# -----------------------
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# -----------------------
# HEADER
# -----------------------
st.markdown('<div class="title">📊 Customer Churn ML Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Advanced Analytics & Prediction System</div>', unsafe_allow_html=True)

st.markdown("---")

# -----------------------
# FILE UPLOAD
# -----------------------
file = st.sidebar.file_uploader("Upload CSV Dataset", type=["csv"])

if file is not None:

    df = pd.read_csv(file)
    df = df.dropna()

    if 'customerID' in df.columns:
        df = df.drop('customerID', axis=1)

    if 'Churn' not in df.columns:
        st.error("Dataset must contain 'Churn' column")
        st.stop()

    # -----------------------
    # ORIGINAL + ENCODED
    # -----------------------
    df_original = df.copy()
    df_encoded = df.copy()

    le_dict = {}

    for col in df_encoded.columns:
        if df_encoded[col].dtype == 'object':
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col])
            le_dict[col] = le

    # -----------------------
    # SPLIT
    # -----------------------
    X = df_encoded.drop('Churn', axis=1)
    y = df_encoded['Churn']
    feature_names = X.columns

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # -----------------------
    # MODELS
    # -----------------------
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier()
    }

    if xgb_available:
        models["XGBoost"] = XGBClassifier(eval_metric='logloss')

    results = []
    trained_models = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        results.append([name, acc, prec, rec, f1])
        trained_models[name] = model

    results_df = pd.DataFrame(results, columns=["Model","Accuracy","Precision","Recall","F1 Score"])

    best_model_name = results_df.sort_values(by="F1 Score", ascending=False).iloc[0]["Model"]
    best_model = trained_models[best_model_name]

    # -----------------------
    # TABS
    # -----------------------
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Data", "🤖 Models", "📈 Evaluation", "🔮 Prediction"])

    # -----------------------
    # TAB 1: DATA
    # -----------------------
    with tab1:
        st.subheader("📊 Original Dataset")
        st.dataframe(df_original.head())

        st.markdown("---")

        st.subheader("🔢 Encoded Dataset (Used for ML)")
        st.dataframe(df_encoded.head())

    # -----------------------
    # TAB 2: MODELS
    # -----------------------
    with tab2:
        st.subheader("Model Performance")
        st.dataframe(results_df)

        fig, ax = plt.subplots()
        results_df.set_index("Model").plot(kind='bar', ax=ax)
        st.pyplot(fig)

        st.success(f"🏆 Best Model: {best_model_name}")

    # -----------------------
    # TAB 3: EVALUATION
    # -----------------------
    with tab3:
        model_choice = st.selectbox("Select Model", list(trained_models.keys()))
        model_eval = trained_models[model_choice]

        y_pred = model_eval.predict(X_test)
        y_prob = model_eval.predict_proba(X_test)[:,1]

        st.write("### Confusion Matrix")
        st.write(confusion_matrix(y_test, y_pred))

        st.write("### ROC Curve")
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)

        fig2, ax2 = plt.subplots()
        ax2.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
        ax2.plot([0,1],[0,1],'--')
        ax2.legend()
        st.pyplot(fig2)

        if hasattr(model_eval, "feature_importances_"):
            st.write("### Feature Importance")
            importances = model_eval.feature_importances_
            feat_imp = pd.Series(importances, index=feature_names).sort_values()

            fig3, ax3 = plt.subplots()
            feat_imp.plot(kind='barh', ax=ax3)
            st.pyplot(fig3)

    # -----------------------
    # TAB 4: PREDICTION
    # -----------------------
    with tab4:

        model_choice = st.selectbox(
            "Choose Model",
            ["Best Model"] + list(trained_models.keys())
        )

        selected_model = best_model if model_choice == "Best Model" else trained_models[model_choice]

        input_data = []

        for col in feature_names:
            if col in le_dict:
                options = list(le_dict[col].classes_)
                val = st.selectbox(col, options)
                val = le_dict[col].transform([val])[0]
            else:
                val = st.number_input(col, value=0.0)

            input_data.append(val)

        input_array = scaler.transform([input_data])

        if st.button("🚀 Predict"):
            pred = selected_model.predict(input_array)[0]
            prob = selected_model.predict_proba(input_array)[0][1]

            if pred == 1:
                st.error(f"⚠️ Customer will churn (Prob: {prob:.2f})")
            else:
                st.success(f"✅ Customer will stay (Prob: {prob:.2f})")

else:
    st.info("⬅️ Upload dataset to start")