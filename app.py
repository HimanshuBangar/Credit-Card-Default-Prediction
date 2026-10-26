import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)
from sklearn.inspection import permutation_importance

st.set_page_config(
    page_title="Credit Risk Intelligence",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

DATA_URL = "https://github.com/ybifoundation/Dataset/raw/main/Credit%20Default.csv"
FEATURES = ["Income", "Age", "Loan", "Loan to Income"]
TARGET = "Default"

# ---------- Styling ----------
st.markdown("""
<style>
    .block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, rgba(28,45,75,.12), rgba(24,160,150,.08));
        border: 1px solid rgba(120,140,160,.25);
        padding: 16px 18px; border-radius: 14px;
    }
    .hero {
        padding: 1.35rem 1.5rem; border-radius: 18px;
        background: linear-gradient(120deg, #101d36 0%, #173d5b 60%, #087e8b 100%);
        color: white; margin-bottom: 1rem;
    }
    .hero h1 {color:white; margin-bottom:.25rem;}
    .hero p {color:#e2edf7; margin-bottom:0;}
    .small-note {color: #6b7280; font-size: .88rem;}
</style>
""", unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def load_data():
    try:
        df = pd.read_csv(DATA_URL)
        return df, "public dataset"
    except Exception:
        return None, "unavailable"

@st.cache_resource(show_spinner=False)
def train_models(df):
    clean = df.copy()
    clean.columns = [str(c).strip() for c in clean.columns]
    missing = [c for c in FEATURES + [TARGET] if c not in clean.columns]
    if missing:
        raise ValueError("Dataset is missing required columns: " + ", ".join(missing))

    for col in FEATURES + [TARGET]:
        clean[col] = pd.to_numeric(clean[col], errors="coerce")
    clean = clean[FEATURES + [TARGET]].dropna()
    clean = clean[clean[TARGET].isin([0, 1])]
    if len(clean) < 30 or clean[TARGET].nunique() != 2:
        raise ValueError("Need at least 30 valid rows and both target classes (0 and 1).")

    X = clean[FEATURES]
    y = clean[TARGET].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )

    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42))
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=250, max_depth=10, min_samples_leaf=3,
            class_weight="balanced", random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": HistGradientBoostingClassifier(
            max_iter=150, learning_rate=0.08, max_leaf_nodes=15, random_state=42
        ),
    }

    results = {}
    rows = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
        auc = roc_auc_score(y_test, prob) if prob is not None else np.nan
        metrics = {
            "Model": name,
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred, zero_division=0),
            "Recall": recall_score(y_test, pred, zero_division=0),
            "F1-score": f1_score(y_test, pred, zero_division=0),
            "ROC-AUC": auc
        }
        rows.append(metrics)
        results[name] = {
            "model": model, "pred": pred, "prob": prob,
            "metrics": metrics
        }

    score_table = pd.DataFrame(rows).sort_values("F1-score", ascending=False).reset_index(drop=True)
    best_name = str(score_table.iloc[0]["Model"])
    return clean, X_train, X_test, y_train, y_test, results, score_table, best_name

def risk_band(prob):
    if prob >= 0.65:
        return "High"
    if prob >= 0.30:
        return "Medium"
    return "Low"

def hero(title, subtitle):
    st.markdown(
        f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p></div>',
        unsafe_allow_html=True
    )

# ---------- Sidebar ----------
st.sidebar.markdown("## 💳 Credit Risk AI")
st.sidebar.caption("Machine Learning Decision-Support Demo")
page = st.sidebar.radio(
    "Navigation",
    ["Overview", "Predict Customer", "Batch Analysis", "Model Lab", "About Project"]
)
st.sidebar.divider()
st.sidebar.caption("Model outputs are estimates for educational demonstration, not financial decisions.")

uploaded_dataset = st.sidebar.file_uploader(
    "Optional: upload a compatible training CSV",
    type=["csv"],
    help="Must contain Income, Age, Loan, Loan to Income, and Default columns."
)

if uploaded_dataset is not None:
    try:
        source_df = pd.read_csv(uploaded_dataset)
        source_label = "uploaded CSV"
    except Exception as exc:
        st.error(f"Could not read uploaded dataset: {exc}")
        st.stop()
else:
    source_df, source_label = load_data()

if source_df is None:
    st.error("Could not load the public dataset. Check your internet connection or upload a compatible CSV using the sidebar.")
    st.info("Required columns: Income, Age, Loan, Loan to Income, Default.")
    st.stop()

try:
    with st.spinner("Preparing data and training models..."):
        clean_df, X_train, X_test, y_train, y_test, results, score_table, best_name = train_models(source_df)
except Exception as exc:
    st.error(f"Unable to prepare the dataset: {exc}")
    st.stop()

best = results[best_name]
best_model = best["model"]
best_pred = best["pred"]
best_prob = best["prob"]

# ---------- Pages ----------
if page == "Overview":
    hero("Credit Risk Intelligence", "AI-powered credit default prediction and model analytics")
    st.write("Analyze customer financial attributes, compare machine-learning models, and explore estimated default risk.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Records analyzed", f"{len(clean_df):,}")
    c2.metric("Default cases", f"{int(clean_df[TARGET].sum()):,}")
    c3.metric("Default rate", f"{clean_df[TARGET].mean():.1%}")
    c4.metric("Best F1 model", best_name)

    left, right = st.columns([1.15, 1])
    with left:
        st.subheader("Default class distribution")
        counts = clean_df[TARGET].value_counts().reindex([0, 1], fill_value=0).rename(index={0: "No Default", 1: "Default"})
        fig = px.pie(
            values=counts.values, names=counts.index, hole=0.62,
            color=counts.index,
            color_discrete_map={"No Default": "#20A39E", "Default": "#F46036"}
        )
        fig.update_layout(margin=dict(t=10, b=10, l=5, r=5), legend_title_text="")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.subheader("Model comparison")
        view = score_table.melt(id_vars="Model", value_vars=["Precision", "Recall", "F1-score", "ROC-AUC"], var_name="Metric", value_name="Score")
        fig = px.bar(view, x="Metric", y="Score", color="Model", barmode="group", range_y=[0, 1])
        fig.update_layout(margin=dict(t=10, b=10, l=5, r=5), legend_title_text="Model")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Dataset preview")
    st.dataframe(clean_df.head(10), use_container_width=True, hide_index=True)
    st.caption(f"Data source: {source_label}. Train/test split: 70% / 30%, stratified, random_state=42.")

elif page == "Predict Customer":
    hero("Customer Risk Assessment", "Enter a customer's financial profile to estimate default probability")
    st.warning("This is a demonstration model trained on a small sample dataset. Do not use it to approve or reject real credit applications.")
    with st.form("prediction_form"):
        a, b = st.columns(2)
        with a:
            income = st.number_input("Annual income", min_value=0.0, value=45000.0, step=1000.0)
            age = st.number_input("Age", min_value=18.0, max_value=100.0, value=35.0, step=1.0)
        with b:
            loan = st.number_input("Loan amount", min_value=0.0, value=5000.0, step=250.0)
            ratio = st.number_input("Loan-to-income ratio", min_value=0.0, max_value=10.0, value=(5000.0/45000.0), step=0.01, format="%.4f")
            auto_ratio = st.checkbox("Calculate ratio automatically", value=True)
        selected_model = st.selectbox("Prediction model", list(results.keys()), index=list(results.keys()).index(best_name))
        submitted = st.form_submit_button("Analyze default risk", type="primary", use_container_width=True)

    if submitted:
        final_ratio = (loan / income) if auto_ratio and income > 0 else (0.0 if auto_ratio else ratio)
        row = pd.DataFrame([[income, age, loan, final_ratio]], columns=FEATURES)
        chosen_model = results[selected_model]["model"]
        probability = float(chosen_model.predict_proba(row)[0, 1])
        prediction = int(chosen_model.predict(row)[0])
        band = risk_band(probability)

        st.divider()
        x1, x2, x3 = st.columns(3)
        x1.metric("Estimated default probability", f"{probability:.1%}")
        x2.metric("Predicted class", "Default risk detected" if prediction else "No default predicted")
        x3.metric("Illustrative risk band", band)
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=probability*100,
            number={"suffix": "%"},
            title={"text": "Estimated probability"},
            gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#087E8B"},
                   "steps": [{"range": [0, 30], "color": "#D7F3E3"},
                             {"range": [30, 65], "color": "#FFF0C2"},
                             {"range": [65, 100], "color": "#FADBD8"}]}
        ))
        fig.update_layout(height=280, margin=dict(t=35, b=10, l=20, r=20))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Risk bands are illustrative thresholds for this demo, not calibrated lending policy thresholds.")
        st.write("**Input summary**")
        st.dataframe(row, use_container_width=True, hide_index=True)

elif page == "Batch Analysis":
    hero("Batch Risk Analysis", "Upload customer records and export predictions")
    st.write("Upload a CSV with the four model input columns. The target `Default` column is not required for prediction.")
    template = pd.DataFrame({
        "Income": [45000, 62000],
        "Age": [35, 48],
        "Loan": [5000, 9000],
        "Loan to Income": [5000/45000, 9000/62000]
    })
    st.download_button("Download sample CSV template", template.to_csv(index=False).encode("utf-8"), "credit_risk_template.csv", "text/csv")
    file = st.file_uploader("Upload customer CSV", type=["csv"], key="batch_file")
    model_name = st.selectbox("Choose model for batch prediction", list(results.keys()), index=list(results.keys()).index(best_name), key="batch_model")
    if file:
        try:
            batch = pd.read_csv(file)
            missing = [c for c in FEATURES if c not in batch.columns]
            if missing:
                st.error("Missing required input columns: " + ", ".join(missing))
            else:
                X_batch = batch[FEATURES].apply(pd.to_numeric, errors="coerce")
                if X_batch.isna().any().any():
                    st.error("Some input values are missing or non-numeric. Fix the CSV and upload again.")
                else:
                    mdl = results[model_name]["model"]
                    probs = mdl.predict_proba(X_batch)[:, 1]
                    preds = mdl.predict(X_batch)
                    output = batch.copy()
                    output["Default Probability"] = np.round(probs, 4)
                    output["Predicted Default"] = preds
                    output["Risk Band (demo)"] = [risk_band(float(p)) for p in probs]
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Rows processed", f"{len(output):,}")
                    m2.metric("Predicted defaults", int(np.sum(preds)))
                    m3.metric("Average predicted probability", f"{np.mean(probs):.1%}")
                    st.dataframe(output, use_container_width=True, hide_index=True)
                    st.download_button("Download prediction report", output.to_csv(index=False).encode("utf-8"), "credit_risk_predictions.csv", "text/csv", type="primary")
                    fig = px.histogram(output, x="Default Probability", nbins=20, title="Distribution of predicted default probabilities")
                    st.plotly_chart(fig, use_container_width=True)
        except Exception as exc:
            st.error(f"Could not process CSV: {exc}")

elif page == "Model Lab":
    hero("Model Lab", "Evaluate and compare classifiers using a held-out test set")
    st.write("The models are trained on 70% of the data and evaluated on a stratified 30% hold-out set.")
    st.dataframe(score_table.style.format({
        "Accuracy": "{:.3f}", "Precision": "{:.3f}", "Recall": "{:.3f}",
        "F1-score": "{:.3f}", "ROC-AUC": "{:.3f}"
    }), use_container_width=True, hide_index=True)

    selected = st.selectbox("Inspect model", list(results.keys()), index=list(results.keys()).index(best_name), key="inspect_model")
    chosen = results[selected]
    pred = chosen["pred"]
    prob = chosen["prob"]
    cm = confusion_matrix(y_test, pred, labels=[0, 1])
    left, right = st.columns(2)
    with left:
        st.subheader("Confusion matrix")
        cm_df = pd.DataFrame(cm, index=["Actual: No Default", "Actual: Default"], columns=["Predicted: No Default", "Predicted: Default"])
        fig = px.imshow(cm_df, text_auto=True, aspect="auto", color_continuous_scale="Teal")
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.subheader("ROC curve")
        if prob is not None:
            fpr, tpr, _ = roc_curve(y_test, prob)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{selected} (AUC {roc_auc_score(y_test, prob):.3f})"))
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random baseline", line=dict(dash="dash")))
            fig.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate", xaxis_range=[0, 1], yaxis_range=[0, 1])
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("ROC probabilities are not available for this model.")

    st.subheader("Classification report")
    report = classification_report(y_test, pred, output_dict=True, zero_division=0)
    st.dataframe(pd.DataFrame(report).transpose().round(3), use_container_width=True)

    st.subheader("Feature importance (permutation-based)")
    with st.spinner("Calculating feature importance..."):
        try:
            perm = permutation_importance(
                chosen["model"], X_test, y_test, n_repeats=5,
                random_state=42, scoring="f1", n_jobs=-1
            )
            imp = pd.DataFrame({"Feature": FEATURES, "Importance": perm.importances_mean}).sort_values("Importance", ascending=True)
            fig = px.bar(imp, x="Importance", y="Feature", orientation="h", title="Change in F1-score when a feature is shuffled")
            st.plotly_chart(fig, use_container_width=True)
        except Exception as exc:
            st.info(f"Feature importance could not be calculated: {exc}")

elif page == "About Project":
    hero("About the Project", "A final-year-level upgrade of the original Credit Card Default Prediction notebook")
    st.subheader("Problem statement")
    st.write("Credit default can create financial losses for lenders. This project demonstrates how customer financial attributes can be used to estimate default risk and communicate model performance.")
    st.subheader("Objectives")
    st.markdown("""
    - Prepare and validate the dataset.
    - Compare multiple supervised machine-learning classifiers.
    - Evaluate performance using precision, recall, F1-score and ROC-AUC.
    - Provide individual and batch prediction interfaces.
    - Present feature influence and model limitations.
    """)
    st.subheader("Workflow")
    st.code("Dataset → Validation → Stratified Train/Test Split → Model Training → Evaluation → Prediction UI → CSV Report", language="text")
    st.subheader("Dataset and limitations")
    st.write(f"Current source: {source_label}. The included public demo dataset has a small number of rows and simplified features. Results should not be generalized to real lending decisions.")
    st.write("Default is a binary target: 0 = no default, 1 = default. The risk-band thresholds are illustrative. The app does not make or recommend real credit decisions.")
    st.subheader("Suggested future scope")
    st.markdown("""
    - Validate on a larger, representative dataset.
    - Add cross-validation and hyperparameter tuning.
    - Calibrate probability estimates and assess fairness.
    - Track model drift and version model artifacts.
    - Add authentication and audit logging before any real-world use.
    """)

st.sidebar.divider()
st.sidebar.caption(f"Current model selection: {best_name}")
st.sidebar.caption("Built with Python · Scikit-learn · Streamlit · Plotly")
