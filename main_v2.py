# ========================
#  PCOS Detection Tool v5
# ========================

import os
import tempfile
import urllib.request
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.figure_factory as ff
import streamlit as st

from PIL import Image
from fpdf import FPDF
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, roc_auc_score,
    roc_curve, auc
)
from sklearn.model_selection import (
    train_test_split, GridSearchCV, RandomizedSearchCV,
    StratifiedKFold, KFold, RepeatedStratifiedKFold
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# Streamlit app configuration
st.set_page_config(page_title="PCOS Detection Tool v5", layout="wide")
st.title("🧬 PCOS Detection – Real Data Model")

# Suppress unnecessary warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# ========== File Upload ==========
st.sidebar.header("📤 Upload Real Survey Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

# if uploaded_file is None:
#     st.info("Please upload a CSV file containing real survey responses.")
#     st.stop()

data = pd.read_csv("PCOS_Data.csv")

# ========== Preprocessing Utilities ==========
ordinal_maps = {
    "never": 0, "rarely": 1, "sometimes": 2, "often": 3, "very often": 4,
    "low": 1, "medium": 2, "high": 3
}

yes_no_map = {
    "yes": 1, "no": 0, "true": 1, "false": 0
}

def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the input dataframe:
    - Normalize and map ordinal/boolean text values
    - One-hot encode remaining categorical columns
    - Ensure numeric types
    """
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype(str).str.strip().str.lower()
            if set(df[col].unique()).issubset(ordinal_maps):
                df[col] = df[col].map(ordinal_maps)
            elif set(df[col].unique()).issubset(yes_no_map):
                df[col] = df[col].map(yes_no_map)

    remaining_obj_cols = df.select_dtypes(include='object').columns.tolist()
    if remaining_obj_cols:
        df = pd.get_dummies(df, columns=remaining_obj_cols, drop_first=True)

    return df.apply(pd.to_numeric, errors='coerce').fillna(0)

def preprocess_family_conditions(df):
    conditions = ["diabetes", "hypertension", "cancer", "heart disease", "thalassemia",
                  "adenopathies", "parkinsons", "osteoporosis", "obesity"]
    df["family_history"] = df["family_conditions"].notna().astype(int)

    for cond in conditions:
        df[f"family_conditions_{cond}"] = df["family_conditions"].str.contains(cond, case=False, na=False).astype(int)

    return df

# Apply preprocessing once
# Step 1: Generate clean binary flags from family_conditions text
data = preprocess_family_conditions(data)

# Step 2: Drop the raw string column to avoid one-hot duplication
data = data.drop(columns=["family_conditions"], errors="ignore")

# Step 3: Apply standard preprocessing
data = preprocess_dataframe(data)

# Step 4: Prepare features/target
non_feature_cols = ['pcos_onset_age']
features = data.drop(columns=[col for col in non_feature_cols if col in data.columns] + ['has_pcos'], errors='ignore')
target = data['has_pcos'] if 'has_pcos' in data.columns else None

# ========== Tab Layout ==========
tab1, tab2 = st.tabs([
    "🔍 PCOS Predictor (User Tool)",
    "🧪 Model Experimentation (Research Panel)"
])

# ========== Tab 2: Experimentation Panel ==========
with tab2:
    st.subheader("📋 Uploaded Data Preview")
    with st.expander("Show DataFrame", expanded=True):
        st.dataframe(data, hide_index=True)

    st.subheader("🧪 Experiment Plan – Combo Selector")

    default_combo_df = pd.DataFrame([
        ["C1", 0.2, "Stratified K-Fold", "Random Forest", "Grid Search", 100, "Balanced performance with tree-based model"],
        ["C2", 0.3, "Repeated Stratified K-Fold", "Random Forest", "Randomized Search", 100, "More generalization + noise tolerance"],
        ["C3", 0.2, "Stratified K-Fold", "Logistic Regression", "Grid Search", None, "Interpretable baseline model"],
        ["C4", 0.4, "K-Fold", "Logistic Regression", "Randomized Search", None, "Stress test with less training data"],
        ["C5", 0.25, "Stratified K-Fold", "SVM", "Grid Search", None, "Margin-based classifier on balanced split"],
        ["C6", 0.3, "Repeated Stratified K-Fold", "SVM", "Randomized Search", None, "Kernel model tested for stability"],
        ["C7", 0.5, "Stratified K-Fold", "Random Forest", "Grid Search", 50, "Risky split – checking model consistency"],
        ["C8", 0.15, "Stratified K-Fold", "Logistic Regression", "Grid Search", None, "High training volume, small test risk"],
        ["C9", 0.2, "K-Fold", "SVM", "Grid Search", None, "Moderate CV without stratification"],
        ["C10", 0.4, "Stratified K-Fold", "Random Forest", "Grid Search", 200, "Imbalance sensitivity check"]
    ], columns=[
        "Combo ID", "Train/Test Split", "Cross-Validation", "Model", 
        "Search Type", "n_estimators", "Key Insight Target"
    ])

    edited_combo_df = st.data_editor(default_combo_df, num_rows="fixed", key="experiment_table")
    selected_combos = st.multiselect("🔍 Select Combo/s ID to Run", edited_combo_df["Combo ID"].tolist())
    run_button = st.button("🚀 Run Selected Combos")

    # Mapping for CV strategies
    cv_map = {
        "Stratified K-Fold": StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        "K-Fold": KFold(n_splits=5, shuffle=True, random_state=42),
        "Repeated Stratified K-Fold": RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42)
    }

    results = []

    if run_button:
        for combo_id in selected_combos:
            row = edited_combo_df[edited_combo_df["Combo ID"] == combo_id].iloc[0]
            split_ratio = float(row["Train/Test Split"])
            cv_name = row["Cross-Validation"]
            model_name = row["Model"]
            search_type = row["Search Type"]
            custom_estimators = row["n_estimators"] if pd.notna(row["n_estimators"]) else 100

            try:
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    features, target,
                    test_size=split_ratio,
                    stratify=target,
                    random_state=42
                )

                # Get CV strategy
                cv = cv_map.get(cv_name)

                # Select model and hyperparameters
                if model_name == "Random Forest":
                    model = RandomForestClassifier(random_state=42, class_weight="balanced")
                    param_grid = {
                        "n_estimators": [int(custom_estimators)],
                        "max_depth": [None, 10, 20],
                        "min_samples_split": [2, 5]
                    }
                elif model_name == "Logistic Regression":
                    model = LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear")
                    param_grid = {
                        "C": [0.01, 0.1, 1, 10],
                        "penalty": ["l2"]
                    }
                elif model_name == "SVM":
                    model = SVC(probability=True, class_weight="balanced")
                    param_grid = {
                        "C": [0.1, 1, 10],
                        "kernel": ["rbf"],
                        "gamma": ["scale", "auto"]
                    }

                # Select search method
                if search_type == "Grid Search":
                    search = GridSearchCV(model, param_grid, cv=cv, scoring='roc_auc', n_jobs=-1, error_score=np.nan)
                else:
                    search = RandomizedSearchCV(model, param_grid, n_iter=10, cv=cv, scoring='roc_auc', n_jobs=-1, random_state=42, error_score=np.nan)

                search.fit(X_train, y_train)
                best_model = search.best_estimator_
                y_pred = best_model.predict(X_test)
                y_prob = best_model.predict_proba(X_test)[:, 1]
                confidence_avg = np.mean(np.max(best_model.predict_proba(X_test), axis=1))


                # Feature importance
                if model_name == "Random Forest":
                    feat_imp = list(pd.Series(best_model.feature_importances_, index=features.columns).sort_values(ascending=False).head(5).index)
                elif model_name == "Logistic Regression":
                    feat_imp = list(pd.Series(best_model.coef_[0], index=features.columns).abs().sort_values(ascending=False).head(5).index)
                elif model_name == "SVM" and hasattr(best_model, "coef_"):
                    feat_imp = list(pd.Series(best_model.coef_[0], index=features.columns).abs().sort_values(ascending=False).head(5).index)
                else:
                    feat_imp = ["N/A"]

                results.append({
                    "Combo ID": combo_id,
                    "Split Ratio": f"{int((1-split_ratio)*100)}/{int(split_ratio*100)}",
                    "Best Parameters": str(search.best_params_),
                    "ROC AUC": round(roc_auc_score(y_test, y_prob), 3),
                    "Accuracy": f"{accuracy_score(y_test, y_pred) * 100:.2f}%",
                    "Confidence": f"{confidence_avg:.2f}",
                    "Top 5 Features": ", ".join(feat_imp)
                })

            except Exception as e:
                results.append({
                    "Combo ID": combo_id,
                    "Split Ratio": f"{int((1-split_ratio)*100)}/{int(split_ratio*100)}",
                    "Best Parameters": "Error",
                    "ROC AUC": "Error",
                    "Accuracy": "Error",
                    "Top 5 Features": str(e)
                })

    if results:
        st.subheader("📋 Combo Run Results")
        st.dataframe(pd.DataFrame(results)[["Combo ID", "Split Ratio", "Accuracy", "ROC AUC", "Confidence", "Top 5 Features", "Best Parameters"]])


# ========== Tab 1: PCOS Predictor Tool ==========
with tab1:
    st.subheader("📋 Uploaded Data Preview")
    with st.expander("Show DataFrame", expanded=True):
        st.dataframe(data, hide_index=True)

    st.subheader("🔧 Advanced Model Training & Evaluation")

    # --- User controls ---
    split_ratio = st.slider("📐 Select Train/Test Split Ratio", 0.1, 0.5, 0.2, 0.05)
    X_train, X_test, y_train, y_test = train_test_split(
        features, target,
        test_size=split_ratio,
        stratify=target,
        random_state=42
    )
    X_train = X_train.apply(pd.to_numeric, errors='coerce').fillna(0)
    X_test = X_test.apply(pd.to_numeric, errors='coerce').fillna(0)

    # --- Cross-validation selection ---
    cv_method = st.selectbox("🔁 Choose Cross-Validation Strategy", [
        "Stratified K-Fold", "K-Fold", "Repeated Stratified K-Fold"])
    cv_options = {
        "Stratified K-Fold": StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        "K-Fold": KFold(n_splits=5, shuffle=True, random_state=42),
        "Repeated Stratified K-Fold": RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42)
    }
    cv = cv_options[cv_method]

    # --- Model selection & manual input with tuning ---
    model_option = st.selectbox("🧠 Select a Model for Tuning", ["Random Forest", "Logistic Regression", "SVM"])
    selected_params = {}

    if model_option == "Random Forest":
        st.subheader("🌲 Random Forest Parameters")

        n_estimators = st.slider("n_estimators", 50, 200, 100)
        max_depth = st.selectbox("max_depth", [None, 5, 10, 20], index=2)  # Default = 10
        min_samples_split = st.slider("min_samples_split", 2, 10, 2)

        selected_model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            class_weight="balanced",
            random_state=42
        )

        # Grid search will still explore if you include these in the param grid
        selected_params = {
            # Remove from here if you don't want them tuned
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5]
        }

    elif model_option == "Logistic Regression":
        st.subheader("📈 Logistic Regression Parameters")

        C = st.select_slider("C (Regularization Strength)", [0.01, 0.1, 1.0, 10.0], value=1.0)

        selected_model = LogisticRegression(
            C=C,
            penalty="l2",
            solver="liblinear",
            max_iter=1000,
            class_weight="balanced"
        )

        selected_params = {
            "C": [0.01, 0.1, 1.0, 10.0]
        }

    elif model_option == "SVM":
        st.subheader("📊 SVM Parameters")

        C = st.select_slider("C", [0.1, 1.0, 10.0], value=1.0)
        gamma = st.selectbox("gamma", ["scale", "auto"], index=0)

        selected_model = SVC(
            C=C,
            gamma=gamma,
            kernel="rbf",
            probability=True,
            class_weight="balanced"
        )

        selected_params = {
            "C": [0.1, 1.0, 10.0],
            "gamma": ["scale", "auto"]
        }

    # Show parameter overview
    # st.markdown("### 🔍 Parameter Overview")
    # st.write("🔧 Fixed Parameters", selected_model.get_params())
    # st.write("🎯 Tunable Parameters", selected_params)

    # Model insight summary
    def generate_model_insights(params, model_type):
        insights = []
        if model_type == "Random Forest":
            md = params.get("max_depth")
            if md is None:
                insights.append("⚠️ `max_depth=None`: Trees will grow until pure → likely to **overfit**.")
            elif md <= 5:
                insights.append(f"✅ `max_depth={md}`: Shallow trees → likely **fast** and **generalizable**.")
            elif md > 15:
                insights.append(f"⚠️ `max_depth={md}`: Deep trees → could **overfit** small patterns.")
            else:
                insights.append(f"• `max_depth={md}`: Balanced depth for moderate complexity.")

            mss = params.get("min_samples_split", 2)
            if mss <= 2:
                insights.append("⚠️ `min_samples_split=2`: Splits aggressively → could **overfit**.")
            elif mss >= 10:
                insights.append("✅ Higher `min_samples_split` → more conservative, likely to **generalize better**.")
            else:
                insights.append(f"• `min_samples_split={mss}`: Reasonable node size requirement.")

            ne = params.get("n_estimators", 100)
            if ne < 50:
                insights.append(f"⚠️ Only `{ne}` trees → might be **unstable** on noisy data.")
            elif ne > 200:
                insights.append(f"✅ `{ne}` trees → strong ensemble, but **slower training**.")
            else:
                insights.append(f"• `{ne}` trees → balances performance and speed.")

            if params.get("class_weight") == "balanced":
                insights.append("• Class imbalance handled with `class_weight='balanced'` → helps in skewed datasets.")

            if params.get("max_features") == "sqrt":
                insights.append("• `max_features='sqrt'`: Ensures randomness per split → reduces correlation between trees.")

        elif model_type == "Logistic Regression":
            C = params.get("C", 1.0)
            if C < 0.1:
                insights.append(f"✅ `C={C}` → High regularization → avoids overfitting.")
            elif C > 10:
                insights.append(f"⚠️ `C={C}` → Low regularization → may **overfit** complex data.")
            else:
                insights.append(f"• `C={C}` → Balanced between regularization and flexibility.")

            solver = params.get("solver", "")
            insights.append(f"• Using solver: `{solver}` → affects speed and convergence for different penalties.")

            if params.get("class_weight") == "balanced":
                insights.append("• Class imbalance handled with `class_weight='balanced'`.")

        elif model_type == "SVM":
            C = params.get("C", 1.0)
            if C < 0.5:
                insights.append(f"✅ `C={C}` → Smoother margin, less prone to overfitting.")
            elif C > 10:
                insights.append(f"⚠️ `C={C}` → Very tight margin → model might **overfit** outliers.")
            else:
                insights.append(f"• `C={C}` → Balanced decision boundary.")

            gamma = params.get("gamma", "scale")
            if gamma == "auto":
                insights.append("⚠️ `gamma='auto'` uses 1/#features → can lead to **overfitting**.")
            elif gamma == "scale":
                insights.append("✅ `gamma='scale'` adapts to data variance → usually safer.")

            if params.get("class_weight") == "balanced":
                insights.append("• Class imbalance handled with `class_weight='balanced'`.")

        return insights

    st.markdown("### 🧠 Model Insight Summary")
    model_insights = generate_model_insights(selected_model.get_params(), model_option)
    for insight in model_insights:
        st.markdown(insight)

    # --- Search Type + Execution ---
    search_type = st.radio("🔍 Select Search Method", ["Grid Search", "Randomized Search"])

    if "tuning_done" not in st.session_state:
        st.session_state.tuning_done = False

    if st.button("🚀 Run Hyperparameter Tuning"):
        st.session_state.tuning_done = True

    if st.session_state.tuning_done:
        st.info("Running model tuning... Please wait ⏳")

        if search_type == "Grid Search":
            search = GridSearchCV(
                selected_model, selected_params, cv=cv,
                scoring='roc_auc', n_jobs=-1, error_score=np.nan
            )
        else:
            search = RandomizedSearchCV(
                selected_model, selected_params, n_iter=10,
                cv=cv, scoring='roc_auc', n_jobs=-1, random_state=42,
                error_score=np.nan
            )

        try:
            search.fit(X_train, y_train)
        except Exception as e:
            st.error(f"❌ Tuning failed: {str(e)}")
            st.stop()

        best_model = search.best_estimator_
        st.session_state.best_model = best_model

        y_pred = best_model.predict(X_test)
        y_prob = best_model.predict_proba(X_test)[:, 1]

        st.success(f"✅ Best Parameters: {search.best_params_}")


        # === Feature Importance ===
        st.markdown("### 🔬 Feature Importance")

        def plot_horizontal_bar(data, title):
            df_plot = data.head(10).sort_values(ascending=True)
            fig = px.bar(
                df_plot,
                x=df_plot.values,
                y=df_plot.index,
                orientation='h',
                labels={'x': 'Importance', 'index': 'Feature'},
                title=title
            )
            st.plotly_chart(fig, use_container_width=True)

        if model_option == "Random Forest":
            importances = pd.Series(best_model.feature_importances_, index=features.columns).sort_values(ascending=False)
            plot_horizontal_bar(importances, "Top 10 Feature Importances (Gini)")
            st.caption("Technique: Gini Importance")

        elif model_option == "Logistic Regression":
            coef = pd.Series(best_model.coef_[0], index=features.columns)
            plot_horizontal_bar(coef.abs().sort_values(ascending=False), "Top 10 Coefficient Magnitudes")
            st.caption("Technique: Coefficient Magnitude")

        elif model_option == "SVM" and hasattr(best_model, "coef_"):
            coef = pd.Series(best_model.coef_[0], index=features.columns)
            plot_horizontal_bar(coef.abs().sort_values(ascending=False), "Top 10 SVM Coefficients")
            st.caption("Technique: SVM Coefficients")

        else:
            st.info("Feature importance is not available for this model.")

        # === Model Evaluation ===
        st.subheader("📈 Evaluation Using Tuned Model")

        acc = accuracy_score(y_test, y_pred)
        roc_score = roc_auc_score(y_test, y_prob)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)

        col1, col2, col3 = st.columns([3, 3, 2])
        with col1:
            fig_cm, ax_cm = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=['No PCOS', 'PCOS'],
                        yticklabels=['No PCOS', 'PCOS'],
                        ax=ax_cm)
            ax_cm.set_title("Confusion Matrix")
            ax_cm.set_xlabel("Predicted")
            ax_cm.set_ylabel("Actual")
            st.pyplot(fig_cm)

        with col2:
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)
            fig_roc, ax_roc = plt.subplots()
            ax_roc.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
            ax_roc.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            ax_roc.set_xlim([0.0, 1.0])
            ax_roc.set_ylim([0.0, 1.05])
            ax_roc.set_xlabel('False Positive Rate')
            ax_roc.set_ylabel('True Positive Rate')
            ax_roc.set_title('ROC Curve')
            ax_roc.legend(loc="lower right")
            st.pyplot(fig_roc)

        with col3:
            st.metric("Accuracy", f"{acc*100:.2f}%")
            st.metric("ROC-AUC", f"{roc_score:.2f}")
            st.markdown("### Classification Report")
            df_report = pd.DataFrame(report).T.round(2)
            st.dataframe(df_report)

        # === Sidebar Patient Input + Prediction ===
        # === Grouped Sidebar Input ===
        st.sidebar.markdown("## 🧠 Manual Patient Input (Grouped)")
        user_inputs = {}

        # Ensure family condition fields exist in X_train
        family_condition_fields = [
            "family_conditions_diabetes",
            "family_conditions_hypertension",
            "family_conditions_cancer",
            "family_conditions_heart disease",
            "family_conditions_thalassemia",
            "family_conditions_adenopathies",
            "family_conditions_parkinsons",
            "family_conditions_osteoporosis",
        ]
        family_condition_fields = [col for col in family_condition_fields if col in X_train.columns]

        grouped_inputs = {
            "📊 Basic Information": ["age", "bmi", "height_cm", "weight_kg", "waist_size"],
            "🩺 Symptoms": [col for col in X_train.columns if any(sym in col for sym in ["acne", "hirsutism", "fatigue", "pelvic_pain", "hair_loss", "baldness", "irregular_periods", "sleep_difficulty", "mood_changes"])],
            "💉 Medical History": family_condition_fields + [col for col in X_train.columns if "mental_health" in col or "medication" in col or "supplements" in col],
            "🥗 Lifestyle & Diet": [col for col in X_train.columns if any(k in col for k in ["diet_", "sugary_drinks", "physical_activity", "caffeine", "water_intake", "sleep_hours", "sports_type", "meal_frequency", "eating_speed", "dairy"])],
        }

        with st.sidebar.form("grouped_input_form"):
            for section, fields in grouped_inputs.items():
                with st.expander(section, expanded=False):
                    for col in fields:
                        custom_labels = {
                            # Sports & Activity
                            "sports_type_i don't do sports": "No Sport Activity",
                            "sports_type_weight": "Weight Loss Activity",

                            # Sleep & Diet
                            "sleep_hours_more than 8 hours": "Sleep > 8 Hours",
                            "diet_quality_unbalanced": "Unbalanced Diet",
                            "physical_activity_weekly": "Weekly Physical Activity",
                            "water_intake_more than 1 liter": "Water Intake > 1 Liter",
                            "meal_frequency_more than 3 meals a day": "Frequent Meals (>3/day)",
                            "eating_speed_normal": "Normal Eating Speed",
                            "eating_speed_slowly": "Slow Eating Speed",
                            "diet_type_protein-based": "Protein-Based Diet",

                            # Medical History
                            "family_conditions_diabetes": "Family History: Diabetes",
                            "family_conditions_hypertension": "Family History: Hypertension",
                            "family_conditions_cancer": "Family History: Cancer",
                            "family_conditions_heart_disease": "Family History: Heart Disease",
                            "family_conditions_thalassemia": "Family History: Thalassemia",
                            "family_conditions_adenopathies": "Family History: Adenopathies",
                            "family_conditions_parkinsons": "Family History: Parkinson's",
                            "family_conditions_osteoporosis": "Family History: Osteoporosis",
                            "family_conditions_obesity": "Family History: Obesity",
                            "mental_health_dx": "Mental Health Diagnosis",
                            "medication_pcos": "PCOS Medications",
                            "supplements": "Takes Supplements",

                            # Symptoms
                            "irregular_periods": "Irregular Periods",
                            "acne": "Acne",
                            "hirsutism": "Hirsutism (Excess Hair)",
                            "hair_loss": "Hair Loss",
                            "pelvic_pain": "Pelvic Pain",
                            "fatigue": "Fatigue",
                            "baldness": "Baldness",
                            "sleep_difficulty": "Sleep Difficulty",
                            "mood_changes_no": "Mood Changes (No)",
                            "mood_changes_yes": "Mood Changes (Yes)",

                            # Basic Info (optional beautification)
                            "height_cm": "Height (cm)",
                            "weight_kg": "Weight (kg)",
                            "waist_size": "Waist Size (cm)",
                            "bmi": "BMI",
                            "age": "Age",
                        }


                        label = custom_labels.get(col.lower(), col.replace("_", " ").title())


                        # # Skip sub-family conditions unless family_history is checked
                        # if col.startswith("family_conditions_") and not user_inputs.get("family_history", False):
                        #     continue

                        if X_train[col].nunique() == 2 and set(X_train[col].unique()).issubset({0, 1}):
                            user_inputs[col] = st.checkbox(f"{label}")
                        else:
                            user_inputs[col] = st.number_input(f"{label}", value=0.0, step=1.0)


            submitted = st.form_submit_button("🔎 Predict")

        if submitted:
            input_data = pd.DataFrame([{col: int(user_inputs.get(col, 0)) for col in X_train.columns}])
            # Dynamically set family_history = 1 if any family_conditions are checked
            family_condition_cols = [col for col in input_data.columns if col.startswith("family_conditions_")]
            input_data["family_history"] = int(input_data[family_condition_cols].sum(axis=1).iloc[0] > 0)
            
            # Predict
            try:
                pred = int(best_model.predict(input_data)[0])
                prob = best_model.predict_proba(input_data)[0][pred]
                if pred == 1:
                    st.sidebar.error(f"🚨 Likely PCOS\nConfidence: {prob:.2f}")
                else:
                    st.sidebar.success(f"✅ Unlikely PCOS\nConfidence: {prob:.2f}")
            except Exception as e:
                st.sidebar.warning(f"⚠️ Prediction failed: {str(e)}")

            # Export PDF
            def sanitize_text(text):
                return str(text).encode("latin-1", "replace").decode("latin-1")

            # === Redesigned PDF Report Function ===
            def generate_clinical_pdf_report(input_df, prediction, confidence, feature_importance_df):
                from fpdf import FPDF
                import tempfile
                import urllib.request
                import os
                import datetime

                # Download logo
                logo_url = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQ7TmfIutY-gpYCpl4Bl-lhSKWcHlglDarrRA&s"
                logo_path = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
                urllib.request.urlretrieve(logo_url, logo_path)

                pdf = FPDF()
                pdf.add_page()
                pdf.set_auto_page_break(auto=True, margin=15)

                # Header Bar
                pdf.set_fill_color(44, 62, 80)
                pdf.rect(0, 0, 210, 25, style='F')
                pdf.image(logo_path, x=10, y=5, w=15)
                pdf.set_text_color(255, 255, 255)
                pdf.set_font("Arial", "B", 16)
                pdf.set_xy(30, 8)
                pdf.cell(0, 10, "PCOS Clinical Evaluation Report", ln=True)
                pdf.set_font("Arial", "", 11)
                pdf.set_xy(30, 16)
                pdf.cell(0, 8, f"Date: {datetime.date.today().strftime('%B %d, %Y')}", ln=True)

                pdf.set_text_color(0, 0, 0)
                pdf.set_font("Arial", "", 12)
                pdf.ln(20)

                # Prediction Summary
                pdf.set_font("Arial", "B", 14)
                pdf.cell(0, 10, "Prediction Summary", ln=True)
                pdf.ln(2)

                if prediction == 1:
                    pdf.set_fill_color(255, 102, 102)  # red
                    diagnosis_text = "LIKELY PCOS"
                else:
                    pdf.set_fill_color(102, 204, 153)  # green
                    diagnosis_text = "UNLIKELY PCOS"

                pdf.set_text_color(0, 0, 0)
                pdf.set_font("Arial", "B", 12)
                pdf.cell(0, 10, f"Diagnosis: {diagnosis_text}    |    Confidence: {confidence*100:.2f}%", ln=True, fill=True)
                pdf.ln(10)

                # === REPORT METADATA ===
                import random
                patient_id = f"P{random.randint(1000, 9999)}"
                report_id = f"RPT-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"

                pdf.set_font("Arial", "B", 11)
                pdf.cell(0, 8, f"Patient ID: {patient_id}    |    Report ID: {report_id}", ln=True)
                pdf.set_font("Arial", "", 10)
                pdf.cell(0, 6, "Confidential - For Clinical Use Only", ln=True)
                pdf.ln(5)

                # === PATIENT PROFILE (Survey-Inspired, Styled) ===

                def print_section(title, fill_color=(230, 240, 255)):
                    pdf.set_fill_color(*fill_color)
                    pdf.set_font("Arial", "B", 12)
                    pdf.cell(0, 8, f"{title}", ln=True, fill=True)
                    pdf.set_font("Arial", "", 11)

                def bullet_item(text):
                    pdf.cell(5)  # indent
                    pdf.cell(0, 8, f"- {sanitize_text(text)}", ln=True)

                # === 1. Vitals ===
                print_section("Vitals")
                for col in ["age", "height_cm", "weight_kg", "waist_size", "bmi"]:
                    if col in input_df:
                        label = col.replace("_cm", " (cm)").replace("_kg", " (kg)").replace("_", " ").capitalize()
                        value = input_df[col].values[0]
                        pdf.cell(0, 8, f"{sanitize_text(label)}: {sanitize_text(value)}", ln=True)
                pdf.ln(3)

                # === 2. Menstrual & Hormonal Symptoms ===
                print_section("Menstrual & Hormonal Symptoms")
                symptoms_map = {
                    "irregular_periods": "Irregular or missed periods",
                    "pelvic_pain": "Pelvic pain",
                    "acne": "Acne",
                    "hair_loss": "Hair loss",
                    "baldness": "Male-pattern baldness",
                    "hirsutism": "Excessive body hair",
                    "weight_gain": "Weight gain"
                }
                any_symptom = False
                for key, desc in symptoms_map.items():
                    if key in input_df and input_df[key].values[0]:
                        bullet_item(desc)
                        any_symptom = True
                if not any_symptom:
                    bullet_item("None reported")
                pdf.ln(3)

                # === 3. Mental Health & Fatigue ===
                print_section("Mental Health & Fatigue")
                mental_keys = {
                    "fatigue": "Fatigue",
                    "depression_diagnosis": "Diagnosed with depression",
                    "mood_changes": "Mood changes",
                    "sleep_difficulty": "Sleep difficulties",
                    "stress_anxiety": "Frequent stress or anxiety"
                }
                any_mental = False
                for key, desc in mental_keys.items():
                    if key in input_df and input_df[key].values[0]:
                        bullet_item(desc)
                        any_mental = True
                if not any_mental:
                    bullet_item("None reported")
                pdf.ln(3)

                # === 4. Lifestyle & Diet ===
                print_section("Lifestyle & Diet")
                diet_keys = {
                    "diet_type_protein-based": "Protein-based diet",
                    "diet_type_carbohydrate-based": "Carbohydrate-based diet",
                    "eating_speed_normal": "Eats at normal speed",
                    "eating_speed_slowly": "Eats slowly",
                    "meal_frequency_more_than_3_meals": "More than 3 meals/day",
                    "water_intake_more_than_1_liter": "Drinks >1L water daily",
                    "caffeine_intake": "Caffeine consumption",
                    "dairy_consumption": "Consumes dairy",
                    "sugary_drinks": "Frequently drinks sugary beverages",
                    "physical_activity_never": "No physical activity",
                    "physical_activity_weekly": "Exercises weekly"
                }
                any_life = False
                for key, desc in diet_keys.items():
                    if key in input_df and input_df[key].values[0]:
                        bullet_item(desc)
                        any_life = True
                if not any_life:
                    bullet_item("No lifestyle risks reported")
                pdf.ln(3)

                # === 5. Family Medical History ===
                print_section("Family Medical History")
                family_conditions = [k for k in input_df.columns if "family_conditions" in k.lower()]
                reported = [k.replace("family_conditions_", "").replace("_", ", ").capitalize()
                            for k in family_conditions if input_df[k].values[0]]
                if reported:
                    for r in reported:
                        bullet_item(r)
                else:
                    bullet_item("No known family history reported")
                pdf.ln(5)

                # === NEW PAGE: Technical Summary ===
                pdf.add_page()

                # Section Title
                pdf.set_font("Arial", "B", 14)
                pdf.set_fill_color(200, 230, 255)
                pdf.cell(0, 10, "Technical Summary & Diagnostic Insights", ln=True, fill=True)
                pdf.set_font("Arial", "", 11)
                pdf.cell(0, 8, f"Submission Date: {datetime.date.today().strftime('%B %d, %Y')}", ln=True)
                pdf.ln(5)

                # Patient Input Table
                # Aligned 2-column table using fixed-height cells
                pdf.set_font("Arial", "B", 12)
                pdf.cell(0, 10, "Patient Input Overview", ln=True)
                pdf.set_font("Arial", "", 11)
                pdf.set_fill_color(245, 245, 245)
                pdf.ln(2)

                from textwrap import wrap

                def clean_entry(text):
                    if str(text).lower() in ["nan", "none", "", "null"]:
                        return None
                    text = str(text).replace("_", " ")
                    text = text.replace("family conditions ", "").replace(":", "")
                    text = " ".join(text.split())
                    return sanitize_text(text.capitalize())

                inputs = []
                custom_labels_pdf = {
                    "age": "Age",
                    "height_cm": "Height (cm)",
                    "weight_kg": "Weight (kg)",
                    "waist_size": "Waist Size",
                    "bmi": "BMI",
                    "irregular_periods": "Irregular Periods",
                    "acne": "Acne",
                    "hirsutism": "Hirsutism",
                    "smoking": "Smoking",
                    "family_history_pcos": "Family History PCOS",
                    "hair_loss": "Hair Loss",
                    "weight_gain": "Weight Gain",
                    "pelvic_pain": "Pelvic Pain",
                    "fatigue": "Fatigue",
                    "caffeine": "Caffeine",
                    "stress": "Stress",
                    "mental_health_dx": "Mental Health Dx",
                    "sugary_drink_freq": "Sugary Drink Frequency",
                    "sugary_drinks": "Sugary Drinks",
                    "dairy": "Dairy",
                    "sleep_difficulty": "Sleep Difficulty",
                    "baldness": "Baldness",
                    "medication_pcos": "Medication PCOS",
                    "supplements": "Supplements",
                    "mood_changes_no": "Mood Changes: No",
                    "mood_changes_yes": "Mood Changes: Yes",
                    "diet_quality_unbalanced": "Unbalanced Diet",
                    "physical_activity_weekly": "Weekly Activity",
                    "sports_type_i don't do sports": "No Sport Activity",
                    "sports_type_weight": "Weight Loss Activity",
                    "sleep_hours_more than 8 hours": "Sleep > 8 Hours",
                    "water_intake_more than 1 liter": "Water > 1L",
                    "diet_type_protein-based": "Protein-Based Diet",
                    "meal_frequency_more than 3 meals a day": "Meals > 3/day",
                    "eating_speed_normal": "Eats at Normal Speed",
                    "eating_speed_slowly": "Eats Slowly",
                    "family_conditions_diabetes": "Family: Diabetes",
                    "family_conditions_hypertension": "Family: Hypertension",
                    "family_conditions_cancer": "Family: Cancer",
                    "family_conditions_heart disease": "Family: Heart Disease",
                    "family_conditions_thalassemia": "Family: Thalassemia",
                    "family_conditions_adenopathies": "Family: Adenopathies",
                    "family_conditions_parkinsons": "Family: Parkinson's",
                    "family_conditions_osteoporosis": "Family: Osteoporosis",
                    "family_conditions_obesity": "Family: Obesity"
                }

                def clean_entry(text):
                    if str(text).lower() in ["nan", "none", "", "null"]:
                        return None
                    text = str(text).strip().lower()
                    return sanitize_text(custom_labels_pdf.get(text, text.replace("_", " ").capitalize()))

                for col, val in input_df.iloc[0].items():
                    label = clean_entry(col)
                    value = clean_entry(val)
                    if label and value:
                        inputs.append((label, value))

                half = len(inputs) // 2 + len(inputs) % 2
                col1 = inputs[:half]
                col2 = inputs[half:]

                label_w, val_w = 55, 20
                spacing = 8
                row_height = 8

                for i in range(max(len(col1), len(col2))):
                    # First column
                    if i < len(col1):
                        l1, v1 = col1[i]
                        pdf.cell(label_w, row_height, f"{l1}:", border=0)
                        pdf.cell(val_w, row_height, v1, border=0)
                    else:
                        pdf.cell(label_w + val_w, row_height, "", border=0)

                    pdf.cell(spacing, row_height, "", border=0)  # Spacer

                    # Second column
                    if i < len(col2):
                        l2, v2 = col2[i]
                        pdf.cell(label_w, row_height, f"{l2}:", border=0)
                        pdf.cell(val_w, row_height, v2, border=0)
                    else:
                        pdf.cell(label_w + val_w, row_height, "", border=0)

                    pdf.ln(row_height)

                # === NEW PAGE: Feature Importance ===
                pdf.add_page()

                # === Feature Importance Table ===
                pdf.set_font("Arial", "B", 13)
                pdf.set_fill_color(200, 230, 255)
                pdf.cell(0, 10, "Top 10 Contributing Features", ln=True, fill=True)
                pdf.ln(2)

                # Table Header
                pdf.set_font("Arial", "B", 11)
                pdf.set_fill_color(230, 230, 230)
                pdf.set_text_color(0)
                pdf.cell(110, 8, "Feature", border=1, fill=True)
                pdf.cell(30, 8, "Importance", border=1, ln=True, fill=True)

                # Table Rows
                pdf.set_font("Arial", "", 10)
                pdf.set_text_color(50)

                row_colors = [(255, 255, 255), (245, 245, 245)]  # alternating white and light gray

                for idx, (i, row) in enumerate(feature_importance_df.head(10).iterrows()):
                    label = row['Feature'].replace("_", " ").capitalize()
                    imp = f"{row['Importance']:.4f}"
                    fill_color = row_colors[idx % 2]
                    pdf.set_fill_color(*fill_color)
                    pdf.cell(110, 8, sanitize_text(label), border=1, fill=True)
                    pdf.cell(30, 8, sanitize_text(imp), border=1, ln=True, fill=True)

                pdf.ln(5)


                # Save & return path
                tmp_path = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf").name
                pdf.output(tmp_path)
                os.remove(logo_path)
                return tmp_path

            importance_df = pd.DataFrame({
                "Feature": importances.index,
                "Importance": importances.values
            }).sort_values(by="Importance", ascending=False)

            pdf_path = generate_clinical_pdf_report(input_data, pred, prob, importance_df)
            with open(pdf_path, "rb") as f:
                st.sidebar.download_button(
                    label="Download PDF Report",
                    data=f,
                    file_name="pcos_clinical_report.pdf",
                    mime="application/pdf"
                )
            os.remove(pdf_path)


        # === Visual Insights ===
        cola, colb = st.columns([2, 3])

        with cola:
            st.markdown("### 🧬 PCOS Distribution")
            if 'has_pcos' in data.columns:
                pcos_count = data['has_pcos'].value_counts().rename({0: 'No PCOS', 1: 'PCOS'})
                fig1, ax1 = plt.subplots()
                ax1.pie(pcos_count, labels=pcos_count.index, autopct='%1.1f%%', startangle=90, colors=['#66b3ff','#ff9999'])
                ax1.axis('equal')
                st.pyplot(fig1)

            st.markdown("### 📈 Feature Correlation Heatmap")
            numeric_data = data.select_dtypes(include=[np.number]).copy()
            if numeric_data.shape[1] > 1:
                corr = numeric_data.corr()
                fig2, ax2 = plt.subplots(figsize=(10, 8))
                sns.heatmap(corr, cmap="coolwarm", center=0, annot=False, fmt=".2f", linewidths=0.5)
                st.pyplot(fig2)

            st.markdown("### 📌 Distribution of Selected Features")
            selected_features = ["bmi", "age", "weight_kg", "waist_size"]
            for feature in selected_features:
                if feature in data.columns:
                    fig3, ax3 = plt.subplots()
                    sns.histplot(data=data, x=feature, hue=data['has_pcos'] if 'has_pcos' in data.columns else None, kde=True, ax=ax3)
                    ax3.set_title(f"Distribution of {feature.title()} by PCOS Status")
                    st.pyplot(fig3)

        with colb:
            st.markdown("### 🧬 PCOS Distribution (Interactive)")
            if 'has_pcos' in data.columns:
                pcos_labels = {0: "No PCOS", 1: "PCOS"}
                pcos_counts = data['has_pcos'].map(pcos_labels).value_counts().reset_index()
                pcos_counts.columns = ['PCOS Status', 'Count']
                fig_pie = px.pie(pcos_counts, names='PCOS Status', values='Count', title="PCOS Class Distribution", hole=0.4)
                st.plotly_chart(fig_pie)

            st.markdown("### 📈 Feature Correlation Heatmap (Interactive)")
            numeric_df = data.select_dtypes(include=[np.number])
            if numeric_df.shape[1] > 1:
                corr_matrix = numeric_df.corr().round(2)
                fig_heatmap = px.imshow(corr_matrix, text_auto=True, aspect="auto", color_continuous_scale="RdBu_r", title="Correlation Between Numeric Features")
                st.plotly_chart(fig_heatmap)

            st.markdown("### 📌 Feature Distributions by PCOS Status (Interactive)")
            for col in selected_features:
                if col in data.columns:
                    fig_hist = px.histogram(data, x=col, color='has_pcos',
                                            barmode='overlay',
                                            color_discrete_map={0: "skyblue", 1: "salmon"},
                                            title=f"{col.title()} Distribution by PCOS Status")
                    fig_hist.update_traces(opacity=0.6)
                    st.plotly_chart(fig_hist)

        # === Advanced Visuals ===
        with cola:
            st.markdown("### 🎯 3D Scatter Plot: Age vs BMI vs Waist Size")
            if all(col in numeric_data.columns for col in ['age', 'bmi', 'waist_size']):
                fig_3d = px.scatter_3d(
                    numeric_data,
                    x='age',
                    y='bmi',
                    z='waist_size',
                    color='has_pcos' if 'has_pcos' in numeric_data.columns else None,
                    title="3D Plot: Age, BMI, Waist Size by PCOS Status",
                    color_discrete_map={0: "skyblue", 1: "salmon"}
                )
                st.plotly_chart(fig_3d)

        with colb:
            st.markdown("### 🔬 Dimensionality Reduction (PCA)")
            if 'has_pcos' in numeric_data.columns and numeric_data.shape[1] > 3:
                features_only = numeric_data.drop(columns=['has_pcos'])
                pca = PCA(n_components=2)
                components = pca.fit_transform(features_only)
                pca_df = pd.DataFrame(components, columns=["PC1", "PC2"])
                pca_df['has_pcos'] = numeric_data['has_pcos'].values

                fig_pca = px.scatter(
                    pca_df,
                    x="PC1",
                    y="PC2",
                    color='has_pcos',
                    title="PCA: Visualizing Clusters by PCOS",
                    color_discrete_map={0: "skyblue", 1: "salmon"}
                )
                st.plotly_chart(fig_pca)



