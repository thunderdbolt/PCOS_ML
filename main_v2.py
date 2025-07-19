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
import plotly.graph_objects as go
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

# data = pd.read_csv(uploaded_file)
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

@st.cache_data(show_spinner=False)
def run_selected_combos(combo_matrix, selected_combos, features, target):
    from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, accuracy_score
    results = []
    model_curves = {}
    feature_importances_all = {}

    cv_map = {
        "StratifiedKFold": StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        "RepeatedStratifiedKFold": RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=42),
        "KFold": KFold(n_splits=5, shuffle=True, random_state=42)
    }

    for combo_id in selected_combos:
        config = combo_matrix[combo_matrix["Combo ID"] == combo_id].iloc[0]

        X_train, X_test, y_train, y_test = train_test_split(
            features, target,
            test_size=config["Split"], stratify=target, random_state=42
        )

        cv = cv_map[config["CV"]]

        # Model & param setup
        if config["Model"] == "Random Forest":
            model = RandomForestClassifier(class_weight="balanced", random_state=42)
            tuned_params = {
                "n_estimators": [100, 150, 200],
                "max_depth": [None, 10, 20],
                "min_samples_split": [2, 5],
                "min_samples_leaf": [1, 2, 4]
            }
        elif config["Model"] == "Logistic Regression":
            model = LogisticRegression(max_iter=1000, solver="liblinear", class_weight="balanced")
            tuned_params = {
                "C": [0.01, 0.1, 1.0, 10.0],
                "penalty": ["l2"]
            }
        elif config["Model"] == "SVM":
            model = SVC(probability=True, class_weight="balanced")
            tuned_params = {
                "C": [0.1, 1.0, 10.0],
                "gamma": ["scale", "auto"],
                "kernel": ["rbf", "linear", "poly"]
            }

        if config["Search"] == "Grid":
            search = GridSearchCV(model, tuned_params, cv=cv, scoring="roc_auc", n_jobs=-1, error_score=np.nan)
        else:
            search = RandomizedSearchCV(model, tuned_params, cv=cv, n_iter=10, random_state=42, scoring="roc_auc", n_jobs=-1, error_score=np.nan)

        try:
            search.fit(X_train, y_train)
            best_model = search.best_estimator_
            y_pred = best_model.predict(X_test)
            y_prob = best_model.predict_proba(X_test)[:, 1]
            acc = accuracy_score(y_test, y_pred)
            roc = roc_auc_score(y_test, y_prob)
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
            sensitivity = tp / (tp + fn) if (tp + fn) else 0
            specificity = tn / (tn + fp) if (tn + fp) else 0
            confidence_avg = np.mean(np.max(best_model.predict_proba(X_test), axis=1))
            warning = "⚠️" if config["Split"] < 0.2 else ""
            params = search.best_params_

            results.append({
                "Combo ID": f"{combo_id} {warning}",
                "Accuracy": round(acc, 2),
                "ROC-AUC": round(roc, 2),
                "Sensitivity": round(sensitivity, 2),
                "Specificity": round(specificity, 2),
                "Confidence": round(confidence_avg, 2),
                "Search Method": config["Search"],
                "Model": config["Model"],
                "n_estimators": params.get("n_estimators", "N/A"),
                "max_depth": params.get("max_depth", "N/A"),
                "min_samples_split": params.get("min_samples_split", "N/A"),
                "C": params.get("C", "N/A"),
                "gamma": params.get("gamma", "N/A"),
                "Best Params": params
            })

            fpr, tpr, _ = roc_curve(y_test, y_prob)
            model_curves[combo_id] = (fpr, tpr, config["Model"])

            if config["Model"] == "Random Forest":
                feature_importances_all[combo_id] = best_model.feature_importances_

        except Exception as e:
            results.append({
                "Combo ID": combo_id,
                "Accuracy": None,
                "ROC-AUC": None,
                "Sensitivity": None,
                "Specificity": None,
                "Confidence": None,
                "Search Method": config["Search"],
                "Model": config["Model"],
                "n_estimators": "N/A",
                "max_depth": "N/A",
                "min_samples_split": "N/A",
                "C": "N/A",
                "gamma": "N/A",
                "Best Params": str(e)
            })

    return results, model_curves, feature_importances_all

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

    # Placeholder: Define features and target
    target_col = "has_pcos"
    features = data.drop(columns=[target_col])
    target = data[target_col]

    # Combo Matrix Definition with editable options
    default_data = pd.DataFrame([
        {"Combo ID": "C1", "Split": 0.2, "CV": "StratifiedKFold", "Model": "Random Forest", "Search": "Grid"},
        {"Combo ID": "C2", "Split": 0.3, "CV": "RepeatedStratifiedKFold", "Model": "Random Forest", "Search": "Random"},
        {"Combo ID": "C3", "Split": 0.2, "CV": "StratifiedKFold", "Model": "Logistic Regression", "Search": "Grid"},
        {"Combo ID": "C4", "Split": 0.4, "CV": "KFold", "Model": "Logistic Regression", "Search": "Random"},
        {"Combo ID": "C5", "Split": 0.25, "CV": "StratifiedKFold", "Model": "SVM", "Search": "Grid"},
        {"Combo ID": "C6", "Split": 0.3, "CV": "RepeatedStratifiedKFold", "Model": "SVM", "Search": "Random"},
        {"Combo ID": "C7", "Split": 0.5, "CV": "StratifiedKFold", "Model": "Random Forest", "Search": "Grid"},
        {"Combo ID": "C8", "Split": 0.15, "CV": "StratifiedKFold", "Model": "Logistic Regression", "Search": "Grid"},
        {"Combo ID": "C9", "Split": 0.2, "CV": "KFold", "Model": "SVM", "Search": "Grid"},
        {"Combo ID": "C10", "Split": 0.4, "CV": "StratifiedKFold", "Model": "Random Forest", "Search": "Grid"}
    ])

    edit_types = {
        "CV": st.column_config.SelectboxColumn(
            "Cross-Validation", options=["StratifiedKFold", "RepeatedStratifiedKFold", "KFold"]
        ),
        "Model": st.column_config.SelectboxColumn(
            "Model", options=["Random Forest", "Logistic Regression", "SVM"]
        ),
        "Search": st.column_config.SelectboxColumn(
            "Search Method", options=["Grid", "Random"]
        )
    }

    st.markdown("### 📋 Combo Configuration Matrix")
    combo_matrix = st.data_editor(default_data, column_config=edit_types, use_container_width=False)

    with st.expander("🧠 Methodology Note", expanded=False):
        st.markdown("""
        #### 🧠 Methodology Note
        In our sensitivity analysis, we optimized key hyperparameters for each model to balance performance, interpretability, and runtime feasibility.

        - **🔹 SVM (Support Vector Machine)**  
            - **Dynamic (Tuned)**: `C`, `gamma`, `kernel` (tested with `'rbf'`, `'linear'`, `'poly'`)  
            - **Fixed**:  
                - `probability=True`: allows probability-based predictions for ROC analysis  
                - `class_weight='balanced'`: ensures fairness between majority and minority classes

        - **🌲 Random Forest**  
            - **Dynamic (Tuned)**: `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`  
            - **Fixed**:  
                - `class_weight='balanced'`: adjusts weights inversely proportional to class frequencies  
                - `max_features='sqrt'`: kept constant for consistent feature sampling per tree  
                - `bootstrap=True`: standard setting for sampling with replacement

        - **📈 Logistic Regression**  
            - **Dynamic (Tuned)**: `C`, `penalty='l2'`  
            - **Fixed**:  
                - `solver='liblinear'`: compatible with small datasets and `l1/l2` penalties  
                - `class_weight='balanced'`: ensures model doesn't ignore minority class  
                - `penalty='l1'`: not included yet due to complexity of coefficient shrinkage; can be added later

        > 📌 In future experiments, we may include additional hyperparameters (e.g., `max_features` for Random Forest or `'l1'` regularization in Logistic Regression) to extract further insights.
        """)

    selected_combos = st.multiselect("🎛️ Select Combo IDs to Run", options=combo_matrix["Combo ID"].tolist())
    results = []
    model_curves = {}

    # Define a key for the button
    run_key = "run_combos_clicked"

    # Initialize the state if not already set
    if run_key not in st.session_state:
        st.session_state[run_key] = False

    # Create the button and update the session state
    if st.button("🚀 Run Selected Combos"):
        with st.spinner("Running selected combos..."):
            results, model_curves, feature_importances_all = run_selected_combos(combo_matrix, selected_combos, features, target)

        results_df = pd.DataFrame(results)
        st.markdown("### 📊 Results Summary")
        st.dataframe(results_df)

        if not results_df.empty:
            st.markdown("### 📈 Combo Performance (ROC-AUC Area + Bar Plot)")
            plot_df = results_df.dropna(subset=["ROC-AUC"])
            fig = go.Figure()
            fig.add_trace(go.Bar(x=plot_df["Combo ID"], y=plot_df["ROC-AUC"], name="ROC-AUC", marker_color='indianred'))
            fig.add_trace(go.Scatter(x=plot_df["Combo ID"], y=plot_df["ROC-AUC"], fill='tozeroy', name="Area Under Curve", mode="lines", line=dict(color="lightblue")))
            fig.update_layout(title="ROC-AUC Scores per Combo", xaxis_title="Combo ID", yaxis_title="ROC-AUC", height=400)
            st.plotly_chart(fig)

            if feature_importances_all:
                st.markdown("### 🌲 Feature Importances (Random Forest Only)")
                cols = st.columns(len(feature_importances_all))
                for idx, (combo_id, importances) in enumerate(feature_importances_all.items()):
                    fi_df = pd.DataFrame({"Feature": features.columns, "Importance": importances}).sort_values("Importance", ascending=False)
                    fig = px.bar(fi_df.head(10), x="Importance", y="Feature", orientation="h", title=f"{combo_id} – Top Features")
                    cols[idx].plotly_chart(fig, use_container_width=True)

            col07, col08 = st.columns([3, 2])

            if model_curves:
                # st.markdown("### 🧪 ROC Curve per Model")
                # for combo_id, (fpr, tpr, model_type) in model_curves.items():
                #     fig = go.Figure()
                #     fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', fill='tozeroy', name=f"{combo_id} ({model_type})"))
                #     fig.update_layout(title=f"ROC Curve for {combo_id} – {model_type}", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
                #     st.plotly_chart(fig, use_container_width=True)

                # st.markdown("### 📊 ROC Overlay: All Models")
                fig = go.Figure()
                color_map = {"Random Forest": "blue", "Logistic Regression": "red", "SVM": "green"}
                for combo_id, (fpr, tpr, model_type) in model_curves.items():
                    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f"{combo_id} ({model_type})", fill='tonexty', line=dict(color=color_map.get(model_type, None))))
                fig.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
                with col07:
                    st.markdown("### 📊 ROC Overlay: All Models")
                    st.plotly_chart(fig, use_container_width=True)
            
            with col08:
                st.markdown("### 📋 Final Evaluation Table")
                st.dataframe(results_df[["Combo ID", "Accuracy", "Sensitivity", "Specificity", "ROC-AUC"]], use_container_width=True)


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

    with st.expander("Show Model Insight", expanded=True):
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
