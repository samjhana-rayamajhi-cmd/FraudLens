
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)

# =========================================================
# FRAUDLENS | PREMIUM FINANCIAL RISK ANALYTICS
# =========================================================

st.set_page_config(
    page_title="FraudLens | Risk Intelligence",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------- DESIGN SYSTEM ----------------

NAVY = "#0B1120"
PANEL = "#111B2D"
CARD = "#17243B"
BORDER = "#2B3D58"
CYAN = "#22D3EE"
TEAL = "#2DD4BF"
GREEN = "#34D399"
RED = "#FB7185"
MUTED = "#94A9C5"
WHITE = "#F1F5F9"
BLUE = "#60A5FA"

st.markdown(
    f"""
    <style>
    @import url(
      'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap'
    );

    html, body, [class*="css"] {{
        font-family: 'Inter', 'Segoe UI', sans-serif;
    }}

    .stApp {{
        background: {NAVY};
        color: {WHITE};
    }}

    [data-testid="stHeader"] {{
        background: rgba(11,17,32,0.94);
    }}

    [data-testid="stSidebar"] {{
        background: {PANEL};
        border-right: 1px solid {BORDER};
    }}

    [data-testid="stSidebar"] > div {{
        padding-top: 1.5rem;
    }}

    .block-container {{
        max-width: 1550px;
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }}

    .brand {{
        display: flex;
        align-items: center;
        gap: 14px;
        margin-bottom: 22px;
    }}

    .brand-icon {{
        width: 54px;
        height: 54px;
        border-radius: 16px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 27px;
        background: linear-gradient(135deg, #164E63, #172554);
        border: 1px solid #28617B;
        box-shadow: 0 0 28px rgba(34,211,238,0.10);
    }}

    .brand-name {{
        font-size: 30px;
        font-weight: 800;
        letter-spacing: -1px;
        color: {WHITE};
        line-height: 1.2;
    }}

    .brand-sub {{
        color: {CYAN};
        font-size: 10px;
        letter-spacing: 2px;
        font-weight: 700;
        margin-top: 5px;
    }}

    .hero {{
        background:
            radial-gradient(
                circle at 90% 5%,
                rgba(34,211,238,0.15),
                transparent 35%
            ),
            linear-gradient(135deg, #172554, #111B2D 72%);
        border: 1px solid #304967;
        border-radius: 20px;
        padding: 30px 32px;
        margin-bottom: 22px;
    }}

    .eyebrow {{
        color: {CYAN};
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 2px;
        margin-bottom: 10px;
    }}

    .hero h1 {{
        color: {WHITE};
        font-size: clamp(25px, 3vw, 36px);
        font-weight: 800;
        letter-spacing: -1.2px;
        margin: 0 0 12px 0;
    }}

    .hero p {{
        color: #C6D5E8;
        font-size: 14px;
        line-height: 1.8;
        margin-bottom: 8px;
    }}

    .hero-foot {{
        color: {MUTED};
        font-size: 10px;
        letter-spacing: 1.2px;
        margin-top: 18px;
    }}

    .section-label {{
        color: {CYAN};
        font-size: 11px;
        letter-spacing: 1.7px;
        font-weight: 700;
        margin: 12px 0 15px 0;
    }}

    div[data-testid="stMetric"] {{
        background: linear-gradient(145deg, {CARD}, {PANEL});
        border: 1px solid {BORDER};
        padding: 19px 20px;
        border-radius: 15px;
        min-height: 115px;
        transition: border-color .2s ease;
    }}

    div[data-testid="stMetric"]:hover {{
        border-color: {CYAN};
    }}

    div[data-testid="stMetricLabel"] {{
        color: {MUTED};
        font-size: 12px;
        font-weight: 500;
    }}

    div[data-testid="stMetricValue"] {{
        color: {WHITE};
        font-weight: 750;
        font-size: 28px;
    }}

    div[data-testid="stMetricDelta"] {{
        font-size: 11px;
    }}

    h1, h2, h3, h4 {{
        color: {WHITE};
        letter-spacing: -0.4px;
    }}

    p, li, label {{
        color: #D6E1EF;
    }}

    button[data-baseweb="tab"] {{
        font-weight: 600;
        padding-top: 12px;
        padding-bottom: 12px;
    }}

    button[data-baseweb="tab"][aria-selected="true"] {{
        color: {CYAN};
    }}

    div[data-baseweb="select"] > div {{
        background: {CARD};
        border-color: {BORDER};
        border-radius: 9px;
    }}

    div[data-testid="stNumberInput"] input {{
        background: {CARD};
        color: {WHITE};
        border-color: {BORDER};
        border-radius: 9px;
    }}

    .stButton > button,
    .stFormSubmitButton > button {{
        background: linear-gradient(90deg, #0891B2, #0D9488);
        color: white;
        border: 1px solid #167D91;
        border-radius: 10px;
        font-weight: 700;
        min-height: 42px;
        transition: all .2s ease;
    }}

    .stButton > button:hover,
    .stFormSubmitButton > button:hover {{
        border-color: {CYAN};
        color: white;
        filter: brightness(1.12);
    }}

    div[data-testid="stDataFrame"] {{
        border: 1px solid {BORDER};
        border-radius: 12px;
        overflow: hidden;
    }}

    div[data-testid="stAlert"] {{
        border-radius: 12px;
    }}

    .status-pill {{
        display: inline-block;
        background: #123C3A;
        border: 1px solid #226B61;
        color: {GREEN};
        padding: 6px 11px;
        border-radius: 20px;
        font-size: 11px;
        font-weight: 700;
    }}

    .footer {{
        border-top: 1px solid {BORDER};
        padding-top: 18px;
        margin-top: 28px;
        color: {MUTED};
        font-size: 11px;
        text-align: center;
        line-height: 1.8;
    }}

    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------- CHART HELPERS ----------------

def style_chart(fig, ax):
    """Apply the premium dark theme to Matplotlib charts."""
    fig.patch.set_facecolor(PANEL)
    ax.set_facecolor(PANEL)

    ax.tick_params(colors=MUTED, labelsize=9)

    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    ax.title.set_color(WHITE)

    for spine in ax.spines.values():
        spine.set_color(BORDER)

    ax.grid(
        axis="y",
        color=BORDER,
        linestyle="--",
        linewidth=0.6,
        alpha=0.7,
    )
    ax.set_axisbelow(True)


def style_confusion_chart(fig, ax):
    fig.patch.set_facecolor(PANEL)
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=WHITE, labelsize=9)

    for spine in ax.spines.values():
        spine.set_color(BORDER)


def metric_card_help():
    st.caption(
        "Model scores are estimates based on the uploaded dataset "
        "and the selected evaluation threshold."
    )


# ---------------- BRANDING ----------------

st.markdown(
    """
    <div class="brand">
        <div class="brand-icon">🔎</div>
        <div>
            <div class="brand-name">FraudLens</div>
            <div class="brand-sub">FRAUD INTELLIGENCE PLATFORM</div>
        </div>
    </div>

    <div class="hero">
        <div class="eyebrow">FINANCIAL RISK ANALYTICS</div>
        <h1>See risk. Understand patterns. Act smarter.</h1>
        <p>
            An explainable machine-learning dashboard for exploring
            financial transaction patterns, evaluating fraud detection
            performance, and estimating transaction risk.
        </p>
        <div class="hero-foot">
            MACHINE LEARNING &nbsp; • &nbsp;
            RISK ANALYTICS &nbsp; • &nbsp;
            EXPLAINABLE AI
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------- SIDEBAR ----------------

st.sidebar.markdown(
    f"""
    <div style="padding: 5px 0 14px 0;">
        <div style="font-size:20px;font-weight:800;color:{WHITE};">
            ⚙️ Workspace
        </div>
        <div style="color:{MUTED};font-size:12px;margin-top:6px;">
            Fraud analytics and monitoring
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

uploaded_file = st.sidebar.file_uploader(
    "Upload transaction dataset",
    type=["csv"],
    help="Upload a CSV with a Fraudulent target column.",
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Expected target**")
st.sidebar.caption(
    "Fraudulent: 0 = legitimate, 1 = fraudulent."
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Model**")
st.sidebar.caption("Random Forest Classifier")
st.sidebar.caption("Stratified train/test split")
st.sidebar.caption("Random state: 42")

st.sidebar.markdown("---")
st.sidebar.markdown(
    f"""
    <div style="color:{MUTED};font-size:11px;line-height:1.8;">
        <b style="color:{WHITE};">FRAUDLENS</b><br>
        Explainable AI Portfolio Project<br>
        Educational prototype • Not production financial advice
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------- EMPTY STATE ----------------

if uploaded_file is None:
    st.markdown(
        f"""
        <div style="
            background:{PANEL};
            border:1px solid {BORDER};
            border-radius:16px;
            padding:28px;
            text-align:center;
        ">
            <div style="font-size:38px;">📂</div>
            <h3 style="margin-bottom:8px;">Your analytics workspace is ready</h3>
            <p style="color:{MUTED};">
                Upload your transaction CSV from the sidebar to
                generate fraud analytics and train the model.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-label">PLATFORM CAPABILITIES</div>',
                unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric("01", "Overview")
        st.caption("Transaction patterns and fraud distribution")

    with c2:
        st.metric("02", "Evaluation")
        st.caption("Precision, recall, F1 and AUC metrics")

    with c3:
        st.metric("03", "Explainability")
        st.caption("Rank important model features")

    with c4:
        st.metric("04", "Risk scoring")
        st.caption("Analyze hypothetical transactions")

    st.stop()


# ---------------- LOAD DATA ----------------

try:
    uploaded_file.seek(0)
    df = pd.read_csv(uploaded_file)
except Exception as exc:
    st.error(f"Unable to read CSV: {exc}")
    st.stop()

df.columns = df.columns.astype(str).str.strip()

if df.empty:
    st.error("The uploaded dataset is empty.")
    st.stop()

if "Fraudulent" not in df.columns:
    st.error(
        "Required column 'Fraudulent' was not found. "
        "Please upload the correct dataset."
    )
    st.stop()

original_rows = len(df)
df = df.drop_duplicates().copy()
duplicates_removed = original_rows - len(df)


# ---------------- CLEAN TARGET ----------------

target = df["Fraudulent"]

if target.dtype == "object":
    normalized = target.astype(str).str.strip().str.lower()

    mapping = {
        "0": 0,
        "1": 1,
        "false": 0,
        "true": 1,
        "no": 0,
        "yes": 1,
        "legitimate": 0,
        "fraud": 1,
        "fraudulent": 1,
        "not fraud": 0,
        "genuine": 0,
    }

    y = normalized.map(mapping)
else:
    y = pd.to_numeric(target, errors="coerce")

valid_rows = y.notna()
df = df.loc[valid_rows].copy()
y = y.loc[valid_rows].astype(int)

if len(df) < 20:
    st.error("Please upload at least 20 labeled transactions.")
    st.stop()

if not set(y.unique()).issubset({0, 1}) or set(y.unique()) != {0, 1}:
    st.error(
        "Fraudulent must contain both 0 and 1, "
        "where 1 represents fraud."
    )
    st.stop()

if y.value_counts().min() < 2:
    st.error("Each class needs at least two records.")
    st.stop()


# ---------------- FEATURE ENGINEERING ----------------

if "Transaction_Date" in df.columns:
    dates = pd.to_datetime(
        df["Transaction_Date"],
        format="%d-%m-%Y %H:%M",
        errors="coerce",
    )

    if dates.notna().sum() == 0:
        dates = pd.to_datetime(
            df["Transaction_Date"],
            errors="coerce",
        )

    df["Transaction_Hour"] = dates.dt.hour
    df["Transaction_Day"] = dates.dt.dayofweek

# Avoid using identifiers or potentially leaky features.
excluded = {
    "Fraudulent",
    "Transaction_ID",
    "Customer_ID",
    "Transaction_Date",
    "Suspicious_Keyword",
}

X = df[
    [c for c in df.columns if c not in excluded]
].copy()

X = X.dropna(axis=1, how="all")

if X.empty:
    st.error("No usable features remain in this dataset.")
    st.stop()

for col in X.columns:
    if X[col].dtype == "object":
        numeric_version = pd.to_numeric(X[col], errors="coerce")

        if numeric_version.notna().mean() > 0.90:
            X[col] = numeric_version

numeric_features = X.select_dtypes(
    include=["number", "bool"]
).columns.tolist()

categorical_features = [
    c for c in X.columns if c not in numeric_features
]

for col in numeric_features:
    if X[col].dtype == "bool":
        X[col] = X[col].astype(int)


# ---------------- MODEL TRAINING ----------------

try:
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=42,
    )

    transformers = []

    if numeric_features:
        numeric_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
            ]
        )
        transformers.append(
            ("numeric", numeric_pipe, numeric_features)
        )

    if categorical_features:
        categorical_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "encoder",
                    OneHotEncoder(handle_unknown="ignore"),
                ),
            ]
        )
        transformers.append(
            ("categorical", categorical_pipe, categorical_features)
        )

    preprocessor = ColumnTransformer(
        transformers=transformers
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=200,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)
    probabilities = model.predict_proba(X_test)[:, 1]

except Exception as exc:
    st.error(f"Model training failed: {exc}")
    st.info(
        "Check the CSV data types, feature columns, "
        "and whether both target classes have enough examples."
    )
    st.stop()


# ---------------- NAVIGATION ----------------

overview_tab, evaluation_tab, importance_tab, scoring_tab = st.tabs(
    [
        "📊  Overview",
        "📈  Model Evaluation",
        "🔍  Feature Importance",
        "🧪  Transaction Risk",
    ]
)


# =========================================================
# OVERVIEW
# =========================================================

fraud_count = int((y == 1).sum())
legitimate_count = int((y == 0).sum())
fraud_rate = float(y.mean() * 100)

with overview_tab:
    st.markdown(
        '<div class="section-label">EXECUTIVE OVERVIEW</div>',
        unsafe_allow_html=True,
    )

    st.markdown("### Transaction intelligence")
    st.caption(
        "A high-level view of the uploaded dataset and its fraud distribution."
    )

    a, b, c, d = st.columns(4)

    a.metric(
        "Total Transactions",
        f"{len(df):,}",
        help="Rows remaining after duplicate and target cleaning.",
    )

    b.metric(
        "Fraud Cases",
        f"{fraud_count:,}",
        delta=f"{fraud_rate:.2f}% of data",
        delta_color="inverse",
    )

    c.metric(
        "Legitimate Cases",
        f"{legitimate_count:,}",
    )

    d.metric(
        "Fraud Rate",
        f"{fraud_rate:.2f}%",
        help="Percentage of transactions labeled fraudulent.",
    )

    st.markdown('<div class="section-label">RISK DISTRIBUTION</div>',
                unsafe_allow_html=True)

    left, right = st.columns([1, 1.35], gap="large")

    with left:
        st.markdown("#### Fraud vs. legitimate")

        fig, ax = plt.subplots(figsize=(6, 4))

        bars = ax.bar(
            ["Legitimate", "Fraudulent"],
            [legitimate_count, fraud_count],
            color=[TEAL, RED],
            width=0.55,
        )

        ax.set_ylabel("Number of transactions")
        ax.set_title("Transaction label distribution", loc="left")
        style_chart(fig, ax)

        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{int(bar.get_height()):,}",
                ha="center",
                va="bottom",
                color=WHITE,
                fontsize=10,
                fontweight="bold",
            )

        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with right:
        st.markdown("#### Fraud rate by merchant")

        if "Merchant_Category" in df.columns:
            merchant_df = pd.DataFrame({
                "Merchant": df["Merchant_Category"].astype(str),
                "Fraudulent": y.to_numpy(),
            })

            merchant_rates = (
                merchant_df.groupby("Merchant")["Fraudulent"]
                .mean()
                .mul(100)
                .sort_values(ascending=True)
            )

            fig, ax = plt.subplots(figsize=(7, 4))

            ax.barh(
                merchant_rates.index,
                merchant_rates.values,
                color=CYAN,
                height=0.65,
            )

            ax.set_xlabel("Fraud rate (%)")
            ax.set_title("Fraud rate by merchant category", loc="left")
            style_chart(fig, ax)

            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        else:
            st.info("Merchant_Category is not available.")

    st.markdown('<div class="section-label">DATA QUALITY</div>',
                unsafe_allow_html=True)

    q1, q2, q3, q4 = st.columns(4)

    q1.metric("Model Features", len(X.columns))
    q2.metric("Training Records", f"{len(X_train):,}")
    q3.metric("Test Records", f"{len(X_test):,}")
    q4.metric("Duplicates Removed", f"{duplicates_removed:,}")

    with st.expander("Preview cleaned dataset"):
        preview = df.copy()
        preview["Fraudulent"] = y
        st.dataframe(
            preview.head(20),
            use_container_width=True,
        )

    with st.expander("Review missing values"):
        missing = df.isna().sum()
        missing_df = (
            missing.rename("Missing Values")
            .to_frame()
            .sort_values("Missing Values", ascending=False)
        )
        st.dataframe(missing_df, use_container_width=True)


# =========================================================
# MODEL EVALUATION
# =========================================================

with evaluation_tab:
    st.markdown(
        '<div class="section-label">MODEL PERFORMANCE</div>',
        unsafe_allow_html=True,
    )

    st.markdown("### Evaluation & threshold tuning")
    st.caption(
        "Adjust the threshold to explore the trade-off between "
        "detecting fraud and generating false alarms."
    )

    threshold = st.slider(
        "Fraud probability threshold",
        min_value=0.05,
        max_value=0.95,
        value=0.10,
        step=0.05,
        key="evaluation_threshold",
    )

    predictions = (probabilities >= threshold).astype(int)

    precision = precision_score(
        y_test, predictions, zero_division=0
    )
    recall = recall_score(
        y_test, predictions, zero_division=0
    )
    f1 = f1_score(
        y_test, predictions, zero_division=0
    )
    accuracy = accuracy_score(y_test, predictions)

    pr_auc = average_precision_score(y_test, probabilities)

    try:
        roc_auc = roc_auc_score(y_test, probabilities)
    except ValueError:
        roc_auc = np.nan

    m1, m2, m3, m4 = st.columns(4)

    m1.metric("Precision", f"{precision:.3f}")
    m2.metric("Recall", f"{recall:.3f}")
    m3.metric("F1 Score", f"{f1:.3f}")
    m4.metric("PR-AUC", f"{pr_auc:.3f}")

    m5, m6 = st.columns(2)

    m5.metric("Accuracy", f"{accuracy:.3f}")
    m6.metric(
        "ROC-AUC",
        f"{roc_auc:.3f}" if np.isfinite(roc_auc) else "N/A",
    )

    st.caption(
        "These metrics use the held-out test set. "
        "The threshold changes predictions, not the trained model."
    )

    left, right = st.columns([1, 1.15], gap="large")

    with left:
        st.markdown("#### Confusion matrix")

        cm = confusion_matrix(
            y_test,
            predictions,
            labels=[0, 1],
        )

        fig, ax = plt.subplots(figsize=(5.5, 4.4))

        image = ax.imshow(
            cm,
            cmap="YlGn",
        )

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Legitimate", "Fraud"], rotation=10)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Legitimate", "Fraud"])

        ax.set_xlabel("Predicted class")
        ax.set_ylabel("Actual class")
        ax.set_title("Actual vs. predicted", loc="left")

        for i in range(2):
            for j in range(2):
                value = cm[i, j]
                ax.text(
                    j,
                    i,
                    f"{value:,}",
                    ha="center",
                    va="center",
                    fontsize=14,
                    fontweight="bold",
                    color="#0B1120",
                )

        style_confusion_chart(fig, ax)
        fig.colorbar(image, ax=ax, fraction=0.045)
        fig.tight_layout()

        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with right:
        st.markdown("#### Classification report")

        report = classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["Legitimate", "Fraudulent"],
            output_dict=True,
            zero_division=0,
        )

        report_df = pd.DataFrame(report).transpose().round(3)

        st.dataframe(
            report_df,
            use_container_width=True,
        )

        st.download_button(
            "⬇ Download evaluation report",
            data=report_df.to_csv().encode("utf-8"),
            file_name="fraudlens_evaluation_report.csv",
            mime="text/csv",
            use_container_width=True,
        )

    tn, fp, fn, tp = cm.ravel()

    st.markdown('<div class="section-label">DETECTION SUMMARY</div>',
                unsafe_allow_html=True)

    r1, r2, r3, r4 = st.columns(4)

    r1.metric("True Positives", f"{tp:,}")
    r2.metric("False Positives", f"{fp:,}")
    r3.metric("True Negatives", f"{tn:,}")
    r4.metric("False Negatives", f"{fn:,}")

    st.info(
        "High recall means the model catches more fraud cases. "
        "Low precision means more flagged transactions may be legitimate. "
        "Thresholds should be chosen based on operational costs and validation."
    )


# =========================================================
# FEATURE IMPORTANCE
# =========================================================

with importance_tab:
    st.markdown(
        '<div class="section-label">MODEL EXPLAINABILITY</div>',
        unsafe_allow_html=True,
    )

    st.markdown("### What influences the model?")
    st.caption(
        "Feature importance ranks which encoded input features "
        "the Random Forest uses most when making predictions."
    )

    try:
        fitted_preprocessor = model.named_steps["preprocessor"]
        classifier = model.named_steps["classifier"]

        transformed_names = (
            fitted_preprocessor.get_feature_names_out()
        )

        importances = classifier.feature_importances_

        importance_df = pd.DataFrame({
            "Feature": transformed_names,
            "Importance": importances,
        }).sort_values(
            "Importance",
            ascending=False,
        )

        max_features = len(importance_df)

        if max_features == 0:
            st.info("No feature importances are available.")
        else:
            top_n = st.slider(
                "Features to display",
                min_value=1,
                max_value=min(30, max_features),
                value=min(15, max_features),
            )

            top_features = importance_df.head(top_n).sort_values(
                "Importance",
                ascending=True,
            )

            fig, ax = plt.subplots(
                figsize=(10, max(4, top_n * 0.34))
            )

            ax.barh(
                top_features["Feature"],
                top_features["Importance"],
                color=CYAN,
                height=0.66,
            )

            ax.set_xlabel("Relative importance")
            ax.set_title("Random Forest feature ranking", loc="left")
            style_chart(fig, ax)

            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

            st.markdown("#### Feature ranking table")

            display_df = importance_df.head(top_n).copy()
            display_df["Importance"] = display_df["Importance"].round(5)

            st.dataframe(
                display_df.reset_index(drop=True),
                use_container_width=True,
            )

            st.download_button(
                "⬇ Download feature importance",
                data=importance_df.to_csv(index=False).encode("utf-8"),
                file_name="fraudlens_feature_importance.csv",
                mime="text/csv",
            )

            st.caption(
                "Importance describes model behavior; it does not prove "
                "causation. Correlated features can share importance."
            )

    except Exception as exc:
        st.warning(f"Could not display feature importance: {exc}")


# =========================================================
# TRANSACTION RISK
# =========================================================

with scoring_tab:
    st.markdown(
        '<div class="section-label">TRANSACTION INVESTIGATION</div>',
        unsafe_allow_html=True,
    )

    st.markdown("### Analyze a hypothetical transaction")
    st.caption(
        "Enter transaction details to estimate a fraud probability "
        "with the currently trained model."
    )

    st.warning(
        "This is an experimental model score, not a confirmed fraud "
        "determination. It must not be used as the sole basis for "
        "real financial decisions."
    )

    with st.form("risk_form"):
        input_values = {}

        left, right = st.columns(2, gap="large")

        for index, col in enumerate(X.columns):
            container = left if index % 2 == 0 else right

            with container:
                if col in numeric_features:
                    values = pd.to_numeric(
                        X[col], errors="coerce"
                    ).dropna()

                    default = (
                        float(values.median())
                        if not values.empty
                        else 0.0
                    )

                    low = (
                        float(values.min())
                        if not values.empty
                        else 0.0
                    )

                    high = (
                        float(values.max())
                        if not values.empty
                        else max(1.0, default)
                    )

                    if low == high:
                        input_values[col] = st.number_input(
                            col.replace("_", " "),
                            value=default,
                            key=f"risk_{col}",
                        )
                    else:
                        input_values[col] = st.number_input(
                            col.replace("_", " "),
                            min_value=low,
                            max_value=high,
                            value=min(max(default, low), high),
                            key=f"risk_{col}",
                        )

                else:
                    options = (
                        X[col]
                        .dropna()
                        .astype(str)
                        .value_counts()
                        .head(50)
                        .index
                        .tolist()
                    )

                    if not options:
                        options = ["Unknown"]

                    input_values[col] = st.selectbox(
                        col.replace("_", " "),
                        options=options,
                        key=f"risk_{col}",
                    )

        submitted = st.form_submit_button(
            "🔎 Analyze Transaction",
            use_container_width=True,
        )

    if submitted:
        transaction = pd.DataFrame(
            [input_values],
            columns=X.columns,
        )

        for col in numeric_features:
            transaction[col] = pd.to_numeric(
                transaction[col],
                errors="coerce",
            )

        try:
            fraud_probability = float(
                model.predict_proba(transaction)[0, 1]
            )

            threshold_value = st.session_state.get(
                "evaluation_threshold",
                0.10,
            )

            is_flagged = fraud_probability >= threshold_value

            st.markdown("---")
            st.markdown("#### Risk assessment")

            a, b = st.columns(2)

            a.metric(
                "Estimated Fraud Probability",
                f"{fraud_probability * 100:.2f}%",
            )

            b.metric(
                "Selected Threshold",
                f"{threshold_value * 100:.0f}%",
            )

            st.progress(
                min(1.0, max(0.0, fraud_probability)),
                text="Model-estimated fraud probability",
            )

            if is_flagged:
                st.error(
                    "🚨 FLAGGED FOR REVIEW — the estimated probability "
                    "meets or exceeds the selected threshold."
                )
            else:
                st.success(
                    "✓ BELOW THE SELECTED THRESHOLD — the transaction "
                    "was not flagged by this model setting."
                )

            st.caption(
                "A low score does not guarantee safety, and a high score "
                "does not establish fraud. Human review and further "
                "validation are required."
            )

        except Exception as exc:
            st.error(f"Could not analyze transaction: {exc}")


# ---------------- FOOTER ----------------

st.markdown(
    """
    <div class="footer">
        <b>FraudLens</b> · Explainable Financial Fraud Detection<br>
        Built with Streamlit and scikit-learn · Educational portfolio prototype<br>
        Model outputs are estimates and require validation before real-world use.
    </div>
    """,
    unsafe_allow_html=True,
)
