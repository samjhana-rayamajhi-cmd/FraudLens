
import os
import html
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

st.set_page_config(
    page_title="FraudLens | Financial Risk Analytics",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------- PREMIUM DARK THEME ----------------
st.markdown("""
<style>
:root {
    --bg:#07111f;
    --panel:#101f32;
    --line:#29435e;
    --text:#f3f7ff;
    --muted:#a8bdd5;
    --teal:#22e6c4;
    --cyan:#29c9ff;
    --amber:#ffb454;
    --red:#ff667a;
}
html, body, [data-testid="stAppViewContainer"] {
    background:#07111f;
    color:#f3f7ff;
}
[data-testid="stHeader"] {background:rgba(7,17,31,.95);}
[data-testid="stSidebar"] {
    background:#0b1728;
    border-right:1px solid #29435e;
}
[data-testid="stSidebar"] * {color:#f3f7ff;}
.block-container {
    max-width:1500px;
    padding-top:1rem;
    padding-bottom:2rem;
}
h1,h2,h3 {color:#f3f7ff !important; letter-spacing:-.025em;}
p,label {color:#d4e1ee;}
[data-testid="stMetric"] {
    background:linear-gradient(145deg,#142941,#0c192a);
    border:1px solid #29435e;
    padding:13px 16px;
    border-radius:13px;
    min-height:100px;
}
[data-testid="stMetricLabel"] {color:#a8bdd5 !important;}
[data-testid="stMetricValue"] {color:#f3f7ff !important;}
div[data-testid="stTabs"] button {
    color:#c6d7e9;
    border-radius:8px 8px 0 0;
}
div[data-testid="stTabs"] button[aria-selected="true"] {
    color:#22e6c4;
    border-bottom-color:#22e6c4;
}
.stButton button,.stDownloadButton button {
    border:1px solid #22bba9;
    background:linear-gradient(100deg,#08786f,#075b78);
    color:white;
    border-radius:9px;
    font-weight:700;
}
.stButton button:hover,.stDownloadButton button:hover {
    border-color:#22e6c4;
    color:white;
}
.hero {
    background:
      radial-gradient(ellipse at 80% 35%,rgba(0,203,255,.16),transparent 40%),
      linear-gradient(120deg,#03101f,#0a2035 60%,#061323);
    border:1px solid #245276;
    border-radius:16px;
    padding:22px 25px 17px;
    margin-bottom:18px;
    overflow:hidden;
}
.hero-row {display:flex;align-items:center;gap:15px;}
.hero-icon {
    width:52px;height:52px;flex-shrink:0;
    display:flex;align-items:center;justify-content:center;
    font-size:28px;border-radius:13px;
    background:linear-gradient(145deg,#08796e,#073957);
    border:1px solid #22e6c4;
    box-shadow:0 0 20px #22e6c422;
}
.brand {
    font-size:clamp(1.8rem,4vw,2.8rem);
    font-weight:900;letter-spacing:.04em;line-height:1.1;
    color:#f3f7ff;
}
.brand span {color:#22e6c4;}
.eyebrow {
    font-size:.68rem;letter-spacing:.22em;font-weight:700;
    color:#91b7d5;margin-top:5px;
}
.hero-sub {color:#b4c7db;margin:13px 0 0;font-size:.92rem;}
.ticker {
    margin-top:17px;overflow:hidden;white-space:nowrap;
    border:1px solid #1687a6;border-radius:50px;
    background:#03101fe8;padding:10px 0;
    box-shadow:inset 0 0 18px #00d9ff10;
}
.ticker-track {
    display:inline-flex;align-items:center;gap:28px;
    width:max-content;padding-left:100%;
    animation:scroll-ticker 20s linear infinite;
    font-weight:800;font-size:clamp(.85rem,1.5vw,1.05rem);
}
.ticker:hover .ticker-track {animation-play-state:paused;}
.ticker .cyan {color:#29c9ff;}
.ticker .teal {color:#22e6c4;}
.ticker .amber {color:#ffb454;}
@keyframes scroll-ticker {
    from {transform:translateX(0);}
    to {transform:translateX(-100%);}
}
.kicker {
    font-size:.7rem;letter-spacing:.18em;font-weight:800;
    color:#22e6c4;margin:5px 0 9px;
}
.panel {
    background:linear-gradient(145deg,#12243a,#0b192a);
    border:1px solid #29435e;border-radius:13px;padding:17px;
}
.muted {color:#a8bdd5;font-size:.85rem;}
[data-testid="stDataFrame"] {
    border:1px solid #29435e;border-radius:10px;overflow:hidden;
}
@media(max-width:650px) {
    .block-container {padding:.7rem .65rem 1.5rem;}
    .hero {padding:16px 14px;}
    .hero-icon {width:43px;height:43px;font-size:23px;}
    .ticker-track {gap:19px;}
}
@media(prefers-reduced-motion:reduce) {
    .ticker-track {animation:none;padding-left:0;}
}
</style>
""", unsafe_allow_html=True)

# ---------------- HERO BANNER ----------------
banner = "fraudlens_banner.png"

if os.path.exists(banner):
    st.image(banner, use_container_width=True)
    st.markdown("""
    <div class="ticker">
      <div class="ticker-track">
        <span class="cyan">⌁ SEE RISK.</span>
        <span>◆</span>
        <span class="teal">⌕ UNDERSTAND PATTERNS.</span>
        <span>◆</span>
        <span class="amber">✦ ACT SMARTER.</span>
        <span>◆</span>
        <span class="cyan">⌁ SEE RISK.</span>
        <span>◆</span>
        <span class="teal">⌕ UNDERSTAND PATTERNS.</span>
        <span>◆</span>
        <span class="amber">✦ ACT SMARTER.</span>
      </div>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <section class="hero">
      <div class="hero-row">
        <div class="hero-icon">🛡️</div>
        <div>
          <div class="brand">FRAUD<span>LENS</span></div>
          <div class="eyebrow">FINANCIAL RISK ANALYTICS</div>
        </div>
      </div>
      <p class="hero-sub">
        Explainable machine learning for financial transaction monitoring,
        fraud pattern discovery and risk analysis.
      </p>
      <div class="ticker">
        <div class="ticker-track">
          <span class="cyan">⌁ SEE RISK.</span><span>◆</span>
          <span class="teal">⌕ UNDERSTAND PATTERNS.</span><span>◆</span>
          <span class="amber">✦ ACT SMARTER.</span><span>◆</span>
          <span class="cyan">⌁ SEE RISK.</span><span>◆</span>
          <span class="teal">⌕ UNDERSTAND PATTERNS.</span><span>◆</span>
          <span class="amber">✦ ACT SMARTER.</span>
        </div>
      </div>
    </section>
    """, unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
st.sidebar.markdown("## 🛡️ FRAUDLENS")
st.sidebar.caption("FINANCIAL SECURITY • RISK MONITORING")
st.sidebar.divider()

uploaded = st.sidebar.file_uploader(
    "Upload transaction dataset",
    type=["csv"],
    help="CSV must contain a Fraudulent target column."
)

st.sidebar.markdown("**Target column**")
st.sidebar.caption("Fraudulent: 0 = legitimate, 1 = fraud")
st.sidebar.markdown("**Model**")
st.sidebar.caption("Random Forest Classifier")
st.sidebar.divider()
st.sidebar.caption("Educational prototype — not a production fraud-control system.")

# ---------------- EMPTY STATE ----------------

# ---------------- EMPTY STATE ----------------
if uploaded is None:
    st.markdown("""
    <div class="panel" style="
        text-align:center;
        padding:22px 18px;
        margin-bottom:20px;
        background:linear-gradient(135deg,#142941,#0b192a);
        border-color:#31516f;
    ">
      <div style="font-size:2rem;margin-bottom:5px;">📂</div>
      <h2 style="margin:5px 0 8px;">Your analytics workspace is ready</h2>
      <p class="muted" style="margin:0;">
        Upload your transaction CSV from the sidebar to train the model
        and explore fraud analytics.
      </p>
    </div>
    """, unsafe_allow_html=True)

    items = [
        ("01", "Overview",
         "Explore transaction patterns and fraud distribution."),
        ("02", "Evaluation",
         "Review precision, recall, F1 score and ROC-AUC."),
        ("03", "Explainability",
         "Discover which features influence model predictions."),
        ("04", "Risk scoring",
         "Estimate risk for hypothetical transactions.")
    ]

    # Four equal-width, equal-height cards
    cards = st.columns(4, gap="medium")

    for col, (number, title, description) in zip(cards, items):
        with col:
            st.markdown(f"""
            <div class="feature-card">
                <div class="feature-number">
                    {number} / FRAUDLENS
                </div>
                <div class="feature-title">{title}</div>
                <div class="feature-description">{description}</div>
            </div>
            """, unsafe_allow_html=True)

            # Clickable button that opens the selected feature details
            if st.button(
                f"Explore {title} ↗",
                key=f"explore_{number}",
                use_container_width=True
            ):
                st.session_state["selected_feature"] = title

    # This information appears below the four cards
    selected = st.session_state.get("selected_feature")

    if selected:
        details = {
            "Overview": (
                "Overview",
                "After uploading the CSV, explore transaction counts, "
                "fraud distribution, dataset quality and merchant categories."
            ),
            "Evaluation": (
                "Model evaluation",
                "After uploading the CSV, review accuracy, precision, "
                "recall, F1 score, ROC-AUC and the confusion matrix."
            ),
            "Explainability": (
                "Feature importance",
                "After uploading the CSV, inspect and download the "
                "features ranked by the fitted Random Forest model."
            ),
            "Risk scoring": (
                "Transaction risk",
                "After uploading the CSV, enter hypothetical transaction "
                "details and estimate the model's fraud probability."
            )
        }

        heading, description = details[selected]

        st.markdown(f"""
        <div class="panel" style="
            margin-top:22px;
            border-color:#22e6c4;
            background:linear-gradient(120deg,#10283a,#0b192a);
        ">
            <div class="kicker">SELECTED FEATURE</div>
            <h2 style="margin:4px 0 10px;">{heading}</h2>
            <p class="muted" style="font-size:.95rem;">
                {description}
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.stop()


# ---------------- READ DATA ----------------
try:
    df = pd.read_csv(uploaded)
except Exception as exc:
    st.error(f"Could not read CSV: {exc}")
    st.stop()

df.columns = df.columns.astype(str).str.strip()

if "Fraudulent" not in df.columns:
    st.error("The CSV must contain a `Fraudulent` column with 0 and 1 values.")
    st.write("Columns found:", df.columns.tolist())
    st.stop()

if "Transaction_Date" in df.columns:
    dates = pd.to_datetime(
        df["Transaction_Date"], errors="coerce", dayfirst=True
    )
    df["Transaction_Hour"] = dates.dt.hour
    df["Transaction_Day"] = dates.dt.dayofweek

df["Fraudulent"] = pd.to_numeric(df["Fraudulent"], errors="coerce")
df = df[df["Fraudulent"].isin([0, 1])].copy()
df["Fraudulent"] = df["Fraudulent"].astype(int)

if df.empty or df["Fraudulent"].nunique() < 2:
    st.error("The target must contain both 0 and 1.")
    st.stop()

# Exclude IDs, raw date and possible target-leaking keyword.
excluded = [
    "Fraudulent", "Transaction_ID", "Customer_ID",
    "Transaction_Date", "Suspicious_Keyword"
]
X = df.drop(columns=excluded, errors="ignore").copy()
y = df["Fraudulent"]

X = X.dropna(axis=1, how="all")
for col in X.select_dtypes(include=["object"]).columns:
    X[col] = X[col].replace(r"^\s*$", np.nan, regex=True)

if X.shape[1] == 0:
    st.error("No usable feature columns were found.")
    st.stop()

numeric = X.select_dtypes(include=["number", "bool"]).columns.tolist()
categorical = [c for c in X.columns if c not in numeric]

transformers = []
if numeric:
    transformers.append((
        "num",
        SimpleImputer(strategy="median"),
        numeric
    ))
if categorical:
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])
    transformers.append(("cat", cat_pipe, categorical))

preprocessor = ColumnTransformer(transformers=transformers)

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
        min_samples_leaf=2
    ))
])

try:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model.fit(X_train, y_train)
    probabilities = model.predict_proba(X_test)[:, 1]
except Exception as exc:
    st.error(f"Model training failed: {exc}")
    st.stop()

# ---------------- METRICS ----------------
threshold = st.sidebar.slider(
    "Fraud alert threshold", 0.05, 0.95, 0.30, 0.05
)
predictions = (probabilities >= threshold).astype(int)

accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions, zero_division=0)
recall = recall_score(y_test, predictions, zero_division=0)
f1 = f1_score(y_test, predictions, zero_division=0)

try:
    auc = roc_auc_score(y_test, probabilities)
except ValueError:
    auc = np.nan

fraud_count = int(y.sum())
legit_count = int((y == 0).sum())
fraud_rate = y.mean() * 100

# ---------------- MAIN DASHBOARD ----------------
st.markdown(
    '<div class="kicker">FINANCIAL SECURITY / LIVE ANALYSIS</div>',
    unsafe_allow_html=True
)
st.title("Fraud intelligence overview")
st.caption(
    "Explore transaction patterns, evaluate fraud detection performance "
    "and investigate estimated transaction risk."
)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Transactions analyzed", f"{len(df):,}")
m2.metric("Fraudulent transactions", f"{fraud_count:,}")
m3.metric("Legitimate transactions", f"{legit_count:,}")
m4.metric("Fraud rate", f"{fraud_rate:.2f}%")

overview, evaluation, explain, risk = st.tabs([
    "◈ Overview",
    "▥ Model evaluation",
    "✦ Explainability",
    "⌕ Transaction risk"
])

# ---------------- OVERVIEW ----------------
with overview:
    st.subheader("Transaction landscape")
    left, right = st.columns([1, 1.35], gap="large")

    with left:
        fig, ax = plt.subplots(figsize=(5, 3.4))
        fig.patch.set_facecolor("#101f32")
        ax.set_facecolor("#101f32")

        bars = ax.bar(
            ["Legitimate", "Fraudulent"],
            [legit_count, fraud_count],
            color=["#22e6c4", "#ff667a"],
            width=0.55
        )
        ax.set_ylabel("Transactions", color="#d4e1ee")
        ax.tick_params(colors="#d4e1ee")
        ax.grid(axis="y", alpha=.15, color="#a8bdd5")
        for spine in ax.spines.values():
            spine.set_color("#29435e")

        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width()/2,
                bar.get_height(),
                f"{int(bar.get_height()):,}",
                ha="center", va="bottom",
                color="white", fontweight="bold"
            )

        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with right:
        st.markdown("#### Dataset health")
        snapshot = pd.DataFrame({
            "Measure": [
                "Rows", "Columns", "Duplicate rows",
                "Missing cells", "Fraud rate"
            ],
            "Value": [
                f"{len(df):,}", f"{df.shape[1]:,}",
                f"{df.duplicated().sum():,}",
                f"{df.isna().sum().sum():,}",
                f"{fraud_rate:.2f}%"
            ]
        })
        st.dataframe(snapshot, hide_index=True, use_container_width=True)

        if "Transaction_Amount" in df.columns:
            amounts = pd.to_numeric(
                df["Transaction_Amount"], errors="coerce"
            )
            if amounts.notna().any():
                st.metric("Average transaction amount",
                          f"{amounts.mean():,.2f}")

        if "Merchant_Category" in df.columns:
            st.markdown("#### Common merchant categories")
            merchants = (
                df["Merchant_Category"].fillna("Unknown")
                .value_counts().head(7)
                .rename_axis("Merchant category")
                .reset_index(name="Transactions")
            )
            st.dataframe(merchants, hide_index=True, use_container_width=True)

    with st.expander("Preview transaction data"):
        st.dataframe(df.head(20), use_container_width=True)

    st.download_button(
        "Download data preview",
        df.head(1000).to_csv(index=False).encode("utf-8"),
        "fraudlens_data_preview.csv",
        "text/csv"
    )

# ---------------- MODEL EVALUATION ----------------
with evaluation:
    st.subheader("Model performance")
    st.caption(
        f"Random Forest • 80/20 test split • threshold {threshold:.2f}"
    )

    a, b, c, d, e = st.columns(5)
    a.metric("Accuracy", f"{accuracy:.3f}")
    b.metric("Precision", f"{precision:.3f}")
    c.metric("Recall", f"{recall:.3f}")
    d.metric("F1 score", f"{f1:.3f}")
    e.metric("ROC-AUC", f"{auc:.3f}" if np.isfinite(auc) else "N/A")

    cm = confusion_matrix(y_test, predictions, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(6, 4))
    fig.patch.set_facecolor("#101f32")
    ax.set_facecolor("#101f32")
    ax.imshow(cm, cmap="winter")

    ax.set_xticks([0, 1], ["Pred. legitimate", "Pred. fraud"])
    ax.set_yticks([0, 1], ["Actual legitimate", "Actual fraud"])
    ax.set_title("Confusion matrix", color="white", pad=12)
    ax.tick_params(colors="#d4e1ee")

    for i in range(2):
        for j in range(2):
            ax.text(
                j, i, f"{cm[i, j]:,}",
                ha="center", va="center",
                color="white", fontsize=14, fontweight="bold"
            )

    for spine in ax.spines.values():
        spine.set_color("#29435e")

    st.pyplot(fig, use_container_width=False)
    plt.close(fig)

    report = classification_report(
        y_test, predictions,
        labels=[0, 1],
        target_names=["Legitimate", "Fraudulent"],
        output_dict=True,
        zero_division=0
    )
    st.markdown("#### Detailed classification report")
    st.dataframe(
        pd.DataFrame(report).transpose().round(3),
        use_container_width=True
    )

    metrics = pd.DataFrame({
        "Metric": ["Threshold", "Accuracy", "Precision",
                   "Recall", "F1 score", "ROC-AUC"],
        "Value": [threshold, accuracy, precision, recall, f1, auc]
    })
    st.download_button(
        "Download evaluation report",
        metrics.to_csv(index=False).encode("utf-8"),
        "fraudlens_evaluation.csv",
        "text/csv"
    )

# ---------------- EXPLAINABILITY ----------------
with explain:
    st.subheader("Understand model decisions")
    st.caption(
        "Feature importance shows how the fitted Random Forest uses features; "
        "it does not prove that a feature causes fraud."
    )

    try:
        prep = model.named_steps["preprocessor"]
        forest = model.named_steps["classifier"]
        names = prep.get_feature_names_out()
        importance = forest.feature_importances_

        imp = pd.DataFrame({
            "Feature": names,
            "Importance": importance
        })
        imp["Feature"] = imp["Feature"].str.replace(
            r"^(num|cat)__", "", regex=True
        )
        imp = imp.sort_values(
            "Importance", ascending=False
        ).head(15)

        chart_df = imp.sort_values("Importance", ascending=True)
        fig, ax = plt.subplots(figsize=(9, 5))
        fig.patch.set_facecolor("#101f32")
        ax.set_facecolor("#101f32")
        ax.barh(
            chart_df["Feature"],
            chart_df["Importance"],
            color="#22d4bd"
        )
        ax.set_xlabel("Importance", color="#d4e1ee")
        ax.tick_params(colors="#d4e1ee", labelsize=9)
        ax.grid(axis="x", alpha=.15, color="#a8bdd5")

        for spine in ax.spines.values():
            spine.set_color("#29435e")

        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        st.dataframe(
            imp.reset_index(drop=True),
            hide_index=True,
            use_container_width=True
        )
        st.download_button(
            "Download feature importance",
            imp.to_csv(index=False).encode("utf-8"),
            "fraudlens_feature_importance.csv",
            "text/csv"
        )
    except Exception as exc:
        st.warning(f"Could not display feature importance: {exc}")

# ---------------- TRANSACTION RISK ----------------
with risk:
    st.subheader("Score a hypothetical transaction")
    st.caption(
        "Adjust the values to estimate fraud probability. "
        "This experimental score should not be used alone for real financial decisions."
    )

    values = {}
    columns = st.columns(2)

    for i, name in enumerate(X.columns):
        series = X[name]
        with columns[i % 2]:
            if name in numeric:
                s = pd.to_numeric(series, errors="coerce")
                default = float(s.median()) if s.notna().any() else 0.0
                low = float(s.min()) if s.notna().any() else default - 1
                high = float(s.max()) if s.notna().any() else default + 1

                if not np.isfinite(low):
                    low = default - 1
                if not np.isfinite(high):
                    high = default + 1
                if low == high:
                    high = low + 1

                values[name] = st.number_input(
                    name.replace("_", " "),
                    min_value=low,
                    max_value=high,
                    value=min(max(default, low), high),
                    key=f"input_{name}"
                )
            else:
                options = (
                    series.dropna().astype(str)
                    .value_counts().index.tolist()
                )
                if not options:
                    options = ["Unknown"]

                values[name] = st.selectbox(
                    name.replace("_", " "),
                    options,
                    key=f"select_{name}"
                )

    if st.button("Analyze transaction risk", type="primary",
                 use_container_width=True):
        try:
            input_df = pd.DataFrame([values], columns=X.columns)
            probability = float(model.predict_proba(input_df)[0, 1])
            pct = probability * 100

            if probability >= max(threshold, 0.60):
                level = "HIGH RISK"
                color = "#ff667a"
                advice = "Flag for further review and verify the transaction context."
                icon = "🚨"
            elif probability >= threshold:
                level = "ELEVATED RISK"
                color = "#ffb454"
                advice = "Consider additional verification before taking action."
                icon = "⚠️"
            else:
                level = "LOWER MODEL RISK"
                color = "#22e6c4"
                advice = "The score is below the selected threshold; this does not guarantee legitimacy."
                icon = "✓"

            st.markdown(f"""
            <div class="panel" style="margin-top:16px">
              <div class="kicker">FRAUDLENS / RISK RESULT</div>
              <h2 style="color:{color}!important">{icon} {level}</h2>
              <div style="font-size:2.3rem;font-weight:900;color:#f3f7ff">
                {pct:.1f}%
              </div>
              <div class="muted">Estimated fraud probability</div>
              <p style="margin-top:12px">{html.escape(advice)}</p>
            </div>
            """, unsafe_allow_html=True)

            st.progress(probability, text="Estimated fraud probability")

            result = pd.DataFrame([values])
            result["Estimated_Fraud_Probability"] = probability
            result["Selected_Threshold"] = threshold
            result["Model_Flag"] = int(probability >= threshold)

            st.download_button(
                "Download scored transaction",
                result.to_csv(index=False).encode("utf-8"),
                "fraudlens_scored_transaction.csv",
                "text/csv"
            )
        except Exception as exc:
            st.error(f"Could not score this transaction: {exc}")

# ---------------- FOOTER ----------------
st.divider()
st.markdown("""
<div style="display:flex;justify-content:space-between;gap:10px;
flex-wrap:wrap;color:#8ea7c2;font-size:.76rem;">
  <span>🛡️ FRAUDLENS · FINANCIAL RISK ANALYTICS</span>
  <span>Educational prototype · Verify results before real-world use</span>
</div>
""", unsafe_allow_html=True)
