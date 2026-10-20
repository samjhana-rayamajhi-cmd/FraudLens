# 🛡️ FraudLens — Financial Fraud Detection & Risk Analytics

**See Risk. Understand Patterns. Act Smarter.**

FraudLens is an interactive financial fraud detection and risk analytics dashboard built using Python, Streamlit, and Machine Learning. It helps users explore transaction data, evaluate fraud detection performance, understand important model features, and estimate the fraud probability of hypothetical transactions.

🚀 **Live Demo:** https://fraudlens-fnj5ow6hle59pype4khyuw.streamlit.app/

💻 **GitHub Repository:** https://github.com/samjhana-rayamajhi-cmd/FraudLens

---

## 📌 Project Overview

Financial fraud can lead to significant financial losses. Identifying suspicious transactions and understanding transaction patterns can help support fraud investigation and risk assessment.

FraudLens uses a Random Forest Classifier to analyze labeled transaction data and estimate the likelihood of fraud. Its interactive dashboard presents analytical results through charts, performance metrics, feature importance, and transaction risk scoring.

**Note:** FraudLens is an educational prototype and is not intended to replace professional fraud investigation or production financial security systems.

## ✨ Key Features

### 1. 📊 Transaction Overview
- Analyze the total number of transactions.
- View fraudulent and legitimate transaction counts.
- Examine the overall fraud rate.
- Review dataset quality, missing values, and duplicate rows.
- Explore merchant categories and transaction amounts.
- Preview and download transaction data.

### 2. 🎯 Model Evaluation
Evaluate the Random Forest model using:
- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Confusion Matrix
- Detailed Classification Report

Users can adjust the fraud alert threshold to explore how classification decisions change.

### 3. 🔍 Model Explainability
- Visualize the top 15 important features.
- Explore the relative importance of input features.
- View and download the feature importance table.

Feature importance indicates how the fitted model uses features; it does not establish causation.

### 4. 🚨 Transaction Risk Scoring
- Enter hypothetical transaction details.
- Estimate the model's fraud probability.
- View a risk category based on the configured threshold.
- Download the scored transaction and its estimated probability.

The score is an estimate, not a guarantee that a transaction is fraudulent or legitimate.

## 🧰 Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Streamlit | Interactive web dashboard |
| Pandas | Data processing and analysis |
| NumPy | Numerical operations |
| Matplotlib | Data visualization |
| Scikit-learn | Machine learning, preprocessing, and evaluation |
| Random Forest Classifier | Fraud classification |
| Git and GitHub | Version control and source code hosting |
| Streamlit Community Cloud | Web app deployment |

## 🏗️ Application Workflow

1. Upload a labeled transaction dataset in CSV format.
2. Validate and preprocess the data.
3. Separate the input features from the target column.
4. Handle missing values using imputation.
5. Encode categorical features using one-hot encoding.
6. Split the dataset into training and testing sets using an 80/20 split.
7. Train the Random Forest Classifier.
8. Evaluate model performance using classification metrics.
9. Explore feature importance and estimate transaction risk.

## 📂 Dataset Requirements

Upload a CSV dataset containing a target column named `Fraudulent`.

- `0` = Legitimate transaction
- `1` = Fraudulent transaction

The dataset should contain both classes and usable input features.

Example:

| Transaction_Amount | Payment_Method | Is_International | Fraudulent |
|---:|---|---:|---:|
| 250.00 | Card | 0 | 0 |
| 8500.00 | Online | 1 | 1 |
| 1200.00 | UPI | 0 | 0 |

*The values above are illustrative examples, not actual model predictions.*

The application can use different compatible CSV datasets, provided they contain the required target column and suitable input features.

## ⚙️ Installation and Setup

### Prerequisites
- Python 3.10 or a compatible Python version
- pip
- Git (optional, for cloning the repository)

### 1. Clone the repository

```bash
git clone https://github.com/samjhana-rayamajhi-cmd/FraudLens.git
cd FraudLens
```

### 2. Create a virtual environment

**Windows:**

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

The `requirements.txt` file should include:

```text
streamlit
pandas
numpy
matplotlib
scikit-learn
```

### 4. Run the application

```bash
python -m streamlit run app.py
```

Open the local URL displayed in the terminal, usually:

`http://localhost:8501`

### 5. Upload your dataset

Use the sidebar file uploader to upload your CSV file and explore the dashboard.

## 🌐 Live Deployment

Try the deployed application here:

**https://fraudlens-fnj5ow6hle59pype4khyuw.streamlit.app/**

The application is hosted on Streamlit Community Cloud.

## 🔮 Future Enhancements

Potential future improvements include:

- Interactive transaction filtering and search.
- Additional fraud detection algorithms.
- Model comparison and hyperparameter tuning.
- Improved model monitoring and data drift analysis.
- More detailed explainability using SHAP or similar methods.
- Enhanced reporting and dashboard customization.
- Improved input validation and security for production use.

These are proposed enhancements and are not necessarily implemented in the current version.

## ⚠️ Limitations

- Model performance depends on the quality and representativeness of the uploaded dataset.
- The model is trained on the uploaded data and evaluated using a random train/test split.
- The selected threshold affects fraud classification results.
- Feature importance does not prove that a feature causes fraud.
- Predicted probabilities may not be calibrated to real-world fraud rates.
- This educational prototype should not be used as the sole basis for real financial decisions.

## 👩‍💻 Author

**Samjhana Rayamajhi**

Computer Engineering Student

GitHub: https://github.com/samjhana-rayamajhi-cmd

---

⭐ If you find FraudLens useful, explore the live demo and share your feedback!

**FraudLens — See Risk. Understand Patterns. Act Smarter.**