# 💳 Credit Risk Scoring — Machine Learning Project

A complete **end-to-end ML pipeline** for predicting loan defaults, built with Python and deployed as a FastAPI REST API.

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Latest-orange?logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Latest-green)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-teal?logo=fastapi&logoColor=white)

---

## 📌 Overview

This project builds a credit-risk scoring model that predicts whether a borrower will default on a loan within the next two years. It uses the **"Give Me Some Credit"** dataset from Kaggle and covers:

- Exploratory Data Analysis (EDA)
- Feature Engineering
- Handling class imbalance with **SMOTEENN**
- Training & comparing **Logistic Regression**, **Random Forest**, and **XGBoost**
- Hyperparameter tuning
- Model explainability with **SHAP**
- Deployment via a **FastAPI** endpoint

---

## 🗂️ Project Structure

```
ml-project/
├── Credit_Risk_Scoring_ML_Project.ipynb  # Full detailed notebook
├── Credit_Risk_Simple.ipynb              # Simplified / beginner-friendly version
├── app.py                                # FastAPI prediction API
├── feature_names.txt                     # Feature list used by the model
├── requirements.txt                      # Python dependencies
├── Credit_Risk_Scoring_Presentation.pptx # Project presentation
└── README.md
```

> **Note:** Large files like `cs-training.csv`, `loan.csv`, and `.pkl` model files are excluded from the repo. See [Dataset](#-dataset) below.

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/Credit-Risk-Scoring-ML.git
cd Credit-Risk-Scoring-ML
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the dataset

Download **"Give Me Some Credit"** from Kaggle:  
🔗 <https://www.kaggle.com/c/GiveMeSomeCredit/data>

Place `cs-training.csv` in the project root directory.

### 5. Run the notebook

Open Jupyter and run `Credit_Risk_Scoring_ML_Project.ipynb` (or `Credit_Risk_Simple.ipynb` for a simpler walkthrough).

```bash
jupyter notebook
```

### 6. Start the API server

After running the notebook (which saves `credit_risk_model.pkl` and `credit_scaler.pkl`):

```bash
uvicorn app:app --reload
```

Visit the interactive docs at **http://localhost:8000/docs**.

---

## 📊 Dataset

| Dataset | Source |
|---------|--------|
| Give Me Some Credit | [Kaggle Competition](https://www.kaggle.com/c/GiveMeSomeCredit/data) |

The dataset contains **150,000 rows** and **11 features** including income, debt ratio, age, and delinquency history.

---

## 🧠 Models & Results

| Model | ROC-AUC |
|-------|---------|
| Logistic Regression | ~0.80 |
| Random Forest | ~0.86 |
| **XGBoost (Tuned)** | **~0.87** |

The final model is **XGBoost** with hyperparameter tuning and SMOTEENN resampling.

---

## 🔍 Model Explainability

SHAP (SHapley Additive exPlanations) is used to explain individual predictions and understand global feature importance.

---

## 🛠️ Tech Stack

- **Python 3.9+**
- **pandas**, **NumPy** — data processing
- **scikit-learn** — modeling & evaluation
- **XGBoost** — gradient boosting
- **imbalanced-learn** — SMOTEENN resampling
- **SHAP** — model explainability
- **matplotlib**, **seaborn** — visualization
- **FastAPI** — REST API deployment

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to open an issue or submit a pull request.

---

## 👤 Author

**Aditya**  
📧 [GitHub Profile](https://github.com/<your-username>)
