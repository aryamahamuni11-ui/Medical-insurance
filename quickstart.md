# 🚀 Medical Insurance Price Prediction - Quick Start Guide

## 📌 Overview

This project predicts medical insurance charges using machine learning models.

---

## ⚙️ Setup

### 1. Clone Repository

```bash
git clone <repo-url>
cd project-folder
```

### 2. Install Dependencies

```bash
py -m pip install numpy pandas matplotlib seaborn scikit-learn xgboost feature-engine joblib
```

---

## 📂 Dataset

Place `insurance.csv` in the root directory.

---

## ▶️ Run Notebook

```bash
jupyter notebook
```

Open:

```
insurance_prediction.ipynb
```

---

## 🧠 Workflow

1. Data Loading & Cleaning
2. Exploratory Data Analysis (EDA)
3. Feature Engineering
4. Model Training
5. Evaluation
6. Final Model Selection

---

## 📊 Models Used

- Linear Regression
- Random Forest
- Gradient Boosting
- XGBoost (Best Performing)

---

## 🏆 Best Model

XGBoost Regressor with tuned parameters:

- n_estimators = 15
- max_depth = 3
- gamma = 0

---

## 📈 Evaluation Metrics

- R² Score
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)

---

## 💾 Model Saving

```python
joblib.dump(model, "model.pkl")
```

---

## 🔮 Make Predictions

```python
model.predict(new_data)
```

---

## 📌 Key Insights

- Smoking is the strongest factor affecting charges
- BMI and age moderately influence cost
- Region has minimal impact

---

## 🛠 Future Improvements

- Add deployment (Flask / FastAPI)
- Try Deep Learning models
- Feature engineering (interaction terms)
- Use SHAP for explainability

---
