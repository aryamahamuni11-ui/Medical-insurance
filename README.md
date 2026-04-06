# 🏥 Medical Insurance Price Prediction

_A Machine Learning Regression Project_

---

# 📌 1. Introduction

This project aims to **predict individual medical insurance charges** based on demographic and health-related features using machine learning.

The dataset includes attributes such as age, BMI, smoking status, and region, which are used to estimate insurance costs.

---

# 🎯 2. Problem Statement

Predict the **insurance charges (`charges`)** for an individual given:

- Age
- Sex
- BMI
- Number of children
- Smoking status
- Region

---

## 🧩 Problem Type

| Attribute       | Value                 |
| --------------- | --------------------- |
| Type            | Supervised Learning   |
| Category        | Regression            |
| Target Variable | `charges`             |
| Dataset Size    | 1338 rows × 7 columns |

---

# 📂 3. Dataset Description

| Feature  | Description                     |
| -------- | ------------------------------- |
| age      | Age of the individual           |
| sex      | Gender (male/female)            |
| bmi      | Body Mass Index                 |
| children | Number of dependents            |
| smoker   | Smoking status (yes/no)         |
| region   | Residential area                |
| charges  | Medical insurance cost (target) |

---

# ⚙️ 4. Project Pipeline Workflow

## 🔄 End-to-End Flow

```
Raw Data
   ↓
Data Cleaning
   ↓
Exploratory Data Analysis (EDA)
   ↓
Outlier Treatment
   ↓
Feature Transformation
   ↓
Train-Test Split
   ↓
Preprocessing Pipeline
   ↓
Model Training
   ↓
Hyperparameter Tuning
   ↓
Model Evaluation
   ↓
Feature Importance
   ↓
Model Saving
   ↓
Prediction
```

---

# 🧹 5. Data Preprocessing

## ✅ 5.1 Missing Values

- No missing values found
- No imputation required

## ✅ 5.2 Duplicate Handling

- 1 duplicate row removed

---

# 📊 6. Exploratory Data Analysis (EDA)

EDA helps in understanding:

- Data distribution
- Relationships between variables
- Feature importance insights

---

## 📌 6.1 Target Variable Analysis

### Observation:

- `charges` is **right-skewed**
- Presence of high-value outliers

### Insight:

- Real-world data: few individuals incur extremely high medical costs

---

## 📌 6.2 Categorical Features

### Observations:

- Sex → balanced distribution
- Smoker → imbalanced (~20% smokers)
- Region → evenly distributed

### Key Insight:

- Despite imbalance, **smoker is the most impactful feature**

---

## 📌 6.3 Bivariate Analysis

### Charges vs Smoker:

- Smokers pay **3–4× higher costs**

### Charges vs Age:

- Positive relationship (older → higher cost)

### Charges vs BMI:

- Weak correlation overall
- Strong effect when combined with smoking

---

## 📌 6.4 Correlation Analysis

| Feature | Correlation with Charges |
| ------- | ------------------------ |
| smoker  | High (~0.79)             |
| age     | Moderate                 |
| bmi     | Weak                     |
| others  | Minimal                  |

---

# 🚨 7. Outlier Treatment

## BMI Outliers

### Method:

- Interquartile Range (IQR)

```
Lower Bound = Q1 - 1.5 * IQR
Upper Bound = Q3 + 1.5 * IQR
```

### Action:

- Applied **Winsorization (capping)**

### Decision:

- BMI → capped
- Charges → retained (real-world values)

---

# 🔄 8. Feature Transformation

## Log Transformation on Target

```
charges → log1p(charges)
```

### Purpose:

- Reduce skewness
- Improve model performance

### Result:

- Distribution becomes approximately normal

---

# 🔀 9. Train-Test Split

- 80% → Training
- 20% → Testing

Purpose:

- Evaluate model on unseen data

---

# 🧠 10. Preprocessing Pipeline

## Tools Used:

- `ColumnTransformer`
- `Pipeline`

---

## Operations:

| Feature Type | Transformation |
| ------------ | -------------- |
| Numerical    | StandardScaler |
| Categorical  | OneHotEncoder  |

---

## Benefits:

- Prevents data leakage
- Ensures reproducibility
- Production-ready

---

# 🤖 11. Model Training

## Models Used:

| Model             | Type              |
| ----------------- | ----------------- |
| Linear Regression | Baseline          |
| Random Forest     | Bagging           |
| Gradient Boosting | Boosting          |
| XGBoost           | Advanced Boosting |

---

# 📏 12. Evaluation Metrics

| Metric   | Meaning                  |
| -------- | ------------------------ |
| R² Score | Goodness of fit          |
| MAE      | Average prediction error |
| RMSE     | Penalizes large errors   |

---

# 🏆 13. Model Performance

### Observations:

- XGBoost & Gradient Boosting performed best
- Linear model performed worse → confirms non-linearity

---

# ⚙️ 14. Hyperparameter Tuning

## Method:

- GridSearchCV

## Tuned Parameters:

- n_estimators
- max_depth
- learning_rate
- gamma

---

# 📊 15. Feature Importance

### Key Findings:

| Feature | Importance |
| ------- | ---------- |
| smoker  | Highest    |
| age     | Moderate   |
| bmi     | Moderate   |
| others  | Low        |

---

## Insight:

Smoking is the **dominant driver** of insurance cost.

---

# 💾 16. Model Saving

```
joblib.dump(model, "model.pkl")
```

---

# 🔮 17. Prediction Pipeline

### Steps:

1. Input raw data
2. Pipeline handles preprocessing
3. Model predicts log value
4. Convert back using:

```
np.expm1(prediction)
```

---

# 🏗️ 18. System Design

## 🧩 Architecture

```
                ┌────────────────────┐
                │   User Input       │
                └─────────┬──────────┘
                          ↓
                ┌────────────────────┐
                │  Preprocessing     │
                │ (Scaling + Encoding)
                └─────────┬──────────┘
                          ↓
                ┌────────────────────┐
                │   ML Model         │
                │ (XGBoost Pipeline) │
                └─────────┬──────────┘
                          ↓
                ┌────────────────────┐
                │ Log Prediction     │
                └─────────┬──────────┘
                          ↓
                ┌────────────────────┐
                │ Inverse Transform  │
                │ (expm1)            │
                └─────────┬──────────┘
                          ↓
                ┌────────────────────┐
                │ Final Output ($)   │
                └────────────────────┘
```

---

# 📦 19. Deployment Readiness

The project is **production-ready** because:

- Uses Pipeline (no manual preprocessing)
- Model saved as `.pkl`
- Handles raw input directly

---

# 🚀 20. Future Improvements

- Add SHAP explainability
- Deploy using FastAPI / Flask
- Build frontend UI
- Add interaction features (e.g., smoker × BMI)
- Try deep learning models

---

# 🏁 21. Conclusion

This project demonstrates:

- End-to-end ML workflow
- Strong EDA and data understanding
- Proper preprocessing and transformation
- Model comparison and tuning
- Production-ready pipeline

---

## 💡 Final Insight

The model learns:

- Smoking status is the **strongest predictor**
- Age and BMI moderately affect cost
- Feature interactions significantly impact predictions

---


