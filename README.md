# Skin Hydration Prediction using ML

This project focuses on predicting skin hydration status (Hydrated or Not Hydrated) based on input features using machine learning models: **Random Forest** and **XGBoost**. It uses a balanced dataset and includes preprocessing, model training, evaluation, and manual prediction capabilities.

---

## 🔍 Project Overview

The goal is to build a classification model that can accurately predict whether a skin sample is hydrated based on multiple features. Two powerful machine learning models—**Random Forest** and **XGBoost**—are trained and compared to select the best-performing one.

---

## 📁 Dataset

- **Name**: `balanced_skin_hydration_dataset.csv`
- **Target Variable**: `Target` (1 = Hydrated, 0 = Not Hydrated)
- **Features**: 10 numeric features representing various skin characteristics.

---

## ⚙️ Workflow

1. **File Upload** (for Google Colab)
2. **Data Loading** using `pandas`
3. **Preprocessing**:
   - Feature scaling using `StandardScaler`
   - Train-test split (80% train, 20% test)
4. **Model Training**:
   - Random Forest Classifier
   - XGBoost Classifier
5. **Evaluation**:
   - Accuracy and F1 Score
   - Comparison of model performance
6. **Feature Importance Plot** for the best model
7. **Train vs Test Distribution Visualization**
8. **Manual Prediction Example** with synthetic input

---

## 🧠 Model Description

### 1. Random Forest Classifier
- Ensemble model using bagging of decision trees.
- Handles non-linear data well.
- Robust to overfitting due to averaging over many trees.
- Configured with:
  - `n_estimators=100`
  - `random_state=42`

### 2. XGBoost Classifier
- Gradient boosting model with regularization.
- Excellent performance in many structured data tasks.
- Tuned with:
  - `use_label_encoder=False`
  - `eval_metric='logloss'`
  - `random_state=42`

After training both models, the one with the higher F1 score is selected as the **best model**.

---

## 📊 Performance Metrics

Each model is evaluated using:

- **Accuracy**: Proportion of correctly predicted samples.
- **F1 Score**: Harmonic mean of precision and recall, useful for imbalanced classification.

Example output:

```
Random Forest - Accuracy: 0.8700, F1: 0.8600
XGBoost       - Accuracy: 0.8800, F1: 0.8700
```

---

## 🧪 Manual Prediction Example

A manually constructed sample input is scaled and passed to the best model for prediction. The output tells whether the skin is predicted to be **Hydrated** or **Not Hydrated**.

---

## 📦 Dependencies

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `xgboost`

---

## ▶️ How to Run

1. Upload the dataset using the `files.upload()` cell.
2. Run each cell in order within the Jupyter notebook.
3. View evaluation metrics and plots.
4. Modify the manual input to predict custom skin profiles.
