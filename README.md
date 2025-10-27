# Predictive Maintenance using NASA Turbofan Dataset

## Overview
This project aims to predict the **Remaining Useful Life (RUL)** of turbofan engines using the NASA Turbofan Engine Degradation Simulation Dataset (CMAPSS). The objective is to anticipate equipment failure in advance and assess whether an engine is operating within safe limits based on sensor readings.  

Two approaches were explored:
1. **Regression modeling** – to predict the continuous Remaining Useful Life.  
2. **Classification modeling** – to predict whether the engine will work more than 50 cycles or not.  

A **Streamlit web application** was developed and deployed to allow users to input sensor readings and receive both the RUL estimate and a safety status message.

---

## Project Workflow

### 1. Data Exploration and Cleaning
- Loaded and examined the CMAPSS turbofan dataset to understand its structure and characteristics.  
- Performed exploratory data analysis (EDA) to visualize degradation patterns and identify feature distributions.  
- Removed:
  - Constant and quasi-constant features with negligible variance.  
  - Highly correlated features to reduce redundancy and multicollinearity.  
- Finalized a clean, normalized dataset for modeling.

### 2. Regression Modeling
The goal of regression modeling was to estimate the **Remaining Useful Life (RUL)** in continuous form.

**Models trained:**
- Linear Regression  
- Ridge Regression  
- Lasso Regression  
- Random Forest Regressor  
- Gradient Boosting Regressor  
- XGBoost Regressor  

**Best Model:**  
- **Gradient Boosting Regressor**  
- Achieved an **R² score of approximately 0.80** on the test set.

**Model Interpretability:**
- Used **SHAP (SHapley Additive exPlanations)** to analyze the contribution of each sensor feature to the model’s predictions.  
- Generated a **waterfall chart** for a sample engine instance to visualize how individual sensor readings influenced the predicted RUL value.  
- This interpretability step helped identify the most influential sensors affecting degradation and provided insight into model reasoning and transparency.

### 3. Classification Modeling
A secondary classification problem was formulated to categorize the engine’s state as **“Operating Safely”** or **“Unsafe”** based on RUL thresholds or sensor-derived conditions.

**Models trained:**
- Logistic Regression  
- Random Forest Classifier  
- Gradient Boosting Classifier  
- XGBoost Classifier  

**Best Model:**  
- **Logistic Regression**  
- Achieved an **F1 score of 0.95**, indicating highly reliable classification performance.

### 4. Streamlit Application
A **Streamlit-based web application** was developed (with AI assistance) to demonstrate the system in real-time.

**Features:**
- User can input sensor readings through a simple interface.  
- The app displays:
  - The **predicted RUL** from the regression model.  
  - A **status message** indicating whether the engine is operating within safe limits or nearing failure.  
- The app is **deployed using Streamlit Cloud**, making it accessible via a browser.

**Example workflow:**
1. User enters sensor values in the app.  
2. The model predicts RUL.  
3. A safety message is displayed — e.g., *“Engine operating safely”* or *“Maintenance required soon.”*

---

## Results Summary

| Task | Best Model | Metric | Score |
|------|-------------|---------|-------|
| Regression (RUL Prediction) | Gradient Boosting Regressor | R² | 0.78 |
| Classification (Safety Prediction) | Logistic Regression | F1-score | 0.95 |
