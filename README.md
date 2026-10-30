# 🧠 Stroke Risk Prediction & Exploratory Data Analysis

A comprehensive data science and machine learning project focused on investigating physiological, clinical, and lifestyle risk factors contributing to **strokes**, followed by benchmarking four supervised classification models to predict stroke likelihood.

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Key Objectives](#-key-objectives)
- [Dataset Summary](#-dataset-summary)
- [Data Preprocessing Pipeline](#-data-preprocessing-pipeline)
- [Exploratory Data Analysis (EDA) Highlights](#-exploratory-data-analysis-eda-highlights)
- [Machine Learning Workflow](#-machine-learning-workflow)
- [Model Evaluation & Comparison](#-model-evaluation--comparison)
- [Visualizations & Screenshots](#-visualizations--screenshots)
- [Project Structure](#-project-structure)
- [Installation & Usage](#-installation--usage)
- [Key Conclusions](#-key-conclusions)
- [Future Enhancements](#-future-enhancements)

---

## 🔬 Overview

A **stroke** occurs when blood supply to part of the brain is interrupted or reduced, preventing brain tissue from getting oxygen and nutrients. Timely risk assessment can save lives and prevent permanent disability.

This project analyzes patient demographic, clinical, and lifestyle metrics to uncover primary risk drivers and develops a machine learning classification pipeline. Because medical stroke datasets are inherently imbalanced (only ~4.87% of patients had strokes), this project addresses data skewness using **SMOTE (Synthetic Minority Over-sampling Technique)** on the training set to prevent bias toward the majority class.

---

## 🎯 Key Objectives

1. **Objective 1: Stroke Prevalence & Class Distribution**  
   Quantify the distribution and severe imbalance of stroke occurrences across the patient population.
2. **Objective 2: Age Correlation & Stratification**  
   Examine how stroke risk scales across different age brackets (`0-18`, `19-40`, `41-60`, `61-80`, `81+`).
3. **Objective 3: Impact of Pre-existing Medical Conditions**  
   Assess risk differentials associated with **hypertension** and **heart disease**.
4. **Objective 4: Physiological Biomarkers**  
   Evaluate correlations between **average glucose level**, **body mass index (BMI)**, and stroke incidence.
5. **Objective 5: Lifestyle & Socio-Demographic Drivers**  
   Investigate relationships between stroke probability and smoking status, work type, marital status, and residence location.
6. **Machine Learning Benchmarking**  
   Train and compare four distinct algorithms (**KNN**, **Gaussian Naive Bayes**, **Logistic Regression**, and **Random Forest**) using accuracy, precision, recall, and F1-score.

---

## 📊 Dataset Summary

- **File**: [`healthcare-dataset-stroke-data.csv`](file:///c:/Users/dewbr/OneDrive/Desktop/Python_project/healthcare-dataset-stroke-data.csv)
- **Source**: Healthcare Stroke Dataset ([AIKosh - IndiaAI](https://indiaai.gov.in/) / Kaggle)
- **Total Records**: 5,110 patients
- **Feature Count**: 12 columns (11 features + 1 target variable)

### Feature Dictionary

| Feature | Type | Description | Values / Range |
| :--- | :--- | :--- | :--- |
| `id` | Identifier | Unique patient identifier | Dropped during preprocessing |
| `gender` | Categorical | Biological sex | `Male`, `Female` (`Other` cleaned) |
| `age` | Numerical | Age of the patient | 0.08 - 82.0 years |
| `hypertension` | Binary | Chronic high blood pressure | `0` = No, `1` = Yes |
| `heart_disease` | Binary | History of heart disease | `0` = No, `1` = Yes |
| `ever_married` | Binary | Marital status history | `No`, `Yes` |
| `work_type` | Categorical | Occupation category | `Private`, `Self-employed`, `Govt_job`, `children`, `Never_worked` |
| `Residence_type`| Categorical | Living environment | `Urban`, `Rural` |
| `avg_glucose_level` | Numerical | Blood glucose level | 55.12 - 271.74 mg/dL |
| `bmi` | Numerical | Body Mass Index | 10.3 - 97.6 kg/m² (Contains 201 missing values) |
| `smoking_status` | Categorical | Tobacco consumption history | `formerly smoked`, `never smoked`, `smokes`, `Unknown` |
| **`stroke`** | **Target (Binary)** | Diagnosis of stroke event | **`0` = No Stroke (95.13%), `1` = Stroke (4.87%)** |

---

## ⚙️ Data Preprocessing Pipeline

Implemented in [`second.py`](file:///c:/Users/dewbr/OneDrive/Desktop/Python_project/second.py#L32-L47):

1. **Deduplication & Identifier Removal**:
   - Checked and confirmed zero duplicate records.
   - Removed the non-predictive `id` column.
2. **Missing Value Imputation**:
   - `bmi`: 201 missing values imputed with the feature **median** (robust against outliers).
3. **Inconsistent Category Resolution**:
   - `gender`: A single occurrence of `'Other'` was replaced with the statistical mode (`'Female'`).
   - `smoking_status`: Patients with `'Unknown'` status were imputed with the mode (`'never smoked'`).
4. **Feature Encoding**:
   - Categorical columns (`gender`, `ever_married`, `work_type`, `Residence_type`, `smoking_status`) converted into numerical features via **One-Hot Encoding** (`pd.get_dummies(..., dtype=int)`).
5. **Stratified Train-Test Split**:
   - Data partitioned into **80% training** and **20% testing** sets.
   - Stratification (`stratify=y`) ensures identical stroke proportions across both subsets.
6. **Feature Standardization**:
   - Scaled using `StandardScaler`, fitted on `X_train` and transformed on `X_test` to prevent data leakage.
7. **Class Imbalance Handling (SMOTE)**:
   - Applied **SMOTE** strictly to the training partition (`X_train`, `y_train`).
   - Balanced training set from `3,889` negative and `199` positive samples to `3,889` in each class.

---

## 📈 Exploratory Data Analysis (EDA) Highlights

### 1. Age Factor (Objective 2)
- **Average Age**: Patients who suffered a stroke had an average age of **67.73 years**, compared to **41.97 years** for non-stroke patients.
- **Stratified Stroke Rates**:
  - `0-18`: 0.22%
  - `19-40`: 0.45%
  - `41-60`: 4.10%
  - `61-80`: 12.96%
  - `81+`: **19.83%** (Nearly 1 in 5 individuals over 80 suffered a stroke)

### 2. Clinical Conditions (Objective 3)
- **Hypertension**:
  - Patients without hypertension: **3.97%** stroke rate
  - Patients with hypertension: **13.25%** stroke rate (**~3.3x increase**)
- **Heart Disease**:
  - Patients without heart disease: **4.18%** stroke rate
  - Patients with heart disease: **17.03%** stroke rate (**~4x increase**)

### 3. Glucose & BMI Metrics (Objective 4)
- **Average Glucose Level**:
  - Non-Stroke: **104.80 mg/dL**
  - Stroke Patients: **132.54 mg/dL** (Significant elevation in diabetic/prediabetic ranges)
- **BMI**:
  - Non-Stroke: **28.80 kg/m²**
  - Stroke Patients: **30.09 kg/m²**

### 4. Lifestyle & Demographics (Objective 5)
- **Smoking Status**:
  - Formerly Smoked: **7.91%**
  - Smokes: **5.32%**
  - Never Smoked: **3.99%**
- **Occupation (`work_type`)**:
  - Self-employed: **7.94%** (highest among occupations, linked with higher median age)
  - Private Sector: **5.09%**
  - Government Job: **5.02%**
  - Children: **0.29%**
- **Marital Status**: Married individuals exhibited a higher stroke rate (**6.56%** vs **1.65%** for unmarried), directly correlated with the age distribution.

---

## 🤖 Machine Learning Workflow

The script [`second.py`](file:///c:/Users/dewbr/OneDrive/Desktop/Python_project/second.py#L125-L230) evaluates four supervised learning algorithms:

1. **K-Nearest Neighbors (KNN)**: `n_neighbors=5`, Euclidean distance metric.
2. **Gaussian Naive Bayes (GNB)**: Probabilistic classifier assuming feature independence.
3. **Logistic Regression (LR)**: Linear decision boundary with L2 regularization (`max_iter=1000`).
4. **Random Forest Classifier (RF)**: Ensemble of 100 decision trees (`n_estimators=100`, `random_state=42`).

---

## 🏆 Model Evaluation & Comparison

Testing evaluation on the 20% holdout test set (1,022 unseen patients; 972 Non-Stroke, 50 Stroke):

| Model | Overall Accuracy | Stroke Precision (Class 1) | Stroke Recall (Class 1) | Stroke F1-Score | Clinical Strengths & Trade-offs |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Random Forest** | **0.898 (89.8%)** | **0.145** | 0.220 | 0.175 | **Highest overall accuracy**; lowest false alarm rate. |
| **KNN** | 0.799 (79.9%) | 0.086 | 0.320 | 0.135 | Balanced baseline; moderate false positives. |
| **Logistic Regression** | 0.745 (74.5%) | 0.137 | **0.800** | **0.235** | **Best balanced clinical screening model**; catches 80% of all strokes with highest F1. |
| **Gaussian Naive Bayes** | 0.363 (36.3%) | 0.070 | **0.980** | 0.131 | **Highest sensitivity (98% Recall)**; catches almost all positive cases but produces high false alarms. |

### Best Model Selection Analysis

- **Best by Accuracy**: **Random Forest (89.8%)**
  - Confusion Matrix:
    ```text
    [[907  65]
     [ 39  11]]
    ```
- **Best for Medical Screening**: **Logistic Regression (Recall: 80%, F1: 0.235)**
  - In healthcare risk prediction, **Recall (Sensitivity)** is often favored over raw accuracy because failing to detect a potential stroke (False Negative) has far more severe consequences than a false alarm (False Positive).

---

## 🖼️ Visualizations & Screenshots

The analysis generates comprehensive visualizations saved as screenshots in the root directory:

| Screenshot | Category | Description |
| :--- | :--- | :--- |
| ![Model Comparison](Screenshot%20(152).png) | **Model Benchmark** | Comparison of Accuracy, Precision, Recall, and F1 across KNN, Naive Bayes, Logistic Regression, and Random Forest. |
| ![Hypertension Risk](Screenshot%20(148).png) | **Clinical Risk** | Stroke rate comparison between hypertensive vs. non-hypertensive patients. |
| ![Glucose vs Stroke](Screenshot%20(149).png) | **Biomarker Analysis** | Box plot of blood glucose distributions for stroke and non-stroke cohorts. |
| ![Smoking Status](Screenshot%20(150).png) | **Lifestyle Risk** | Stroke incidence across smoking categories (`formerly smoked`, `never smoked`, `smokes`). |
| ![Work Type](Screenshot%20(151).png) | **Demographics** | Stroke percentage across employment categories (`Self-employed`, `Private`, `Govt_job`, etc.). |
| ![IDLE Execution - Results](Screenshot%20(159).png) | **Console Output** | Python IDLE terminal showing final metrics table, best model identification, and confusion matrix. |

*(Additional execution logs and intermediate charts are recorded in `Screenshot (153).png` through `Screenshot (158).png`)*

---

## 📁 Project Structure

```text
Python_project/
├── healthcare-dataset-stroke-data.csv   # Healthcare Stroke Dataset (5,110 records)
├── second.py                            # Main Python script (EDA + Data Prep + ML Benchmark)
├── requirements.txt                     # Python package dependencies
├── README.md                            # Comprehensive project documentation
├── Screenshot (148).png                 # Chart: Stroke Rate by Hypertension
├── Screenshot (149).png                 # Chart: Average Glucose Level vs Stroke Boxplot
├── Screenshot (150).png                 # Chart: Stroke Rate by Smoking Status
├── Screenshot (151).png                 # Chart: Stroke Rate by Work Type
├── Screenshot (152).png                 # Chart: Model Comparison Bar Chart
├── Screenshot (153).png                 # Terminal: Head, Info & Missing Values
├── Screenshot (154).png                 # Terminal: Statistical Summary (Describe)
├── Screenshot (155).png                 # Terminal: Objective 1 (Distribution)
├── Screenshot (156).png                 # Terminal: Objectives 2 & 3 (Age & Medical Conditions)
├── Screenshot (157).png                 # Terminal: Objectives 4 & 5 (Biomarkers & Lifestyle)
├── Screenshot (158).png                 # Terminal: SMOTE balancing & KNN Classification Report
└── Screenshot (159).png                 # Terminal: Final Model Comparison & Confusion Matrix
```

---

## 🚀 Installation & Usage

### 1. Prerequisites
- Python **3.10+** (tested on Python 3.12)
- Recommended: A virtual environment (`venv`)

### 2. Setup Environment
```bash
# Clone or navigate to the project directory
cd Python_project

# Create a virtual environment (optional but recommended)
python -m venv venv

# Activate the virtual environment
# Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# Windows (Command Prompt):
.\venv\Scripts\activate.bat
# Linux/macOS:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```
*(Or install manually: `pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn`)*

### 4. Run the Pipeline
```bash
python second.py
```
> **Note**: The script displays exploratory data charts sequentially using Matplotlib. Close each plot window to proceed to the next analysis phase until the final model comparison chart is rendered.

---

## 💡 Key Conclusions

1. **Age is the Strongest Predictor**: Stroke probability increases exponentially after age 60, peaking at nearly 20% in patients aged 81+.
2. **Cardiovascular Health Matters**: Prior history of hypertension and heart disease elevate stroke risk by **3.3x** and **4.0x**, respectively.
3. **Elevated Blood Sugar is a Major Indicator**: Average glucose levels are markedly higher in stroke patients (~132.5 mg/dL vs. ~104.8 mg/dL).
4. **Addressing Class Imbalance is Vital**: Without techniques like SMOTE, standard classifiers predict the majority class (No Stroke) for all patients, yielding high accuracy but zero utility in detecting actual stroke risk.

---

## 🔮 Future Enhancements

- [ ] **Gradient Boosting**: Experiment with XGBoost, LightGBM, and CatBoost for higher non-linear predictive capacity.
- [ ] **Hyperparameter Optimization**: Conduct `GridSearchCV` or Bayesian Optimization on Random Forest and Logistic Regression.
- [ ] **Probability Threshold Tuning**: Fine-tune decision thresholds on Logistic Regression to maximize F2-score (emphasizing recall while controlling false alarms).
- [ ] **Interactive Dashboard**: Build a Streamlit or Gradio web app for real-time patient risk profiling.
- [ ] **Feature Importance & Interpretability**: Implement SHAP (SHapley Additive exPlanations) values to explain individual patient risk predictions.

---

*Authored for the Healthcare Stroke Data Science Project.*
