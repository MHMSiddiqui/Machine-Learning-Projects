# Telco Customer Churn Prediction and Driver Analysis

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Jupyter Notebook](https://img.shields.io/badge/Tools-Jupyter%20Notebook-orange)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Complete-success)

## 📌 Project Overview

This project focuses on predicting customer churn in the telecommunications industry using machine learning techniques. By analyzing customer demographics, service usage, and account information, the goal is to identify key drivers of churn and build a predictive model to help businesses retain customers proactively.

The project encompasses the entire data science pipeline:
1.  **Data Ingestion & Cleaning:** Handling missing values and preparing the dataset.
2.  **Exploratory Data Analysis (EDA):** Visualizing distributions and correlations to understand data patterns.
3.  **Feature Engineering:** Creating new features like `TotalAddonServices` to improve model performance.
4.  **Data Preprocessing:** Encoding categorical variables and scaling numerical features.
5.  **Model Building:** Training and tuning multiple classifiers (Logistic Regression, Random Forest, Gradient Boosting).
6.  **Model Evaluation:** Assessing performance using Accuracy, Precision, Recall, F1-Score, and ROC-AUC.

---

## 📂 Dataset

The dataset used is the [Telco Customer Churn](https://www.kaggle.com/blastchar/telco-customer-churn) dataset.

* **Rows:** 7,043
* **Columns:** 21
* **Target Variable:** `Churn` (Yes/No)

### Key Features:
* **Demographics:** Gender, Senior Citizen, Partner, Dependents.
* **Services:** Phone Service, Multiple Lines, Internet Service, Online Security, Tech Support, Streaming TV/Movies.
* **Account Info:** Tenure, Contract, Paperless Billing, Payment Method, Monthly Charges, Total Charges.

---

## 🛠️ Tech Stack

* **Language:** Python
* **Libraries:**
    * `Pandas`, `NumPy` (Data Manipulation)
    * `Matplotlib`, `Seaborn` (Visualization)
    * `Scikit-learn` (Machine Learning & Preprocessing)

---

## 📊 Methodology

### 1. Data Cleaning
* Handled missing values in the `TotalCharges` column by replacing empty strings with `NaN` and imputing with the median.
* Converted the target variable `Churn` into a binary numeric format (1 for Yes, 0 for No).
* Removed irrelevant identifiers like `customerID`.

### 2. Exploratory Data Analysis (EDA)
* **Univariate Analysis:** Histograms and boxplots were used to examine the distribution of numerical features like `tenure` and `MonthlyCharges`.
* **Bivariate Analysis:** Count plots visualized churn rates across categorical features (e.g., Contract type, Payment Method).
* **Correlation Analysis:** A heatmap was generated to identify multicollinearity among numerical variables.

### 3. Feature Engineering
* **New Feature:** `TotalAddonServices` was created by summing up active add-on services (e.g., Online Security, Tech Support) to capture a customer's engagement level.

### 4. Machine Learning Models
The data was split into training (80%) and testing (20%) sets. The following models were evaluated:

| Model | Accuracy | ROC-AUC | Key Strengths |
| :--- | :---: | :---: | :--- |
| **Logistic Regression** | **81.69%** | **0.86** | Simple, interpretable, good baseline performance. |
| **Gradient Boosting** | 80.91% | 0.85 | Strong predictive power, captures complex patterns. |
| **Random Forest** | 79.13% | 0.84 | Robust to overfitting, handles non-linear data well. |

---

## 📈 Key Findings

1.  **Contract Type:** Customers with month-to-month contracts have a significantly higher churn rate compared to those with one or two-year contracts.
2.  **Tenure:** Newer customers are more likely to churn. As tenure increases, churn probability decreases.
3.  **Payment Method:** Electronic check users show the highest churn rate among all payment methods.
4.  **Internet Service:** Fiber optic users churn more frequently than DSL users, possibly due to higher costs or service issues.
5.  **Add-on Services:** Customers with fewer add-on services (like Tech Support or Online Security) are more prone to churning.

---

## 🚀 How to Run the Project

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/yourusername/telco-churn-prediction.git](https://github.com/yourusername/telco-churn-prediction.git)
    ```
2.  **Navigate to the project directory:**
    ```bash
    cd telco-churn-prediction
    ```
3.  **Install dependencies:**
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn
    ```
4.  **Open the Notebook:**
    Launch Jupyter Notebook and open `Logistic_Regression_Project.ipynb`.

---

## 🔮 Future Improvements

* **Hyperparameter Tuning:** Use GridSearch or RandomSearch to optimize model parameters further.
* **SMOTE:** Implement Synthetic Minority Over-sampling Technique to handle class imbalance in the target variable.
* **Deep Learning:** Experiment with neural networks (e.g., ANN) to potentially improve accuracy.
