# ML-Models-for-Customer-Segmentation

# Predictive Modeling for Customer Segmentation – Personal Loan Campaign

## 📌 Project Objective

This project aims to build a predictive machine learning model to help **AllLife Bank** identify existing customers who are most likely to accept a **personal loan offer**. The objective is to improve campaign efficiency, reduce acquisition costs, and increase loan adoption by applying intelligent segmentation and targeting strategies.

---

## 🧠 Business Problem

Despite offering a full range of financial products, AllLife Bank struggles with low uptake of personal loans. Traditional blanket marketing campaigns have proven inefficient. A data-driven solution is needed to focus efforts on customers with the highest likelihood of loan acceptance.

---

## ✅ Solution Overview

A **Decision Tree Classifier** was developed to:
- Classify potential customers into “Likely to Accept” and “Not Likely to Accept” segments.
- Use customer demographic, financial, and behavioral data to derive predictive insights.
- Evaluate model performance using **Accuracy**, **Precision**, **Recall**, and **F1-Score**.

---

## 🗂 Dataset

- **Source**: Internal retail banking dataset of 5,000 customers.
- **Features**: 14 customer attributes across demographic, financial, and behavioral categories.
- **Target Variable**: `Personal_Loan` (0 = No, 1 = Yes)

### Feature Categories:
- **Demographic**: Age, Experience, ZIP Code, Family Size, Education
- **Financial**: Income, Mortgage, Credit Card Average Spend (CCAvg)
- **Product Usage**: Securities Account, CD Account, Online Banking, Credit Card
- **Target**: Personal Loan

---

## 🔍 Data Preprocessing

- Treated unrealistic values (e.g., negative experience)
- One-Hot Encoding for categorical variables
- Removed redundant features (e.g., Experience highly correlated with Age)
- Train-Test Split: 70% training / 30% testing

---

## 📈 Exploratory Data Analysis (EDA)

- **Univariate and Bivariate Analysis** conducted on all features.
- **Key Predictors Identified**:
  - Income
  - Education Level
  - Credit Card Spend (CCAvg)
  - CD Account Presence

---

## 🤖 Model Development

- **Model Used**: `DecisionTreeClassifier` from `sklearn`
- **Criterion**: Gini Impurity
- **Hyperparameter Tuning**:
  - Pre-Pruning: `max_depth`, `max_leaf_nodes`, `min_samples_split`
  - Post-Pruning: Cost-Complexity Pruning (`ccp_alpha`)
- **Model Evaluation**:
  - Confusion Matrix
  - Accuracy, Precision, Recall, F1 Score

---

## 🧪 Results & Evaluation

### ✅ Best Model: **Post-Pruned Decision Tree**

- Balanced performance on training and testing sets
- Superior generalization compared to default and pre-pruned models
- **Top Features**:
  - Income (66% importance)
  - Family Size
  - CCAvg
  - Education

---

## 📊 Key Insights

- Customers with **Income > $100K** and **CCAvg > $2.5K** are more likely to accept loans.
- **Graduate and Professional degree holders** show higher loan uptake.
- **CD Account holders** are good cross-sell targets.

---

## 💡 Business Recommendations

1. **Precision Marketing**:
   - Focus campaigns on high-income, high-spending, well-educated customers.

2. **Product Bundling**:
   - Combine loan offers with CD promotions and credit card usage.

3. **Smart Segmentation**:
   - Use decision rules to identify sub-segments based on income, family size, and education.

4. **Model Deployment**:
   - Integrate into marketing workflows and retrain periodically with new campaign data.

---

## 🧱 Additional Data Suggestions

To improve predictive performance:
- Credit Score and Credit History
- Debt and Liability Information
- Business Ownership and Employment Type
- Behavioral Data (App Usage, Website Interactions)
- Life Events (Marriage, Children, etc.)

---

## 🛠 Tech Stack

- Python
- Pandas, NumPy
- scikit-learn
- Matplotlib, Seaborn
- Jupyter Notebook

---

## 📂 Project Structure
📁 customer-loan-prediction/
├── data/
│ └── customer_data.csv
├── notebooks/
│ └── loan_modeling.ipynb
├── models/
│ └── decision_tree_model.pkl
├── visuals/
│ └── feature_importance.png
├── README.md
└── requirements.txt


---

## 📬 Contact

**Author**: Suhaib Khalid  
**License**: © Suhaib Khalid. All Rights Reserved. Unauthorized use or distribution prohibited.

---


