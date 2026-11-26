# HR-attrition
HR employee attrition analysis in Python. Found overtime staff leave at 30.5% vs 10.4%, and low-income and new hires leave most. Built a Random Forest model to predict attrition using Pandas, Seaborn, and Scikit-learn on 1,470 employee records.
# HR Employee Attrition Analysis & Prediction

An end-to-end data analysis and machine learning project that explores why employees leave a company and builds a classification model to predict attrition.

## Project Overview

Employee attrition is costly for any organization. This project analyzes the IBM HR Analytics Employee Attrition dataset to:

- Find which factors are most linked to employees leaving
- Compare attrition across departments, income levels, tenure, and overtime
- Train a Random Forest model to predict whether an employee will leave

## Dataset

- **Source:** IBM HR Analytics Employee Attrition & Performance (Kaggle)
- **File:** `WA_Fn-UseC_-HR-Employee-Attrition.csv`
- **Size:** 1,470 employees, 35 original features
- **Target:** `Attrition` (Yes / No)
- **Class balance:** 1,233 stayed (83.9%) vs 237 left (16.1%)

## Tools & Libraries

- Python
- Pandas
- Seaborn
- Matplotlib
- Scikit-learn
- Google Colab / Jupyter Notebook

## Workflow

1. **Data loading:** read the CSV with Pandas
2. **Data cleaning:** checked for missing values (none found), removed duplicates, and dropped constant columns (`EmployeeCount`, `Over18`, `StandardHours`)
3. **Exploratory data analysis:**
   - Attrition distribution (pie chart)
   - Correlation heatmap of numeric features
   - Monthly income vs attrition (bar plot)
   - Attrition rate by department, overtime, job satisfaction, income band, and tenure band
4. **Feature engineering:** created `IncomeBand` (quartiles) and `TenureBand` (0-2, 3-5, 6-10, 10+ years)
5. **Modeling:** one-hot encoded categorical variables, then trained a Random Forest classifier (`class_weight='balanced'`) on an 80/20 stratified train-test split
6. **Evaluation:** accuracy and classification report

## Key Insights

| Factor | Finding |
|---|---|
| **Overtime** | Employees working overtime leave at **30.5%** vs **10.4%** for those who don't |
| **Income** | Lowest income quartile has **29.3%** attrition vs about **10%** in the top two quartiles |
| **Tenure** | Employees with 0-2 years at the company leave at **29.8%**, dropping to **8.1%** after 10+ years |
| **Department** | Sales (20.6%) and HR (19.0%) have higher attrition than R&D (13.8%) |
| **Job Satisfaction** | Satisfaction level 1 shows **22.8%** attrition vs **11.3%** at level 4 |

## Model Results

| Metric | Value |
|---|---|
| Accuracy | 84.0% |
| Precision (Left) | 0.50 |
| Recall (Left) | 0.06 |

The model has high overall accuracy, but it mostly predicts "stayed" because the dataset is imbalanced. Its recall for employees who left is low.

## Future Improvements

- Handle class imbalance with SMOTE or threshold tuning
- Try other models (Logistic Regression, XGBoost, Gradient Boosting)
- Hyperparameter tuning with GridSearchCV
- Add feature importance analysis
- Build a Power BI / Streamlit dashboard for HR teams

## How to Run

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
pip install pandas seaborn matplotlib scikit-learn jupyter
jupyter notebook HR.ipynb
```

> Place `WA_Fn-UseC_-HR-Employee-Attrition.csv` in the project folder and update the file path in the notebook's data loading cell.

## Repository Structure

```
├── HR.ipynb
├── WA_Fn-UseC_-HR-Employee-Attrition.csv
├── filtered_data.csv
└── README.md
```

## Author

**Abinaya Anbazhagan**
B.Tech Computer Science Engineering | Data Analytics & Machine Learning
