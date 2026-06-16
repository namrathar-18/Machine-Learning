# Lab Exercise 3 — Simple Linear Regression
**Student Awareness Survey: CIA % and Attendance % as Predictors of GPA**

---

## Files Submitted

| File | Description |
|------|-------------|
| `Student_Awareness_Survey_Responses.csv` | Raw survey dataset exported from Google Forms |
| `Lab3.ipynb` | Main Jupyter Notebook — all parts A through E |
| `linear_regression_weights.pkl` | Pickle file containing saved slope & intercept for both experiments |

---

## Dataset Description

- **Source:** Google Form filled by MCA students during the lab session  
- **Records:** 50 responses → 49 valid records after outlier removal  
- **Key columns used:**

| Column (original) | Renamed | Valid Range |
|---|---|---|
| Your CIA % of last semester | CIA | 0 – 100 |
| Your GPA of last semester | GPA | 0 – 4.0 |
| Your maximum attendance % till last semester | Attendance | 0 – 100 |

---

## Data Preprocessing Summary

1. **Loaded** raw CSV using Pandas  
2. **Selected** 3 relevant columns (CIA, GPA, Attendance)  
3. **Identified missing values** — a few NaN entries in non-key columns  
4. **Cleaned** string noise: stripped `%` symbols and extra whitespace  
5. **Converted** all columns to numeric (`pd.to_numeric`)  
6. **Removed outliers**: 1 row with GPA = 8.0 (invalid on a 4.0 scale); 1 row with CIA = 7 (likely typo for 70)  
7. **Removed duplicate records** — none found  
8. **Generated statistical summary**

---

## Regression Experiments

### Experiment 1 — CIA % → GPA
- **Equation (Scikit-learn):** `GPA = 0.012800 × CIA% + 2.483380`
- **Equation (Manual OLS):**   `GPA = 0.012800 × CIA% + 2.483380`
- Difference between both methods: **4.44 × 10⁻¹⁶** (machine epsilon — identical)

### Experiment 2 — Attendance % → GPA
- **Equation (Scikit-learn):** `GPA = 0.024410 × Attendance% + 1.140655`
- **Equation (Manual OLS):**   `GPA = 0.024410 × Attendance% + 1.140655`
- Difference between both methods: **4.44 × 10⁻¹⁶** (machine epsilon — identical)

---

## Key Observations

1. **Scikit-learn and Manual OLS are mathematically equivalent.** Any numerical difference is at floating-point machine precision (~10⁻¹⁶), not algorithmic.
2. **CIA % shows a mild positive relationship with GPA** (slope ≈ 0.013). A student scoring 10% higher in CIA is predicted to gain ~0.13 GPA points.
3. **Attendance % has a slightly stronger slope** (≈ 0.024), but the attendance data is clustered in a narrow 85–100% range, limiting variance.
4. **Data cleaning was essential** — raw responses had inconsistent formats that would have prevented numeric computation without preprocessing.
5. **Saving model parameters with Pickle** allows re-use of learned weights without re-training, which is standard practice in ML deployment.

---

## How to Run

```bash
# 1. Make sure the CSV is in the same folder as the notebook
# 2. Open the notebook
jupyter notebook Lab3.ipynb

# 3. Run all cells top to bottom (Kernel → Restart & Run All)
```

**Dependencies:**
```
pandas
numpy
matplotlib
scikit-learn
```

---

## Viva Q&A Reference

| Question | Short Answer |
|----------|--------------|
| What is Simple Linear Regression? | A statistical method that models the linear relationship between one independent variable (X) and a dependent variable (Y): Y = mX + b |
| What is the role of slope and intercept? | Slope (m) = rate of change of Y per unit change in X. Intercept (b) = predicted Y when X = 0 |
| What is Ordinary Least Squares? | An algorithm that finds the line minimising the sum of squared differences between actual and predicted values |
| Why do we square the errors in OLS? | To avoid positive and negative errors cancelling out, and to penalise large errors more heavily |
| Dependent vs Independent variable | Independent (X) is the input/predictor; Dependent (Y) is the output being predicted |
| Why clean data before training? | Dirty data (nulls, wrong types, outliers) leads to incorrect model parameters and unreliable predictions |
| Why are slope and intercept called model parameters? | They are the values the model learns from data and fully define the regression line |
| Why do we save learned weights? | To reuse the model for future predictions without re-training |
