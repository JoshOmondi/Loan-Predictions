# Loan Default / Approval Prediction

## Why I built this

Lending decisions affect real people's lives — whether someone gets approved for a loan can be the difference between growing a business or missing an opportunity. On the flip side, lenders need to manage risk responsibly. This project sits right at that intersection: I built a machine learning pipeline that predicts loan approval and default risk based on applicant demographic and financial data, with the aim of understanding what actually drives these decisions.

## What's in this project

- **Data cleaning** — dealing with messy, real-world applicant data (missing values, inconsistent formatting, outliers).
- **Exploratory Data Analysis (EDA)** — exploring patterns across income, credit history, employment status, and other applicant details to understand what separates approved applicants from rejected ones.
- **Feature engineering** — creating and selecting the features that carry the most predictive signal for loan eligibility, rather than just feeding in raw columns.
- **Model training & comparison** — training and comparing multiple classification algorithms, including Logistic Regression, Decision Tree, and Random Forest, to see which approach handles this problem best.
- **Evaluation** — assessing each model using accuracy, precision, recall, and confusion matrices, since in lending, the *type* of error matters as much as the overall accuracy (a false approval and a false rejection have very different real-world costs).

## Tools & libraries

- **Python**
- **Pandas** & **NumPy** — data wrangling
- **scikit-learn** — model building, training, and evaluation
- **Matplotlib** & **Seaborn** — visualization

## Getting started

```bash
# Clone the repo
git clone https://github.com/JoshOmondi/Loan-Predictions.git
cd Loan-Predictions

# Install dependencies
pip install pandas numpy scikit-learn matplotlib seaborn

# Open the notebook
jupyter notebook
```

## Thinking behind the evaluation

Accuracy alone can be misleading in a loan approval context — a model could look "accurate" while still being systematically biased toward approving or rejecting certain groups, or while making costly errors in one direction. That's why I paid close attention to precision, recall, and the confusion matrix for each model, to get a fuller picture of how each one actually behaves in practice.

## What's next

Some natural next steps would be testing the models on a more diverse or larger dataset, exploring fairness metrics to check for bias across demographic groups, and experimenting with ensemble methods to see if they improve on the individual models.

---
*Built by Josh Omondi as part of ongoing data science practice and portfolio work.*
