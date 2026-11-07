# Customer Churn Predictor
Live app: https://churn-risk-checker.streamlit.app/

Predicts which telecom customers are likely to leave, and suggests actions to keep them.

## Key findings
- Month-to-month customers churn at 43%, compared with 11% on one-year and 3% on two-year contracts
- Most churn happens early: 47% of customers in their first 12 months leave
- Electronic check users churn at 45%, versus 15 to 17% for automatic payers
- Fiber optic customers churn at 42%, versus 19% on DSL
- Customers without tech support churn at 42%, versus 15% with it

## Model
Logistic Regression with `class_weight="balanced"`, catching 78% of customers who leave (recall) on unseen test data.

| Model (balanced) | Recall | Precision |
|---|---|---|
| Logistic Regression | 78% | 50% |
| Decision Tree | 76% | 53% |
| Random Forest | 62% | 55% |

Without balancing, every model missed about half of the churners (recall 48 to 57%). About half of the customers the model flags will not leave. That is an accepted trade-off: a wasted retention call costs less than a lost customer.

## Business recommendations
1. Move month-to-month customers to 1 or 2-year contracts with a discount or free add-on
2. Protect the first year with check-in calls and early offers to high-risk customers
3. Investigate fiber optic quality and price, and bundle tech support or online security
4. Encourage electronic check payers to switch to automatic payment

## Files
- `app.py`: the Streamlit app (trains the model on start-up)
- `Churn_Project.ipynb`: cleaning, EDA, model comparison, tuning and evaluation
- `telco_churn.csv`: the dataset
- `requirements.txt`: Python packages

## Run it yourself
```
pip install -r requirements.txt
streamlit run app.py
```

## Built with
Python, pandas, scikit-learn, Streamlit. Data from Kaggle (Telco Customer Churn).
