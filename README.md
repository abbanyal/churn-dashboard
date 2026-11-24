# Bank Customer Churn: Predictive Modeling and Risk Scoring

Predicts the probability that a retail bank customer will churn, and explains the drivers.

**Live dashboard:** https://churn-riskdashboard.streamlit.app/

## Results
- Dataset: 10,000 customers, 20.4% churn rate
- Best model: Gradient Boosting, ROC-AUC 0.869
- At threshold 0.3: precision 0.625, recall 0.646, F1 0.635
- Top drivers: number of products, age, geography (Germany), activity status

## Dashboard features
- Customer churn risk calculator
- What-if scenario simulator
- Feature importance and probability distribution

## Files
- `app.py`: Streamlit dashboard
- `model.joblib`: trained Gradient Boosting pipeline
- `test_probs.csv`: test-set predictions used by the dashboard
- `analysis.ipynb`: data analysis, modelling and explainability
- `figures/`: charts used in the paper

## Run locally
pip install -r requirements.txt
streamlit run app.py
