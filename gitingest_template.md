# Interpretable ML: SHAP Explanations for Telco Churn Prediction

**Dataset:** WA_Fn-UseC_-Telco-Customer-Churn.csv

## Model & Training
- Model: XGBoost via scikit-learn Pipeline
- Best hyperparameters (RandomizedSearchCV): {'subsample': 1.0, 'reg_lambda': 5, 'reg_alpha': 1, 'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.1, 'colsample_bytree': 0.7}
- Test AUC: 0.8402, Test Accuracy: 0.7974, Test F1: 0.5803

## Outputs in /content/churn_project_outputs
- Trained pipeline: xgb_pipeline.joblib
- Global SHAP plots: shap_summary_bar.png, shap_summary_beeswarm.png
- Local SHAP: local_shap_waterfall_idx_*.png and local_shap_force_idx_*.html
- Selected customers CSV: selected_customers.csv

## Instructions
Open the HTML force plots in a web browser to inspect interactive explanations. Include the PNGs and your textual interpretations in the submission box.

