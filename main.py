"""
main.py
Main pipeline for the Market Mood Agent project.

Steps:
1. Download and preprocess Numerai data
2. Split data into train/validation sets
3. Train multiple ML models
4. Ensemble predictions
5. Explain model decisions (SHAP)
6. Generate market mood signals
7. Simulate trading/backtest
8. Save results for dashboard
"""

import os
from src.data_utils import download_numerai_data, load_data, preprocess_data, split_by_era
from src.models import train_logistic_regression, train_random_forest, train_xgboost, evaluate_model
from src.ensemble import average_ensemble
from src.interpret import shap_summary_plot
from src.backtest import generate_signals, backtest

def main():
    # 1. Download & Load Data
    if not os.path.exists("data/raw/train.parquet"):
        download_numerai_data()
    df = load_data()
    df = preprocess_data(df)

    # 2. Split Data (by era)
    train_df, val_df = split_by_era(df)
    X_train = train_df.drop(["target", "era"], axis=1)
    y_train = train_df["target"]
    X_val = val_df.drop(["target", "era"], axis=1)
    y_val = val_df["target"]

    # 3. Train Models
    print("Training Logistic Regression...")
    lr_model = train_logistic_regression(X_train, y_train)
    print("Training Random Forest...")
    rf_model = train_random_forest(X_train, y_train)
    print("Training XGBoost...")
    xgb_model = train_xgboost(X_train, y_train)

    # 4. Model Evaluation
    models = {'LR': lr_model, 'RF': rf_model, 'XGB': xgb_model}
    for name, model in models.items():
        metrics = evaluate_model(model, X_val, y_val)
        print(f"{name} Validation Metrics: {metrics}")

    # 5. Ensemble
    lr_preds = lr_model.predict_proba(X_val)[:,1]
    rf_preds = rf_model.predict_proba(X_val)[:,1]
    xgb_preds = xgb_model.predict_proba(X_val)[:,1]
    ensemble_preds = average_ensemble([lr_preds, rf_preds, xgb_preds])

    # 6. Explain (on validation set, XGB model shown as example)
    shap_summary_plot(xgb_model, X_val, out_file="data/processed/shap_summary.png")

    # 7. Generate Signals
    signals = generate_signals(ensemble_preds)
    print(f"Sample signals: {signals[:10]}")

    # 8. Backtest
    returns_df = backtest(signals, y_val)
    returns_df.to_csv("data/processed/returns.csv", index=False)
    print("Cumulative returns head:")
    print(returns_df.head())

    # 9. Save processed data for dashboard
    val_df_out = val_df.copy()
    val_df_out["ensemble_pred"] = ensemble_preds
    val_df_out["signal"] = signals
    val_df_out.to_csv("data/processed/val_with_signals.csv", index=False)
    print("Saved validation predictions & signals for dashboard.")

if __name__ == "__main__":
    main()