"""
models.py
Defines ML models and cross-validation routines.
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.metrics import roc_auc_score, log_loss

def train_logistic_regression(X, y):
    lr = LogisticRegression(max_iter=500)
    lr.fit(X, y)
    return lr

def train_random_forest(X, y):
    rf = RandomForestClassifier(n_estimators=50, random_state=42)
    rf.fit(X, y)
    return rf

def train_xgboost(X, y):
    model = xgb.XGBClassifier(n_estimators=50, use_label_encoder=False, eval_metric='logloss')
    model.fit(X, y)
    return model

def evaluate_model(model, X, y):
    prob = model.predict_proba(X)[:,1]
    auc = roc_auc_score(y, prob)
    loss = log_loss(y, prob)
    return {"AUC": auc, "LogLoss": loss}