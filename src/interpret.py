"""
interpret.py
Model explainability using SHAP.
"""

import shap
import matplotlib.pyplot as plt

def shap_summary_plot(model, X, out_file="shap_summary.png"):
    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)
    shap.summary_plot(shap_values, X, show=False)
    plt.savefig(out_file)
    print(f"SHAP summary plot saved to {out_file}")