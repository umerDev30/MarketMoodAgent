"""
backtest.py
Simulate trading based on model/ensemble signals.
"""

import numpy as np
import pandas as pd

def generate_signals(pred_probs, bullish_thresh=0.6, bearish_thresh=0.4):
    signals = []
    for p in pred_probs:
        if p >= bullish_thresh:
            signals.append("Bullish")
        elif p <= bearish_thresh:
            signals.append("Bearish")
        else:
            signals.append("Neutral")
    return signals

def backtest(signals, labels, long_ret=0.01, short_ret=0.01):
    returns = []
    for sig, lbl in zip(signals, labels):
        if sig == "Bullish":
            returns.append(long_ret if lbl == 1 else -long_ret)
        elif sig == "Bearish":
            returns.append(short_ret if lbl == 0 else -short_ret)
        else:
            returns.append(0)
    df = pd.DataFrame({"returns": returns})
    df["cumulative"] = df["returns"].cumsum()
    return df