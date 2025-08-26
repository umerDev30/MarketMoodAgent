"""
streamlit_app.py
Streamlit dashboard: visualizes signals, returns, model insights.
"""

import streamlit as st
import pandas as pd

def show_dashboard(df, signals, cum_returns):
    st.title("Market Mood Agent Dashboard")
    st.subheader("Latest Signal")
    st.write(f"Signal: {signals[-1]}")
    
    st.subheader("Signals Over Time")
    st.line_chart(pd.Series(signals).value_counts())
    
    st.subheader("Cumulative Returns")
    st.line_chart(cum_returns)
    
    st.subheader("Sample Data")
    st.dataframe(df.head())

if __name__ == "__main__":
    # Placeholder demo: use real data in your main pipeline!
    st.write("Dashboard will appear here after running main pipeline.")