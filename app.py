import streamlit as st
import matplotlib.pyplot as plt
import sys, os

# Allow importing from the scripts folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.load_data import load_data
from scripts.anomaly_detector import detect_anomalies

# Page config
st.set_page_config(page_title="Business Insight Engine", layout="wide")
st.title("📊 Business Insight Engine - Real-Time Anomaly Detection")

# Load and display data
data_path = "data/ec2_cpu.csv"
df = load_data(data_path)
st.write("✅ Data loaded successfully")

# Run anomaly detection
df, model = detect_anomalies(df)
st.success("Anomaly detection completed using Isolation Forest.")

# Plot anomalies
st.subheader("📈 CPU Utilization with Detected Anomalies")
fig, ax = plt.subplots(figsize=(15, 6))
ax.plot(df['timestamp'], df['value'], label='CPU Utilization')
ax.scatter(df[df['anomaly'] == 1]['timestamp'], df[df['anomaly'] == 1]['value'],
           color='red', label='Anomalies')
plt.xticks(rotation=45)
plt.xlabel("Timestamp")
plt.ylabel("Value")
plt.legend()
st.pyplot(fig)

# Show top anomalies
st.subheader("🚨 Detected Anomalies")
st.dataframe(df[df['anomaly'] == 1][['timestamp', 'value']].head(10))

# SHAP Explanation
if st.button("🔍 Explain Top Anomaly with SHAP"):
    from scripts.explainer import explain
    first_anomaly = df[df['anomaly'] == 1]
    if not first_anomaly.empty:
        st.info("Generating SHAP explanation (check popup window or notebook cell)...")
        explain(model, first_anomaly)
    else:
        st.warning("No anomalies to explain.")
