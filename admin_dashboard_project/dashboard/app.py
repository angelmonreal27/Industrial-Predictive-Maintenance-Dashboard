import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix


# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(page_title="Industrial Dashboard", layout="wide")

st.title("Industrial Predictive Maintenance Dashboard")
st.markdown("### Smart Machine Failure Monitoring System")
st.markdown("---")


# -------------------------
# LOAD DATA
# -------------------------
@st.cache_data
def load_data():
    return pd.read_csv("data/ai4i2020.csv")

df = load_data()


# -------------------------
# LOAD MODEL + FEATURES + SCALER
# -------------------------
model = joblib.load("models/production_model.pkl")
features = joblib.load("models/features.pkl")

try:
    scaler = joblib.load("models/scaler.pkl")
    use_scaler = True
except:
    use_scaler = False


# -------------------------
# SIDEBAR
# -------------------------
st.sidebar.header("⚙️ Control Panel")
threshold = st.sidebar.slider("Failure Risk Threshold", 0.0, 1.0, 0.3)

show_high_risk_only = st.sidebar.checkbox("Show High Risk Only Data")

view_mode = st.sidebar.selectbox(
    "View Mode",
    ["Overview", "High Risk", "All Data"]
)


# -------------------------
# PREPROCESS
# -------------------------
target = "Machine failure"
drop_cols = [target, 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']

X = pd.get_dummies(df.drop(columns=drop_cols), drop_first=True)
X = X.reindex(columns=features, fill_value=0)

y = df[target]

if use_scaler:
    X = scaler.transform(X)


# -------------------------
# PREDICTIONS
# -------------------------
probs = model.predict_proba(X)[:, 1]
preds = (probs >= threshold).astype(int)

# -------------------------
# DATA VIEW FILTERING
# -------------------------
if view_mode == "High Risk":
    display_df = df[probs > threshold]
elif view_mode == "All Data":
    display_df = df
else:
    display_df = df.head(10)

# -------------------------
# 🚨 LIVE ANOMALY ALERT SYSTEM (AUTO)
# -------------------------
out_of_control = probs > threshold
high_risk_count = np.sum(out_of_control)

st.subheader("🚨 Live Anomaly Alert System")

if high_risk_count > 0:
    st.error(f"⚠️ ALERT: {high_risk_count} high-risk machines detected!")
    st.warning("Immediate inspection recommended for affected machines.")
else:
    st.success("🟢 System Stable: No anomalies detected.")

# -------------------------
# STATUS MESSAGE
# -------------------------
if probs.mean() > 0.5:
    st.error("🚨 HIGH RISK: System showing elevated failure probability")
else:
    st.success("🟢 System Operating Normally")


st.markdown("---")


# -------------------------
# KPIs SECTION
# -------------------------
st.subheader("📊 Model Evaluation Metrics (Trained Logistic Regression)")

acc = accuracy_score(y, preds)
prec = precision_score(y, preds, zero_division=0)
rec = recall_score(y, preds, zero_division=0)

col1, col2, col3, col4 = st.columns(4)

col1.metric("Accuracy", f"{acc:.3f}")
col2.metric("Precision", f"{prec:.3f}")
col3.metric("Recall", f"{rec:.3f}")
col4.metric("Failure Rate", f"{y.mean()*100:.2f}%")


st.metric("📈 Avg Risk Score", f"{probs.mean():.3f}")

st.markdown("---")

# -------------------------
# PRODUCTION TREND LINE CHART
# -------------------------
st.subheader("📈 Production Trend Line Chart")

fig_trend = px.line(
    x=df.index,
    y=probs,
    labels={"x": "Machine Index", "y": "Risk Probability"},
    title="Production Risk Trend Over Time"
)

st.plotly_chart(fig_trend, use_container_width=True)

st.markdown("---")

# -------------------------
# ACTUAL VS PREDICTED
# -------------------------
st.subheader("📈 Actual vs Predicted Risk")

fig1 = px.strip(
    x=y,
    y=probs,
    title="Actual vs Predicted Risk"
)

st.plotly_chart(fig1, use_container_width=True)

st.markdown("---")


# -------------------------
# CONFUSION MATRIX
# -------------------------
st.subheader("📊 Confusion Matrix")

cm = confusion_matrix(y, preds)

fig2 = px.imshow(cm, text_auto=True, color_continuous_scale="Purples")

st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")


# -------------------------
# MODEL COEFFICIENT INTERPRETATION
# -------------------------
st.subheader("📉 Feature Importance / Coefficient Plot")

coef_df = pd.DataFrame({
    "Feature": features,
    "Coefficient": model.coef_[0]
})

coef_df = coef_df.sort_values("Coefficient")

st.dataframe(coef_df)

fig = px.bar(
    coef_df,
    x="Coefficient",
    y="Feature",
    orientation="h",
    title="Feature Impact on Machine Failure (Coefficients)"
)

st.plotly_chart(fig, use_container_width=True)

st.markdown("---")


# -------------------------
# CONTROL CHART
# -------------------------
st.subheader("⚠️ Control Chart (3-Sigma)")
mean = probs.mean()
std = probs.std()

ucl = mean + 3 * std
lcl = max(0, mean - 3 * std)

fig4 = go.Figure()

fig4.add_trace(go.Scatter(y=probs, mode="lines", name="Risk"))

out_of_control = (probs > ucl) | (probs < lcl)

fig4.add_trace(go.Scatter(
    y=np.where(out_of_control, probs, None),
    mode="markers",
    name="Out of Control",
    marker=dict(color="red", size=8)
))

fig4.add_hline(y=mean, line_color="green", annotation_text="Mean")
fig4.add_hline(y=ucl, line_dash="dash", line_color="red", annotation_text="UCL (3σ)")
fig4.add_hline(y=lcl, line_dash="dash", line_color="red", annotation_text="LCL (3σ)")

fig4.update_layout(title="3-Sigma Control Chart (Risk Monitoring)")

st.plotly_chart(fig4, use_container_width=True)

st.markdown("---")


# -------------------------
# HIGH RISK TABLE
# -------------------------
st.subheader("🚨 High Risk Machines")

high_risk = df[probs > threshold]

st.write(f"Detected: {len(high_risk)} high-risk machines")

st.dataframe(display_df.head(10))


# -------------------------
# EXPORT
# -------------------------
st.subheader("📤 Export Filtered Data")

# create export dataset
export_df = df.copy()
export_df["Risk Probability"] = probs
export_df["Prediction"] = preds

# optional: filtered high risk only
filtered_df = export_df[export_df["Risk Probability"] > threshold]

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Full Dataset Export")
    st.download_button(
        label="Download Full CSV",
        data=export_df.to_csv(index=False),
        file_name="full_machine_report.csv",
        mime="text/csv"
    )

with col2:
    st.markdown("### High Risk Only Export")
    st.download_button(
        label="Download Filtered CSV",
        data=filtered_df.to_csv(index=False),
        file_name="high_risk_machines.csv",
        mime="text/csv"
    )

st.markdown("---")