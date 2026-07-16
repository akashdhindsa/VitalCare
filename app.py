# ============================================================
# Vital Care — Smart Healthcare Analytics
# Run with: streamlit run app.py
# Requires risk_model.pkl, scaler.pkl, feature_cols.pkl from train_model.py
# ============================================================
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime, timedelta

st.set_page_config(page_title="Vital Care", page_icon="❤️", layout="wide")

LOG_FILE = "assessment_log.csv"

# ---- Load model artifacts (auto-train on first run if missing, e.g. on Streamlit Cloud) ----
@st.cache_resource
def load_model():
    if not (os.path.exists("risk_model.pkl") and os.path.exists("scaler.pkl") and os.path.exists("feature_cols.pkl")):
        with st.spinner("First-time setup: training model, this takes a few seconds..."):
            import subprocess, sys
            result = subprocess.run([sys.executable, "train_model.py"], capture_output=True, text=True)
            if result.returncode != 0:
                st.error(f"Model training failed:\n{result.stderr}")
                st.stop()
    model = joblib.load("risk_model.pkl")
    scaler = joblib.load("scaler.pkl")
    feature_cols = joblib.load("feature_cols.pkl")
    return model, scaler, feature_cols

try:
    model, scaler, FEATURE_COLS = load_model()
except Exception as e:
    st.error(f"Could not load or train model: {e}")
    st.stop()

# ---- Healthy ranges for metric cards ----
HEALTHY_RANGES = {
    'Glucose': (70, 100, 'mg/dL'),
    'SystolicBP': (90, 120, 'mmHg'),
    'Cholesterol': (0, 200, 'mg/dL'),
    'BMI': (18.5, 24.9, ''),
}

def log_assessment(row: dict, risk_prob: float):
    """Append this assessment to the running log, used by Weekly Analysis Graph."""
    entry = {
        'date': datetime.now().strftime('%Y-%m-%d'),
        'glucose': row['Glucose'],
        'cholesterol': row['Cholesterol'],
        'blood_pressure': row['SystolicBP'],
        'bmi': row['BMI'],
        'age': row['Age'],
        'risk_probability': risk_prob,
    }
    df_new = pd.DataFrame([entry])
    if os.path.exists(LOG_FILE):
        df_new.to_csv(LOG_FILE, mode='a', header=False, index=False)
    else:
        df_new.to_csv(LOG_FILE, index=False)

# ---- Sidebar ----
with st.sidebar:
    st.markdown("### ❤️ Vital Care")
    st.caption("Smart Healthcare Analytics")
    st.markdown("---")
    st.markdown("**Navigation Menu**")
    page = st.radio(
        "Navigation",
        ["🔎 Analyse Health", "📈 Weekly Analysis Graph", "🍎 Diet Chart"],
        label_visibility="collapsed"
    )

# ============================================================
# PAGE 1 — Analyse Health
# ============================================================
if page == "🔎 Analyse Health":
    st.markdown("## Analyse Health")
    st.caption("Enter patient clinical metrics to assess health risk.")

    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("Age (years)", 18, 100, 45)
        bmi = st.slider("BMI (Body Mass Index)", 15.0, 50.0, 25.0, step=0.1)
        pregnancies = st.number_input("Pregnancies", 0, 15, 0)
        dpf = st.slider("Diabetes Pedigree Function", 0.05, 2.5, 0.5, step=0.01)
        heart_rate = st.slider("Resting Heart Rate (bpm)", 40, 150, 72)
    with col2:
        glucose = st.slider("Glucose Level (mg/dL)", 50, 300, 100)
        bp = st.slider("Systolic Blood Pressure (mmHg)", 70, 220, 120)
        cholesterol = st.slider("Cholesterol Level (mg/dL)", 100, 400, 180)
        skin_thickness = st.slider("Skin Thickness (mm)", 5.0, 60.0, 20.0, step=0.1)
        insulin = st.slider("Insulin Level (\u00b5U/mL)", 0.0, 400.0, 100.0, step=1.0)

    submitted = st.button("Analyse Health Risk", type="primary")

    if submitted:
        patient = {
            'Age': age, 'Glucose': glucose, 'BMI': bmi, 'SystolicBP': bp,
            'Pregnancies': pregnancies, 'Cholesterol': cholesterol,
            'DiabetesPedigreeFunction': dpf, 'SkinThickness': skin_thickness,
            'RestingHeartRate': heart_rate, 'Insulin': insulin,
        }
        X = pd.DataFrame([patient])[FEATURE_COLS]
        X_scaled = scaler.transform(X)
        risk_prob = model.predict_proba(X_scaled)[0][1]
        is_at_risk = risk_prob >= 0.5

        log_assessment(patient, risk_prob)

        st.markdown("---")
        left, right = st.columns([2, 1])

        with left:
            if is_at_risk:
                st.error(
                    f"**⚠️ ALERT: High Health Risk Flagged**\n\n"
                    f"Our machine learning analysis indicates that this patient exhibits "
                    f"multiple clinical risk markers. Further clinical evaluation is recommended.\n\n"
                    f"**Model Risk Probability Score: {risk_prob*100:.1f}%**"
                )
            else:
                st.success(
                    f"**✅ Low Health Risk**\n\n"
                    f"This patient's clinical metrics fall within an acceptable range based on "
                    f"our model's assessment.\n\n"
                    f"**Model Risk Probability Score: {risk_prob*100:.1f}%**"
                )

        with right:
            st.info(
                "🔔 **What to do next?**\n\n"
                "👉 Check out the **Diet Chart** tab! We have customized a nutrition plan "
                "specifically based on this patient's flagged health risk metrics."
                if is_at_risk else
                "🔔 **What to do next?**\n\nMaintain current lifestyle. Recheck periodically."
            )

        st.markdown("### 🔍 Specific Metric Insights")
        m1, m2, m3, m4 = st.columns(4)
        metric_values = {'Glucose': glucose, 'SystolicBP': bp, 'Cholesterol': cholesterol, 'BMI': bmi}
        metric_labels = {'Glucose': 'Fasting Glucose', 'SystolicBP': 'Systolic Blood Pressure',
                          'Cholesterol': 'Total Cholesterol', 'BMI': 'BMI'}
        for col, key in zip([m1, m2, m3, m4], metric_values):
            lo, hi, unit = HEALTHY_RANGES[key]
            val = metric_values[key]
            healthy = lo <= val <= hi
            with col:
                st.metric(metric_labels[key], f"{val:.1f} {unit}".strip())
                range_text = f"(Healthy: {lo}-{hi})" if key != 'Cholesterol' else f"(Healthy: < {hi})"
                if healthy:
                    st.markdown(f":green[{range_text}]")
                else:
                    st.markdown(f":red[{range_text}]")

# ============================================================
# PAGE 2 — Weekly Analysis Graph
# ============================================================
elif page == "📈 Weekly Analysis Graph":
    st.markdown("## Weekly Analysis Graph")

    if not os.path.exists(LOG_FILE):
        st.warning("No assessments logged yet. Run some analyses on the **Analyse Health** page first.")
    else:
        log = pd.read_csv(LOG_FILE)
        log['date'] = pd.to_datetime(log['date'])
        daily = log.groupby('date').agg(
            patient_count=('glucose', 'count'),
            glucose=('glucose', 'mean'),
            cholesterol=('cholesterol', 'mean'),
            blood_pressure=('blood_pressure', 'mean'),
        ).reset_index()

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### 📅 Daily Assessment Volume")
            st.bar_chart(daily.set_index('date')['patient_count'])
        with c2:
            st.markdown("#### 🩺 Vital Markers Tracking (Average by Day)")
            st.line_chart(daily.set_index('date')[['glucose', 'cholesterol', 'blood_pressure']])

        st.markdown("---")
        st.markdown("#### Raw Log")
        st.dataframe(log, use_container_width=True)

# ============================================================
# PAGE 3 — Diet Chart
# ============================================================
elif page == "🍎 Diet Chart":
    st.markdown("## Diet Chart")
    st.caption("Personalized nutrition guidance based on the most recent flagged assessment.")

    if not os.path.exists(LOG_FILE):
        st.warning("No assessments yet. Run an analysis on the **Analyse Health** page first.")
    else:
        log = pd.read_csv(LOG_FILE)
        last = log.iloc[-1]

        recs = []
        if last['glucose'] > 130:
            recs.append(("🍚 High Glucose", "Reduce refined carbs and added sugar. "
                          "Favor whole grains, legumes, and high-fiber vegetables. "
                          "Spread carbohydrate intake across smaller meals."))
        if last['blood_pressure'] > 135:
            recs.append(("🧂 High Blood Pressure", "Limit sodium intake (<2g/day). "
                          "Increase potassium-rich foods (bananas, spinach, sweet potato). "
                          "Reduce processed and packaged foods."))
        if last['cholesterol'] > 200:
            recs.append(("🥑 High Cholesterol", "Reduce saturated fat (red meat, fried food). "
                          "Increase soluble fiber (oats, beans) and healthy fats (nuts, olive oil)."))
        if last['bmi'] > 28:
            recs.append(("⚖️ Elevated BMI", "Moderate calorie deficit with balanced macronutrients. "
                          "Prioritize protein and vegetables; pair with regular physical activity."))

        if not recs:
            st.success("✅ No specific dietary flags from the most recent assessment. "
                        "Maintain a balanced diet and regular exercise.")
        else:
            for title, advice in recs:
                with st.container(border=True):
                    st.markdown(f"**{title}**")
                    st.write(advice)
