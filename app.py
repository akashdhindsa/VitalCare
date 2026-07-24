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
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(page_title="Vital Care", page_icon="🩸", layout="wide", initial_sidebar_state="expanded")

# ============================================================
# THEME — dark/moody, red accent. Widget colors come from
# .streamlit/config.toml; this CSS handles everything the
# native theme engine can't touch (fonts, cards, chrome).
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@600;800&family=JetBrains+Mono:wght@400;600&display=swap');

/* Footer branding removed. Header/toolbar left untouched on purpose —
   don't risk hiding the sidebar toggle again. */
footer {visibility: hidden;}

/* Base app background — deep black, not pure Streamlit dark */
.stApp {
    background: radial-gradient(circle at 15% 0%, #1a0d0f 0%, #0a0a0a 45%, #050505 100%);
}

/* Headings in a sharper display font */
h1, h2, h3 {
    font-family: 'Orbitron', sans-serif !important;
    letter-spacing: 0.5px;
}

h1 {
    color: #e63946 !important;
    text-shadow: 0 0 18px rgba(230, 57, 70, 0.35);
}

/* Numeric/metric values in monospace for that dashboard feel */
[data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace !important;
    color: #f1f1f1 !important;
}
[data-testid="stMetricLabel"] {
    color: #9a9a9a !important;
    text-transform: uppercase;
    font-size: 0.75rem !important;
    letter-spacing: 1px;
}

/* Card container styling (Diet Chart uses st.container(border=True)) */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: #121212;
    border: 1px solid #2a2a2a !important;
    border-radius: 10px;
    box-shadow: 0 0 0 1px rgba(230,57,70,0.05), 0 4px 20px rgba(0,0,0,0.4);
    transition: border-color 0.2s ease;
}
div[data-testid="stVerticalBlockBorderWrapper"]:hover {
    border-color: rgba(230, 57, 70, 0.5) !important;
}

/* Primary button — sharp, no rounded corporate pill */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #e63946, #a4161a);
    border: none;
    border-radius: 4px;
    font-weight: 700;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    font-size: 0.85rem;
    box-shadow: 0 4px 14px rgba(230, 57, 70, 0.3);
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}
.stButton > button[kind="primary"]:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 20px rgba(230, 57, 70, 0.5);
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0d0d0d;
    border-right: 1px solid #221111;
}
[data-testid="stSidebar"] .stRadio label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.9rem;
}

/* Slider accent */
.stSlider [data-baseweb="slider"] div[role="slider"] {
    background-color: #e63946 !important;
}

/* Alert boxes — keep semantics (red=risk, green=ok) but darken backgrounds */
[data-testid="stAlertContentError"], .stAlert:has(> div[data-baseweb="notification"]) {
    border-radius: 8px;
}

/* Divider */
hr {
    border-color: #221111 !important;
}
</style>
""", unsafe_allow_html=True)

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

def risk_gauge(risk_prob: float):
    """Semicircular gauge for the risk probability score."""
    pct = risk_prob * 100
    color = "#e63946" if pct >= 50 else "#2ecc71"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pct,
        number={'suffix': "%", 'font': {'size': 40, 'color': '#f1f1f1', 'family': 'JetBrains Mono'}},
        gauge={
            'axis': {'range': [0, 100], 'tickcolor': '#555', 'tickfont': {'color': '#888'}},
            'bar': {'color': color, 'thickness': 0.3},
            'bgcolor': "#161616",
            'borderwidth': 0,
            'steps': [
                {'range': [0, 50], 'color': 'rgba(46, 204, 113, 0.12)'},
                {'range': [50, 100], 'color': 'rgba(230, 57, 70, 0.12)'},
            ],
            'threshold': {'line': {'color': "#f1f1f1", 'width': 3}, 'thickness': 0.8, 'value': 50},
        }
    ))
    fig.update_layout(
        height=220,
        margin=dict(l=20, r=20, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font={'color': "#eaeaea"},
    )
    return fig

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
    st.markdown("### 🩸 VITAL // CARE")
    st.caption("SMART HEALTHCARE ANALYTICS")
    st.markdown("---")
    st.markdown("**NAVIGATION**")
    page = st.radio(
        "Navigation",
        ["🔎 Analyse Health", "📈 Weekly Analysis Graph", "🍎 Diet Chart"],
        label_visibility="collapsed"
    )

# ============================================================
# PAGE 1 — Analyse Health
# ============================================================
if page == "🔎 Analyse Health":
    st.markdown("""
    <div style="padding: 1.2rem 1.5rem; border-radius: 10px; margin-bottom: 1.2rem;
                background: linear-gradient(120deg, rgba(230,57,70,0.12), rgba(10,10,10,0));
                border-left: 3px solid #e63946;">
        <h2 style="margin:0; padding:0;">ANALYSE HEALTH</h2>
        <p style="color:#9a9a9a; margin:0.3rem 0 0 0;">Enter patient clinical metrics below — grouped by body system — to generate a risk score.</p>
    </div>
    """, unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown("#### 🫀 Vitals & Body")
        v1, v2, v3 = st.columns(3)
        with v1:
            age = st.slider("Age (years)", 18, 100, 45)
        with v2:
            bmi = st.slider("BMI (Body Mass Index)", 15.0, 50.0, 25.0, step=0.1)
        with v3:
            heart_rate = st.slider("Resting Heart Rate (bpm)", 40, 150, 72)

    with st.container(border=True):
        st.markdown("#### 🩸 Metabolic Markers")
        m1, m2, m3 = st.columns(3)
        with m1:
            glucose = st.slider("Glucose Level (mg/dL)", 50, 300, 100)
        with m2:
            bp = st.slider("Systolic Blood Pressure (mmHg)", 70, 220, 120)
        with m3:
            cholesterol = st.slider("Cholesterol Level (mg/dL)", 100, 400, 180)

    with st.container(border=True):
        st.markdown("#### 📋 History & Other Markers")
        h1, h2, h3, h4 = st.columns(4)
        with h1:
            pregnancies = st.number_input("Pregnancies", 0, 15, 0)
        with h2:
            dpf = st.slider("Diabetes Pedigree Fn.", 0.05, 2.5, 0.5, step=0.01)
        with h3:
            skin_thickness = st.slider("Skin Thickness (mm)", 5.0, 60.0, 20.0, step=0.1)
        with h4:
            insulin = st.slider("Insulin (\u00b5U/mL)", 0.0, 400.0, 100.0, step=1.0)

    st.markdown("")
    submitted = st.button("Analyse Health Risk", type="primary", use_container_width=True)

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
        gauge_col, verdict_col, next_col = st.columns([1.1, 1.3, 1.1])

        with gauge_col:
            st.plotly_chart(risk_gauge(risk_prob), use_container_width=True, config={'displayModeBar': False})

        with verdict_col:
            if is_at_risk:
                st.markdown("""
                <div style="background:#1a0f10; border:1px solid #e63946; border-radius:8px;
                            padding:1rem 1.2rem; height:220px; display:flex; flex-direction:column; justify-content:center;">
                    <span style="color:#e63946; font-weight:700; font-family:'Orbitron',sans-serif;">⚠️ HIGH RISK FLAGGED</span>
                    <p style="color:#c9c9c9; margin-top:0.6rem; font-size:0.9rem;">
                        Multiple clinical risk markers detected. Further clinical evaluation is recommended.
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="background:#0f1a12; border:1px solid #2ecc71; border-radius:8px;
                            padding:1rem 1.2rem; height:220px; display:flex; flex-direction:column; justify-content:center;">
                    <span style="color:#2ecc71; font-weight:700; font-family:'Orbitron',sans-serif;">✅ LOW RISK</span>
                    <p style="color:#c9c9c9; margin-top:0.6rem; font-size:0.9rem;">
                        Clinical metrics fall within an acceptable range based on the model's assessment.
                    </p>
                </div>
                """, unsafe_allow_html=True)

        with next_col:
            st.info(
                "🔔 **What to do next?**\n\n"
                "👉 Check out the **Diet Chart** tab! We have customized a nutrition plan "
                "specifically based on this patient's flagged health risk metrics."
                if is_at_risk else
                "🔔 **What to do next?**\n\nMaintain current lifestyle. Recheck periodically."
            )

        st.markdown("### 🔍 SPECIFIC METRIC INSIGHTS")
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
    st.markdown("## WEEKLY ANALYSIS GRAPH")

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
            st.bar_chart(daily.set_index('date')['patient_count'], color="#e63946")
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
    st.markdown("## DIET CHART")
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
