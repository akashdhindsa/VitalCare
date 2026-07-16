# ============================================================
# Vital Care — Model Training Script (FINAL, real dataset)
# Dataset: test_healthcare_data.csv (provided by training program)
#   Real columns: age, bmi, glucose, blood_pressure, skin_thickness,
#                 insulin, pregnancies, diabetes_pedigree, cholesterol,
#                 heart_rate  (1000 rows, no missing values, no target)
# ============================================================
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import xgboost as xgb
import joblib

# Map real column names -> app's internal names
COLUMN_MAP = {
    'age': 'Age', 'bmi': 'BMI', 'glucose': 'Glucose',
    'blood_pressure': 'SystolicBP', 'skin_thickness': 'SkinThickness',
    'insulin': 'Insulin', 'pregnancies': 'Pregnancies',
    'diabetes_pedigree': 'DiabetesPedigreeFunction',
    'cholesterol': 'Cholesterol', 'heart_rate': 'RestingHeartRate'
}

FEATURE_COLS = [
    'Age', 'Glucose', 'BMI', 'SystolicBP', 'Pregnancies',
    'Cholesterol', 'DiabetesPedigreeFunction', 'SkinThickness',
    'RestingHeartRate', 'Insulin'
]

df = pd.read_csv('test_healthcare_data.csv').rename(columns=COLUMN_MAP)
print("Loaded:", df.shape)
assert df.isnull().sum().sum() == 0, "Unexpected missing values — check the CSV."

# ---- Target label: rule-based risk_level (2+ risk factors = At Risk) ----
# Same rule confirmed from the training program's own notebook.
glucose_risk = (df['Glucose'] > 130).astype(int)
bp_risk = (df['SystolicBP'] > 135).astype(int)
cholesterol_risk = (df['Cholesterol'] > 200).astype(int)
bmi_risk = (df['BMI'] > 28).astype(int)
age_risk = (df['Age'] > 60).astype(int)

risk_score = glucose_risk + bp_risk + cholesterol_risk + bmi_risk + age_risk
df['risk_level'] = (risk_score >= 2).astype(int)
print("Risk distribution:\n", df['risk_level'].value_counts(normalize=True))

# ---- Train/test split ----
X = df[FEATURE_COLS]
y = df['risk_level']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---- Train Logistic Regression (used live in the app) ----
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]
print(f"\nLogistic Regression (deployed model):")
print(f"  Accuracy:  {accuracy_score(y_test, y_pred):.3f}")
print(f"  Precision: {precision_score(y_test, y_pred):.3f}")
print(f"  Recall:    {recall_score(y_test, y_pred):.3f}")
print(f"  F1:        {f1_score(y_test, y_pred):.3f}")
print(f"  ROC-AUC:   {roc_auc_score(y_test, y_prob):.3f}")

# ---- Also train RF + XGBoost for report comparison table (not deployed) ----
rf = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
rf.fit(X_train, y_train)
rf_prob = rf.predict_proba(X_test)[:, 1]
print(f"\nRandom Forest (report comparison only):")
print(f"  Accuracy: {accuracy_score(y_test, rf.predict(X_test)):.3f}  ROC-AUC: {roc_auc_score(y_test, rf_prob):.3f}")

xgb_model = xgb.XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                                use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb_model.fit(X_train, y_train)
xgb_prob = xgb_model.predict_proba(X_test)[:, 1]
print(f"\nXGBoost (report comparison only):")
print(f"  Accuracy: {accuracy_score(y_test, xgb_model.predict(X_test)):.3f}  ROC-AUC: {roc_auc_score(y_test, xgb_prob):.3f}")

# ---- Save deployed model artifacts (Logistic Regression, used by app.py) ----
joblib.dump(model, 'risk_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(FEATURE_COLS, 'feature_cols.pkl')
print("\nSaved: risk_model.pkl, scaler.pkl, feature_cols.pkl")
