import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Page Title and Layout
st.set_page_config(page_title="Hospital Readmission Predictor", layout="centered")
st.title("🏥 Hospital Patient Readmission Prediction App")
st.markdown("Use machine learning to estimate a patient's risk of returning to the hospital.")

# 1. Dataset Setup
data = {
    'Age': [45, 72, 29, 60, 81, 50, 38, 65, 74, 52],
    'Diagnosis': ['Diabetes', 'Heart Disease', 'Infection', 'Diabetes', 
                  'Heart Disease', 'Infection', 'Diabetes', 'Heart Disease', 
                  'Infection', 'Diabetes'],
    'Length_Of_Stay': [3, 8, 2, 5, 10, 4, 3, 7, 9, 6],
    'Previous_Admissions': [0, 3, 0, 1, 4, 0, 2, 3, 5, 1],
    'Readmitted': [0, 1, 0, 0, 1, 0, 0, 1, 1, 0]
}
df = pd.DataFrame(data)
df['Diagnosis_Code'] = df['Diagnosis'].astype('category').cat.codes

# 2. Train Models behind the scenes
X = df[['Age', 'Diagnosis_Code', 'Length_Of_Stay', 'Previous_Admissions']]
y = df['Readmitted']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

dt_model = DecisionTreeClassifier(random_state=42).fit(X_train, y_train)
gb_model = GradientBoostingClassifier(random_state=42).fit(X_train, y_train)

dt_acc = accuracy_score(y_test, dt_model.predict(X_test)) * 100
gb_acc = accuracy_score(y_test, gb_model.predict(X_test)) * 100

# Sidebar for User Inputs
st.sidebar.header("Patient Input Features")
age = st.sidebar.slider("Patient Age", 18, 100, 50)
diagnosis = st.sidebar.selectbox("Diagnosis", ['Diabetes', 'Heart Disease', 'Infection'])
length_of_stay = st.sidebar.slider("Length of Stay (Days)", 1, 30, 5)
prev_admissions = st.sidebar.slider("Previous Admissions", 0, 10, 1)

# Convert diagnosis text to code
diag_mapping = {'Diabetes': 0, 'Heart Disease': 1, 'Infection': 2}
diag_code = diag_mapping[diagnosis]

# Prediction Button
if st.sidebar.button("Predict Readmission Risk"):
    user_input = [[age, diag_code, length_of_stay, prev_admissions]]
    prediction = gb_model.predict(user_input)
    
    st.subheader("Prediction Results")
    if prediction[0] == 1:
        st.error("⚠️ **High Risk of Readmission:** This patient has a high likelihood of returning to the hospital. Extra follow-up care is recommended.")
    else:
        st.success("✅ **Low Risk:** This patient is stable and has a low likelihood of readmission.")

# Display Visualizations
st.markdown("---")
st.subheader("📊 Model Performance & Dataset Trends")

col1, col2 = st.columns(2)
with col1:
    st.metric(label="Decision Tree Accuracy", value=f"{dt_acc:.1f}%")
with col2:
    st.metric(label="Gradient Boosting Accuracy", value=f"{gb_acc:.1f}%")

# Plot Charts
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sns.barplot(x=['Decision Tree', 'Gradient Boosting'], y=[dt_acc, gb_acc], palette=['skyblue', 'salmon'], ax=axes[0])
axes[0].set_title('Model Accuracy Comparison')
axes[0].set_ylabel('Accuracy (%)')
axes[0].set_ylim(0, 100)

sns.scatterplot(data=df, x='Age', y='Length_Of_Stay', hue='Readmitted', style='Readmitted', s=120, palette='Set1', ax=axes[1])
axes[1].set_title('Age vs. Length of Stay')

plt.tight_layout()
st.pyplot(fig)