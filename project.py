import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ---------------------------------------------------------
# STEP 1: Create the Dataset inside a Pandas DataFrame
# ---------------------------------------------------------
data = {
    'Age': [45, 72, 29, 60, 81, 50, 38, 65, 74, 52],
    'Diagnosis': ['Diabetes', 'Heart Disease', 'Infection', 'Diabetes', 
                  'Heart Disease', 'Infection', 'Diabetes', 'Heart Disease', 
                  'Infection', 'Diabetes'],
    'Length_Of_Stay': [3, 8, 2, 5, 10, 4, 3, 7, 9, 6],
    'Previous_Admissions': [0, 3, 0, 1, 4, 0, 2, 3, 5, 1],
    'Readmitted': [0, 1, 0, 0, 1, 0, 0, 1, 1, 0]  # Target: 0 = No, 1 = Yes
}

df = pd.DataFrame(data)
print("--- Patient Dataset Preview ---")
print(df, "\n")

# ---------------------------------------------------------
# STEP 2: Encode Text Data into Numbers
# ---------------------------------------------------------
# Convert text diagnosis into numeric codes so models can read them
df['Diagnosis_Code'] = df['Diagnosis'].astype('category').cat.codes

# ---------------------------------------------------------
# STEP 3: Separate Features (X) and Target (y)
# ---------------------------------------------------------
X = df[['Age', 'Diagnosis_Code', 'Length_Of_Stay', 'Previous_Admissions']]
y = df['Readmitted']

# ---------------------------------------------------------
# STEP 4: Split Data into Training and Testing Sets
# ---------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ---------------------------------------------------------
# STEP 5: Train Models
# ---------------------------------------------------------
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

gb_model = GradientBoostingClassifier(random_state=42)
gb_model.fit(X_train, y_train)

# ---------------------------------------------------------
# STEP 6: Make Predictions and Evaluate
# ---------------------------------------------------------
dt_preds = dt_model.predict(X_test)
gb_preds = gb_model.predict(X_test)

dt_acc = accuracy_score(y_test, dt_preds) * 100
gb_acc = accuracy_score(y_test, gb_preds) * 100

print("--- Model Performance Results ---")
print(f"Decision Tree Accuracy:       {dt_acc:.1f}%")
print(f"Gradient Boosting Accuracy:   {gb_acc:.1f}%")

# ---------------------------------------------------------
# STEP 7: Generate Visualization Charts
# ---------------------------------------------------------
# Set up a figure with 2 subplots side-by-side
plt.figure(figsize=(12, 5))

# --- Chart 1: Model Accuracy Bar Chart ---
plt.subplot(1, 2, 1)
models = ['Decision Tree', 'Gradient Boosting']
accuracies = [dt_acc, gb_acc]
sns.barplot(x=models, y=accuracies, palette=['skyblue', 'salmon'])
plt.title('Model Accuracy Comparison')
plt.ylabel('Accuracy (%)')
plt.ylim(0, 100)

# --- Chart 2: Patient Age vs Length of Stay (Colored by Readmission) ---
plt.subplot(1, 2, 2)
sns.scatterplot(
    data=df, 
    x='Age', 
    y='Length_Of_Stay', 
    hue='Readmitted', 
    style='Readmitted', 
    s=120, 
    palette='Set1'
)
plt.title('Age vs. Length of Stay (Readmission Status)')
plt.xlabel('Patient Age')
plt.ylabel('Length of Stay (Days)')

# Adjust layout so labels don't overlap and show the plots
plt.tight_layout()
plt.show()