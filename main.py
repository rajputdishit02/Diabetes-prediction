# Importing the libraries:
import pandas as pd

# Loading the dataset
columns = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"
]
dataframe = pd.read_csv("data/diabetes.csv", names=columns)

# Checking whether the dataset is loaded properly
print("The First 5 rows of the dataset are:")
print(dataframe.head(5))

print("\n The Number of rows and columns within the Dataset are:" )
print(dataframe.shape)

print("\n The column names of the df are:")
print(dataframe.columns)

print(dataframe.info())
print(dataframe.describe())

# The dataframe indicates that the min value of the BP, BMI. Insulin, etc is 0. which is medically not possible.
# To remove the O or incorrect values, we are gonna replace the Zero values with the median values

columns_to_clean = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
for col in columns_to_clean:
    median_val = dataframe[col].median()
    dataframe[col] = dataframe[col].replace(0, median_val)

print("The Dataset has been cleaned by replacing the technical not possible values to Median values")
print(dataframe.describe())

import matplotlib.pyplot as plt
import seaborn as sns

# Histogram of BMI for visualising the distribution Features
plt.figure(figsize=(6,4))
sns.histplot(dataframe['BMI'], bins=30, kde=True)
plt.title('BMI Levels')
plt.xlabel('BMI')
plt.ylabel('Count')
plt.show()


# Getting the correlation of the features having the impact in the Diabetes of the patients
plt.figure(figsize=(8,6))
sns.heatmap(dataframe.corr(), annot=True, cmap='coolwarm')
plt.title('Feature Correlation Heatmap')
plt.show()

# Outcome count:
sns.countplot(x='Outcome', data=dataframe)
plt.title('Number of Patients with/without Diabetes')
plt.show()

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Features (all columns except Outcome)
X = dataframe.drop("Outcome", axis=1)

# Target (Outcome)
y = dataframe["Outcome"]

# Split data: 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize model
model = RandomForestClassifier(random_state=42)

# Train model
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)


# Accuracy
acc = accuracy_score(y_test, y_pred)
print("Accuracy:", acc)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# Classification Report
cr = classification_report(y_test, y_pred)
print("Classification Report:\n", cr)

# Importance of each feature
importances = pd.Series(model.feature_importances_, index=X.columns)
importances.sort_values().plot(kind='barh')
plt.title("Feature Importance")
plt.show()

# saving the model
import pickle

# Save model
with open("diabetes_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved successfully!")

import streamlit as st

# Load saved model
model = pickle.load(open("diabetes_model.pkl", "rb"))

st.title("Diabetes Prediction App")
st.write("Enter patient details below:")

# Input sliders for features
Pregnancies = st.slider("Pregnancies", 0, 20, 0)
Glucose = st.slider("Glucose", 0, 200, 120)
BloodPressure = st.slider("BloodPressure", 0, 140, 70)
SkinThickness = st.slider("SkinThickness", 0, 100, 20)
Insulin = st.slider("Insulin", 0, 900, 80)
BMI = st.slider("BMI", 0.0, 70.0, 25.0)
DiabetesPedigreeFunction = st.slider("Diabetes Pedigree Function", 0.0, 2.5, 0.5)
Age = st.slider("Age", 0, 100, 30)

# Predict button
if st.button("Predict"):
    # Prepare data for prediction
    input_data = [[Pregnancies, Glucose, BloodPressure, SkinThickness,
                   Insulin, BMI, DiabetesPedigreeFunction, Age]]
    
    prediction = model.predict(input_data)
    
    if prediction[0] == 1:
        st.error("The patient is likely to have Diabetes")
    else:
        st.success("The patient is unlikely to have Diabetes")