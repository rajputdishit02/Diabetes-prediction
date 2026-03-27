import streamlit as st
import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# -------------------------------
# Load ML Model & Dataset
# -------------------------------
model = pickle.load(open("diabetes_model.pkl", "rb"))

# Load dataset (for plotting)
columns = ["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction","Age","Outcome"]
df = pd.read_csv("data/diabetes.csv", header=None, names=columns)

# Clean zeros like in main.py
cols_to_clean = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
for col in cols_to_clean:
    median_val = df[col].median()
    df[col] = df[col].replace(0, median_val)

# -------------------------------
# Streamlit App Title
# -------------------------------
st.title("Diabetes Prediction App")
st.write("Explore dataset and predict diabetes for a patient.")

# -------------------------------
# Sidebar: Choose Tab
# -------------------------------
tab = st.sidebar.selectbox("Choose a view:", ["EDA", "Prediction"])

# -------------------------------
# Tab 1: EDA (Exploratory Data Analysis)
# -------------------------------
if tab == "EDA":
    st.header("Exploratory Data Analysis")
    
    # Show raw dataset
    if st.checkbox("Show Raw Data"):
        st.dataframe(df.head())
    
    # Glucose Distribution
    st.subheader("Glucose Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df['Glucose'], bins=30, kde=True, ax=ax)
    st.pyplot(fig)
    
    # Outcome Count
    st.subheader("Number of Patients with/without Diabetes")
    fig2, ax2 = plt.subplots()
    sns.countplot(x='Outcome', data=df, ax=ax2)
    st.pyplot(fig2)
    
    # Correlation Heatmap
    st.subheader("Feature Correlation Heatmap")
    fig3, ax3 = plt.subplots(figsize=(8,6))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', ax=ax3)
    st.pyplot(fig3)

# -------------------------------
# Tab 2: Prediction
# -------------------------------
if tab == "Prediction":
    st.header("Predict Diabetes for a Patient")
    
    # Input sliders
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
        input_data = [[Pregnancies, Glucose, BloodPressure, SkinThickness,
                       Insulin, BMI, DiabetesPedigreeFunction, Age]]
        prediction = model.predict(input_data)
        
        if prediction[0] == 1:
            st.error("The patient is likely to have Diabetes")
        else:
            st.success("The patient is unlikely to have Diabetes")