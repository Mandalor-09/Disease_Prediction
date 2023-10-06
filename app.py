import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import helper
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import pickle
from sklearn.preprocessing import StandardScaler
import pandas as pd

dataset = pd.read_csv('C:\\Users\\DELL\\OneDrive\\Desktop\\Diabetic\\dataset\\dataset.csv')

#standardization = pickle.load(open('model/standard_scaler.pkl', 'rb'))
model = pickle.load(open('model/model.pkl', 'rb'))

X = dataset.iloc[:,1:]

Y= dataset['Outcome']
print(X.columns)
standardization = StandardScaler()
standard_data = standardization.fit_transform(X)
print("Number of features expected by StandardScaler:", standardization.n_features_in_)

def prediction(Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age):
    input_values = [0,Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]
    arr =  np.asarray(input_values)
    reshape_arr = arr.reshape(1, -1)
    s_data = standardization.transform(reshape_arr)
    s_data = s_data[:,1:]
    prediction = model.predict(s_data)
    return prediction
    

st.sidebar.title('Diabatics ANALYSIS')
sidebar_data = st.sidebar.radio(
    'Select an Option',
    ('Diabatis Prediction','Diabatis Analysis','Non Diabatis Analysis')
)

if sidebar_data == 'Diabatis Prediction':
    st.subheader('Diabatis Prediction')
    col1 ,col2 =st.columns(2)
    with col1:
        Pregnancies = st.text_input("Pregnancies count:")
    with col2:
        Glucose = st.text_input("Glucose count:")
    with col1:
        BloodPressure = st.text_input("BloodPressure count:")
    with col2:
        SkinThickness = st.text_input("SkinThickness count:")
    with col1:
        Insulin = st.text_input("Insulin count:")
    with col2:
        BMI = st.text_input("BMI count:")
    with col1:
        DiabetesPedigreeFunction = st.text_input("DiabetesPedigreeFunction count:")
    with col2:
        Age = st.text_input("Age count:")

    if st.button("Check Prediction",key='prediction_button', type="primary"):
        if all([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]):
            #st.write(Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age)
            predicted = prediction(Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age)
            
            st.header(st.write(predicted))
    
        else:
            st.warning("Please fill in all input values before submitting.")