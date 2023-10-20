import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from helper import *
from sklearn.preprocessing import StandardScaler
import pandas as pd

#dataset = pd.read_csv('C:\\Users\\DELL\\OneDrive\\Desktop\\Diabetic\\dataset\\dataset.csv')

train = pd.read_csv(r'C:\Users\DELL\OneDrive\Desktop\Diabetic\dataset\Training.csv')
test = pd.read_csv(r'C:\Users\DELL\OneDrive\Desktop\Diabetic\dataset\Testing.csv')
desc = pd.read_csv(r'C:\Users\DELL\OneDrive\Desktop\Diabetic\dataset\disease_description.csv')
pre = pd.read_csv(r'C:\Users\DELL\OneDrive\Desktop\Diabetic\dataset\disease_precaution.csv')
sym = pd.read_csv(r'C:\Users\DELL\OneDrive\Desktop\Diabetic\dataset\symptom_severity.csv')
    
symptoms = [None,'abdominal_pain', 'abnormal_menstruation', 'acidity', 'acute_liver_failure', 'altered_sensorium', 'anxiety', 'back_pain', 'belly_pain', 'blackheads', 'bladder_discomfort', 'blister', 'blood_in_sputum', 'bloody_stool', 'blurred_and_distorted_vision', 'breathlessness', 'brittle_nails', 'bruising', 'burning_micturition', 'chest_pain', 'chills', 'cold_hands_and_feet', 'coma', 'congestion', 'constipation', 'continuous_feel_of_urine', 'continuous_sneezing', 'cough', 'cramps', 'dark_urine', 'dehydration', 'depression', 'diarrhoea', 'dyschromic_patches', 'distention_of_abdomen', 'dizziness', 'drying_and_tingling_lips', 'enlarged_thyroid', 'excessive_hunger', 'extra_marital_contacts', 'family_history', 'fast_heart_rate', 'fatigue', 'fluid_overload', 'fluid_overload.1', 'foul_smell_of urine', 'headache', 'high_fever', 'hip_joint_pain', 'history_of_alcohol_consumption', 'increased_appetite', 'indigestion', 'inflammatory_nails', 'internal_itching', 'irregular_sugar_level', 'irritability', 'irritation_in_anus', 'itching', 'joint_pain', 'knee_pain', 'lack_of_concentration', 'lethargy', 'loss_of_appetite', 'loss_of_balance', 'loss_of_smell', 'loss_of_taste', 'malaise', 'mild_fever', 'mood_swings', 'movement_stiffness', 'mucoid_sputum', 'muscle_pain', 'muscle_wasting', 'muscle_weakness', 'nausea', 'neck_pain', 'nodal_skin_eruptions', 'obesity', 'pain_behind_the_eyes', 'pain_during_bowel_movements', 'pain_in_anal_region', 'painful_walking', 'palpitations', 'passage_of_gases', 'patches_in_throat', 'phlegm', 'polyuria', 'prominent_veins_on_calf', 'puffy_face_and_eyes', 'pus_filled_pimples', 'receiving_blood_transfusion', 'receiving_unsterile_injections', 'red_sore_around_nose', 'red_spots_over_body', 'redness_of_eyes', 'restlessness', 'runny_nose', 'rusty_sputum', 'scurrying', 'shivering', 'silver_like_dusting', 'sinus_pressure', 'skin_peeling', 'skin_rash', 'slurred_speech', 'small_dents_in_nails', 'spinning_movements', 'spotting_urination', 'stiff_neck', 'stomach_bleeding', 'stomach_pain', 'sunken_eyes', 'sweating', 'swelled_lymph_nodes', 'swelling_joints', 'swelling_of_stomach', 'swollen_blood_vessels', 'swollen_extremities', 'swollen_legs', 'throat_irritation', 'tiredness', 'toxic_look_(typhus)', 'ulcers_on_tongue', 'unsteadiness', 'visual_disturbances', 'vomiting', 'watering_from_eyes', 'weakness_in_limbs', 'weakness_of_one_body_side', 'weight_gain', 'weight_loss', 'yellow_crust_ooze', 'yellow_urine', 'yellowing_of_eyes', 'yellowish_skin']

disease = ['AIDS', 'Acne', 'Alcoholic hepatitis', 'Allergy', 'Arthritis', 'Bronchial Asthma', 'Cervical spondylosis', 'Chicken pox', 'Chronic cholestasis', 'Common Cold', 'Dengue', 'Diabetes ', 'Dimorphic hemorrhoids(piles)', 'Drug Reaction', 'Fungal infection', 'GERD', 'Gastroenteritis', 'Heart attack', 'Hepatitis A', 'Hepatitis B', 'Hepatitis C', 'Hepatitis D', 'Hepatitis E', 'Hypertension ', 'Hyperthyroidism', 'Hypoglycemia', 'Hypothyroidism', 'Impetigo', 'Jaundice', 'Malaria', 'Migraine', 'Osteoarthritis', 'Paralysis (brain hemorrhage)', 'Paroxysmal Positional Vertigo', 'Peptic ulcer disease', 'Pneumonia', 'Psoriasis', 'Tuberculosis', 'Typhoid', 'Urinary tract infection', 'Varicose veins']




st.sidebar.title('Diabatics ANALYSIS')
sidebar_data = st.sidebar.radio(
    'Select an Option',
    ('Diabatis Prediction','Disease Prediction','Non Diabatis Analysis')
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
            
            st.write(predicted)
    
        else:
            st.warning("Please fill in all input values before submitting.")



elif sidebar_data == 'Disease Prediction':
    st.subheader('Disease Prediction')

    # Define columns for layout
    col1, col2 = st.columns(2)

    # Create select boxes for symptoms
    with col1:
        Symptoms_1 = st.selectbox("Symptom 1:", symptoms, key='symptom_1')
        Symptoms_3 = st.selectbox("Symptom 3:", symptoms, key='symptom_3')
        Symptoms_5 = st.selectbox("Symptom 5:", symptoms, key='symptom_5')

    with col2:
        Symptoms_2 = st.selectbox("Symptom 2:", symptoms, key='symptom_2')
        Symptoms_4 = st.selectbox("Symptom 4:", symptoms, key='symptom_4')
        Symptoms_6 = st.selectbox("Symptom 6:", symptoms, key='symptom_6')

    # Button to check disease
    if st.button("Check Disease", key='prediction_button', type="primary"):
        # You can add code to predict the disease here
        data = [Symptoms_1,Symptoms_2,Symptoms_3,Symptoms_4,Symptoms_5,Symptoms_6]
        input_symptoms = [i for i in data if i is not None]
        predicted_disease_svm, prob_svm = predict_disease_svm(input_symptoms,symptoms)
        predicted_disease_dt, prob_dt = predict_disease_dt(input_symptoms,symptoms)
        predicted_disease_rf, prob_rf = predict_disease_rf(input_symptoms,symptoms)
        predicted_disease_lr, prob_lr = predict_disease_lr(input_symptoms,symptoms)

        combined_probs = {label: 0 for label in prob_svm.keys()}

        for label in combined_probs.keys():
            combined_probs[label] += prob_svm[label]
            combined_probs[label] += prob_dt[label]
            combined_probs[label] += prob_rf[label]
            combined_probs[label] += prob_lr[label]

        # Sort the diseases by combined probabilities
        sorted_diseases = sorted(combined_probs.items(), key=lambda x: x[1], reverse=True)

        # Get the top 5 diseases
        top_5_diseases = sorted_diseases[:5]
        

        # Print the top 5 diseases and their probabilities
        print("Top 5 Diseases:")
        for disease, probability in top_5_diseases:
            print(f"Disease: {disease}, Probability: {probability}")

    else:
        st.warning("Please fill in all input values before submitting.")