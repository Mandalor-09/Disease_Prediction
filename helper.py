import joblib
import pandas as pd
#import numpy as np
import pickle

model_filename = 'model/diabaties_model_og.joblib'
model = joblib.load(model_filename)

# Load the models
svm_model = joblib.load('model/svm_model.joblib')
dt_model = joblib.load('model/decision_tree_model.joblib')
rf_model = joblib.load('model/random_forest_model.joblib')
lr_model = joblib.load('model/logistic_regression_model.joblib')

def prediction(Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age):
    # Convert input values to numeric
    a = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]
    a = [float(val) for val in a]  # Convert to float
    
    # Make prediction
    prediction = model.predict([a])[0] 

    if int(prediction) == 1:
        print('Patient is Diabetic')
        return 'Patient is Diabetic'
    else:
        print('Patient is non Diabetic')
        return 'Patient is Non Diabetic'

def predict_disease_lr(input_data,symptoms):
    data = {}
    for i in input_data:
        for j in symptoms:
            if j != i:
                data[j] = 0
            else:
                data[j] = 1

    data = pd.DataFrame([data])

    # Predict using the trained Logistic Regression model
    prediction = lr_model.predict(data)[0]
    prob_scores = lr_model.predict_proba(data)[0]

    unique_labels = lr_model.classes_
    prob_dict = {label: prob for label, prob in zip(unique_labels, prob_scores)}

    return prediction, prob_dict

def predict_disease_rf(input_data,symptoms):
    data = {}
    for i in input_data:
        for j in symptoms:
            if j != i:
                data[j] = 0
            else:
                data[j] = 1

    data = pd.DataFrame([data])

    # Predict using the trained Random Forest model
    prediction = rf_model.predict(data)[0]
    prob_scores = rf_model.predict_proba(data)[0]

    unique_labels = rf_model.classes_
    prob_dict = {label: prob for label, prob in zip(unique_labels, prob_scores)}

def predict_disease_dt(input_data,symptoms):
    data = {}
    for i in input_data:
        for j in symptoms:
            if j != i:
                data[j] = 0
            else:
                data[j] = 1

    data = pd.DataFrame([data])

    # Predict using the trained Decision Tree model
    prediction = dt_model.predict(data)[0]

    # Decision Trees don't have built-in predict_proba method
    prob_dict = {label: 1.0 if label == prediction else 0.0 for label in dt_model.classes_}

    return prediction, prob_dict

def predict_disease_svm(input_data,symptoms):
    data = {}
    for i in input_data:
        for j in symptoms:
            if j != i:
                data[j] = 0
            else:
                data[j] = 1

    data = pd.DataFrame([data])

    # Predict using the trained SVM model
    prediction = svm_model.predict(data)[0]
    prob_scores = svm_model.predict_proba(data)[0]

    unique_labels = svm_model.classes_
    prob_dict = {label: prob for label, prob in zip(unique_labels, prob_scores)}

    return prediction, prob_dict

