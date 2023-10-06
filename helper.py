import numpy as np

def prediction(standardization, best_estimator, Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age):
    # Create a list containing the input values
    input_values = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]
    print(input_values,'<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    # Convert the list to a NumPy array and reshape it
    input_array = np.asarray(input_values).reshape(1, -1)
    print(input_array,'<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>')
    # Standardize the input data using the provided StandardScaler
    standardized_input = standardization.transform(input_array)
    
    # Make a prediction using the best_estimator
    prediction = best_estimator.predict(standardized_input)

    return prediction
