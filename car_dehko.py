import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load the model (which is a GridSearchCV object)
with open("ml_model.pkl", 'rb') as f:
    grid_search = pickle.load(f)

# Extract the best model from GridSearchCV
model = grid_search.best_estimator_

# Feature names (Remove 'Model' from the features)
features = ['Fuel Type', 'Body Type', 'Transmission Type', 'Kms Driven', 'No. of Owners', 'OEM', 
            'Model Year', 'Seats', 'Engine Displacement', 'Mileage', 'City', 'Car Age']

# Categorical variable mappings
categorical_mappings = {
    'Fuel Type': {'Petrol': 0, 'Diesel': 1, 'LPG': 4, 'CNG': 0, 'Electric': 3},
    'Body Type': {
        'Convertibles': 0, 'Coupe': 1, 'Hybrids': 2, 'Hatchback': 3, 'Minivans': 4,
        'MUV': 5, 'Pickup Trucks': 6, 'Sedan': 7, 'SUV': 8, 'Wagon': 9
    },
    'Transmission Type': {'Automatic': 1, 'Manual': 0},
    'OEM': {
        'Audi': 0, 'BMW': 1, 'Chevrolet': 2, 'Citroen': 3, 'Datsun': 4, 'Fiat': 5,
        'Ford': 6, 'Hindustan Motors': 7, 'Honda': 8, 'Hyundai': 9, 'Isuzu': 10,
        'Jaguar': 11, 'Jeep': 12, 'Kia': 13, 'Land Rover': 14, 'Lexus': 15, 'MG': 16,
        'Mahindra': 17, 'Mahindra Renault': 18, 'Mahindra Ssangyong': 19, 'Maruti': 20,
        'Mercedes-Benz': 21, 'Mini': 22, 'Mitsubishi': 23, 'Nissan': 24, 'Opel': 25,
        'Porsche': 26, 'Renault': 27, 'Skoda': 28, 'Tata': 29, 'Toyota': 30,
        'Volkswagen': 31, 'Volvo': 32
    },
    'City': {'Bangalore': 1, 'Delhi': 2, 'Hyderabad': 3, 'Jaipur': 4, 'Chennai': 0}
}

st.title("Car Dheko - Used Car Price Prediction App")

# Creating three columns and placing the title in the center column
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    st.image(r"C:\Users\sandh\OneDrive\Desktop\car_gif.gif")

# Split input fields into two columns for organization
col1, col2 = st.columns(2)

# Column 1 inputs
with col1:
    # Horizontal layout for radio buttons
    fuel_type = st.radio("Fuel Type", list(categorical_mappings['Fuel Type'].keys()), index=0, horizontal=True)
    transmission_type = st.radio("Transmission Type", list(categorical_mappings['Transmission Type'].keys()), index=0, horizontal=True)
    
    oem = st.selectbox("OEM", list(categorical_mappings['OEM'].keys()))
    
    # Range guideline for Kms Driven
    kms_driven = st.number_input("Kms Driven (e.g.,between 0 and 200000)")

    mileage = st.number_input("Mileage (e.g., 10.0 to 50.0 kmpl)", min_value=0.0)

    engine_displacement = st.number_input("Engine Displacement (in CC, e.g., 500 to 5000)", min_value=0)

# Column 2 inputs
with col2:
    # Combined label and example text for "No. of Owners"
    no_of_owners = st.number_input("No. of Owners (e.g., 1 to 5)", min_value=1, max_value=5)


    # Slider for Model Year (between 1900 and 2024)
    model_year = st.slider("Model Year (min: 1900, max: 2023)", min_value=1900, max_value=2023, value=2020)

    seats = st.number_input("Seats (e.g., 2 to 8)", min_value=1)
    
   

    city = st.selectbox("City", list(categorical_mappings['City'].keys()))

    # Move "Car Age" to this column
    st.write("Enter Car Age (years, e.g., 0 to 20):")
    car_age = st.number_input("Car Age (e.g., 0 to 20", min_value=0)

# Convert categorical selections to their encoded values
fuel_type_encoded = categorical_mappings['Fuel Type'][fuel_type]
transmission_type_encoded = categorical_mappings['Transmission Type'][transmission_type]
oem_encoded = categorical_mappings['OEM'][oem]
city_encoded = categorical_mappings['City'][city]

# Prepare input data
input_data = pd.DataFrame({
    'Fuel Type': [fuel_type_encoded],
    'Body Type': [0],  # Default or user can add more logic for Body Type
    'Transmission Type': [transmission_type_encoded],
    'Kms Driven': [kms_driven],
    'No. of Owners': [no_of_owners],
    'OEM': [oem_encoded],
    'Model Year': [model_year],
    'Seats': [seats],
    'Engine Displacement': [engine_displacement],
    'Mileage': [mileage],
    'City': [city_encoded],
    'Car Age': [car_age]
})

# Align input data with the model's expected features
input_data = pd.get_dummies(input_data)
missing_cols = set(model.feature_names_in_) - set(input_data.columns)
for c in missing_cols:
    input_data[c] = 0
input_data = input_data[model.feature_names_in_]

# Predict button
if st.button("Predict Price"):
    predicted_price = model.predict(input_data)
    st.write(f"Predicted Price: {max(0, predicted_price[0]):,.2f}")
