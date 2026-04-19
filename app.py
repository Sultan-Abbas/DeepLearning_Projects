import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

# Load the trained model
model = tf.keras.models.load_model('model.h5')

# Load the encoders and scaler
with open('label_encoder.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('one_hot_encoder.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Define the exact column order expected by the model during training (from X.columns)
X_COLUMNS = [
    'CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard',
    'IsActiveMember', 'EstimatedSalary', 'Gender_encoded', 'Geography_France',
    'Geography_Germany', 'Geography_Spain'
]

# Define the predict_churn function that takes raw inputs
def predict_churn(raw_input_data):
   
    input_df = pd.DataFrame([raw_input_data])

    # Apply Label Encoding for Gender using the loaded label_encoder_gender
    input_df['Gender_encoded'] = label_encoder_gender.transform(input_df['Gender'])

    # Apply One-Hot Encoding for Geography using the loaded onehot_encoder_geo
    # The onehot_encoder_geo was fitted on df[['Gender', 'Geography']] in the notebook.
    # So, transform needs both 'Gender' and 'Geography' columns (raw strings).
    encoded_features = onehot_encoder_geo.transform(input_df[['Gender', 'Geography']])
    all_encoded_cols = onehot_encoder_geo.get_feature_names_out(['Gender', 'Geography'])
    encoded_df_temp = pd.DataFrame(encoded_features, columns=all_encoded_cols)

    # Drop the one-hot encoded Gender columns to match the training data's structure.
    # In the notebook, these were dropped, and Gender was handled by label_encoder_gender.
    encoded_geography_df = encoded_df_temp.drop(columns=['Gender_Female', 'Gender_Male'])

    # Drop original 'Gender' and 'Geography' columns from the input DataFrame
    input_df = input_df.drop(columns=['Gender', 'Geography'])

    # Concatenate all processed features
    input_processed = pd.concat([input_df, encoded_geography_df], axis=1)

    # Ensure column order matches X_COLUMNS (defined globally)
    input_processed = input_processed[X_COLUMNS]

    # Scale the input data using the pre-trained scaler
    input_scaled = scaler.transform(input_processed)

    # Make prediction
    prediction_prob = model.predict(input_scaled)[0][0]
    prediction_class = (prediction_prob > 0.5).astype(int)

    return prediction_prob, prediction_class


## Streamlit app UI
st.title('Customer Churn Prediction') # Corrected typo here

# User input widgets
geography = st.selectbox('Geography', onehot_encoder_geo.categories_[1]) # Corrected to use Geography categories
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

# Prepare the input data as a dictionary of raw values from the Streamlit widgets
raw_user_input_data = {
    'CreditScore': credit_score,
    'Geography': geography,
    'Gender': gender,
    'Age': age,
    'Tenure': tenure,
    'Balance': balance,
    'NumOfProducts': num_of_products,
    'HasCrCard': has_cr_card,
    'IsActiveMember': is_active_member,
    'EstimatedSalary': estimated_salary
}

# Get prediction for the new customer using the predict_churn function
churn_probability, churn_class = predict_churn(raw_user_input_data)

st.write(f'Churn Probability: {churn_probability:.2f}')

if churn_probability > 0.5: # Corrected typo: churn_probablity -> churn_probability
    st.write('The customer is likely to churn.')
else:
    st.write('The customer is not likely to churn.')