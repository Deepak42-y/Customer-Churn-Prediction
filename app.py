import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load your trained model
model = joblib.load('model.pkl')

def create_features(df):
    # Calculate the derived features
    df['Days_1call_duration'] = df['Total day minutes'] / df['Total day calls']
    df['intern_1call_duration'] = df['Total intl minutes'] / df['Total intl calls']
    df['evening_1call_duration'] = df['Total eve minutes'] / df['Total eve calls']
    df['night_1call_duration'] = df['Total night minutes'] / df['Total night calls']
    
    # Handle infinite values
    df.replace([np.inf, -np.inf], 0, inplace=True)
    
    return df

def predict_churn(features):
    """
    Function to predict churn based on input features
    :param features: A dataframe containing user inputs
    :return: Churn prediction (0 or 1)
    """
    return model.predict(features)

def main():
    st.title("Telecom Churn Prediction")
    st.write("Enter the customer details to predict churn:")

    # Create input fields for the features used in your model
    account_length = st.number_input("Account Length", min_value=0)
    area_code = st.selectbox("Area Code",[510, 415,408] )
    intl_plan = st.selectbox("International Plan", ['Yes', 'No'])
    voice_mail_plan = st.selectbox("Voice Mail Plan", ['Yes', 'No'])
    total_day_minutes = st.number_input("Total Day Minutes", min_value=0.0)
    total_day_calls = st.number_input("Total Day Calls", min_value=0)
    total_day_charge = st.number_input("Total Day Charge", min_value=0.0)
    total_eve_minutes = st.number_input("Total Eve Minutes", min_value=0.0)
    total_eve_calls = st.number_input("Total Eve Calls", min_value=0)
    total_eve_charge = st.number_input("Total Eve Charge", min_value=0.0)
    total_night_minutes = st.number_input("Total Night Minutes", min_value=0.0)
    total_night_calls = st.number_input("Total Night Calls", min_value=0)
    total_night_charge = st.number_input("Total Night Charge", min_value=0.0)
    total_intl_minutes = st.number_input("Total Intl Minutes", min_value=0.0)
    total_intl_calls = st.number_input("Total Intl Calls", min_value=0)
    total_intl_charge = st.number_input("Total Intl Charge", min_value=0.0)
    customer_service_calls = st.number_input("Customer Service Calls", min_value=0)

    
    # Convert the inputs to a dataframe
    data = {
        'Account length': [account_length],
        'Area code': [area_code],
        'International plan': [1 if intl_plan == 'Yes' else 0],
        'Voice mail plan': [1 if voice_mail_plan == 'Yes' else 0],
        'Total day minutes': [total_day_minutes],
        'Total day calls': [total_day_calls],
        'Total day charge': [total_day_charge],
        'Total eve minutes': [total_eve_minutes],
        'Total eve calls': [total_eve_calls],
        'Total eve charge': [total_eve_charge],
        'Total night minutes': [total_night_minutes],
        'Total night calls': [total_night_calls],
        'Total night charge': [total_night_charge],
        'Total intl minutes': [total_intl_minutes],
        'Total intl calls': [total_intl_calls],
        'Total intl charge': [total_intl_charge],
        'Customer service calls': [customer_service_calls]
    }
    
    df = pd.DataFrame(data)
    
    # Create derived features
    df = create_features(df)
    
    # Select only the relevant features for prediction
    features = df[['Account length', 'Area code', 'International plan', 'Voice mail plan',
                   'Customer service calls', 'Days_1call_duration',
                   'intern_1call_duration', 'evening_1call_duration', 'night_1call_duration']]
    
    # Predict and display the result
    if st.button("Predict"):
        prediction = predict_churn(features)
        if prediction == 1:
            st.error("YES , The customer is likely to churn.")
        else:
            st.success("NO , The customer is unlikely to churn.")

if __name__ == "__main__":
    main()
