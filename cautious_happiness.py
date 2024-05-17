import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
import streamlit as st
import logging
import traceback

# Configure logging
logging.basicConfig(filename='error.log', level=logging.ERROR)

try:
    # Load the weather data
    weather_data = pd.read_csv('/content/weather_data.csv')

    # Data Preprocessing
    weather_data.fillna(method='ffill', inplace=True)
    scaler = StandardScaler()
    columns_to_normalize = ['T2M', 'PRECTOTCORR', 'RH2M', 'ALLSKY_SFC_SW_DWN', 'WS2M']
    weather_data[columns_to_normalize] = scaler.fit_transform(weather_data[columns_to_normalize])

    # Feature Engineering
    weather_data['T2M_lag_1'] = weather_data['T2M'].shift(1)
    weather_data['PRECTOTCORR_lag_1'] = weather_data['PRECTOTCORR'].shift(1)
    weather_data.dropna(inplace=True)

    # Define features and target variable for disease prediction
    X = weather_data[['T2M', 'PRECTOTCORR', 'RH2M', 'ALLSKY_SFC_SW_DWN', 'WS2M', 'T2M_lag_1', 'PRECTOTCORR_lag_1']]
    y_disease = weather_data['disease_risk']
    X_train, X_test, y_train, y_test = train_test_split(X, y_disease, test_size=0.2, random_state=42)

    # Train the Disease Prediction Model
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    y_pred_disease = clf.predict(X_test)
    disease_accuracy = accuracy_score(y_test, y_pred_disease)

    # Define target variable for yield prediction
    y_yield = weather_data['GWETROOT']
    X_train, X_test, y_train, y_test = train_test_split(X, y_yield, test_size=0.2, random_state=42)

    # Train the Yield Prediction Model
    reg = RandomForestRegressor()
    reg.fit(X_train, y_train)
    y_pred_yield = reg.predict(X_test)
    yield_rmse = mean_squared_error(y_test, y_pred_yield, squared=False)

    # Streamlit App Deployment
    st.title('Crop Disease Risk and Yield Prediction')

    # Input fields for current weather conditions
    T2M = st.number_input('Temperature')
    PRECTOTCORR = st.number_input('Precipitation')
    RH2M = st.number_input('Humidity')
    ALLSKY_SFC_SW_DWN = st.number_input('Solar Radiation')
    WS2M = st.number_input('Wind Speed')
    T2M_lag_1 = st.number_input('Temperature Lag 1')
    PRECTOTCORR_lag_1 = st.number_input('Precipitation Lag 1')

    # Create a dataframe for prediction
    input_data = pd.DataFrame([[T2M, PRECTOTCORR, RH2M, ALLSKY_SFC_SW_DWN, WS2M, T2M_lag_1, PRECTOTCORR_lag_1]],
                            columns=['T2M', 'PRECTOTCORR', 'RH2M', 'ALLSKY_SFC_SW_DWN', 'WS2M', 'T2M_lag_1', 'PRECTOTCORR_lag_1'])

    # Scale only the non-lagged features
    input_data[columns_to_normalize] = scaler.transform(input_data[columns_to_normalize])

    # Predict disease risk
    disease_risk = clf.predict(input_data)
    st.write(f'Disease Risk: {"High" if disease_risk[0] else "Low"}')

    # Predict yield
    yield_prediction = reg.predict(input_data)
    st.write(f'Predicted Yield: {yield_prediction[0]:.2f}')

    # Display evaluation metrics
    st.write(f'Disease Prediction Accuracy: {disease_accuracy:.2f}')
    st.write(f'Yield Prediction RMSE: {yield_rmse:.2f}')

except Exception as e:
    print(f"An error occurred: {e}")
    print(traceback.format_exc())  # Print detailed traceback
    logging.error(f"An error occurred: {e}")
    logging.exception("Exception occurred")  # Log exception traceback
