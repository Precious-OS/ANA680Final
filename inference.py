import json
import os
import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from sklearn.preprocessing import LabelEncoder, StandardScaler
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load the trained model, encoders, and scaler
model = joblib.load("NigeriaHeartAttack.pkl")
label_encoders = joblib.load("label_encoders.pkl")
scaler = joblib.load("scaler.pkl")

# Define categorical columns
categorical_columns = ['State', 'Age_Group', 'Gender', 'Smoking_Status', 'Alcohol_Consumption', 
                      'Exercise_Frequency', 'Hypertension', 'Diabetes', 'Cholesterol_Level', 
                      'Family_History', 'Stress_Level', 'Diet_Type', 'Heart_Attack_Severity', 
                      'Hospitalized', 'Income_Level', 'Urban_Rural', 'Employment_Status']

# Inference endpoint
@app.route('/invocations', methods=['POST'])
def predict():
    try:
        # Get JSON data from the request
        form_data = request.get_json()
        if not form_data:
            logger.error("No JSON data received in POST request")
            return jsonify({'error': 'No data received in POST request'}), 400

        logger.debug(f"Received form data: {form_data}")

        # Convert form data to DataFrame
        input_df = pd.DataFrame([form_data])

        # Ensure all required columns are present and not empty or invalid
        required_columns = ['State', 'Age_Group', 'Gender', 'BMI', 'Smoking_Status', 'Alcohol_Consumption',
                           'Exercise_Frequency', 'Hypertension', 'Diabetes', 'Cholesterol_Level',
                           'Family_History', 'Stress_Level', 'Diet_Type', 'Heart_Attack_Severity',
                           'Hospitalized', 'Income_Level', 'Urban_Rural', 'Employment_Status']
        missing_or_empty_columns = [col for col in required_columns if not input_df[col].iloc[0] or str(input_df[col].iloc[0]).strip() == '']
        if missing_or_empty_columns:
            logger.error(f"Missing or empty required columns: {missing_or_empty_columns}")
            return jsonify({'error': f'Missing or empty required columns: {", ".join(missing_or_empty_columns)}'}), 400

        # Check for invalid BMI (non-numeric or NaN)
        if 'BMI' in input_df.columns and (pd.isna(input_df['BMI'].iloc[0]) or not str(input_df['BMI'].iloc[0]).replace('.', '').isdigit()):
            logger.error("Invalid BMI value: must be a number between 15 and 40")
            return jsonify({'error': 'Invalid BMI value: must be a number between 15 and 40'}), 400

        # Preprocess input data (encode categorical, scale numerical)
        for column in categorical_columns:
            if column in input_df.columns:
                if pd.isna(input_df[column].iloc[0]) or input_df[column].iloc[0] == 'None':
                    input_df[column] = 'No_Alcohol' if column == 'Alcohol_Consumption' else 'No_' + column
                try:
                    fitted_classes = label_encoders[column].classes_
                    value = input_df[column].iloc[0]
                    if value in fitted_classes:
                        input_df[column] = label_encoders[column].transform([value])[0]
                    else:
                        logger.warning(f"Unseen label '{value}' for column {column}. Using default value -1.")
                        input_df[column] = -1
                except ValueError as e:
                    logger.error(f"Value error for column {column}: {str(e)}")
                    return jsonify({'error': f'Value error for {column}: {str(e)}'}), 500
                except Exception as e:
                    logger.error(f"Unexpected error for column {column}: {str(e)}")
                    return jsonify({'error': f'Unexpected error for {column}: {str(e)}'}), 500

        # Scale numerical columns (e.g., BMI)
        numerical_columns = ['BMI']
        if numerical_columns and any(col in input_df.columns for col in numerical_columns):
            try:
                input_df[numerical_columns] = scaler.transform(input_df[numerical_columns])
            except ValueError as e:
                logger.error(f"Scaling error for numerical columns: {str(e)}")
                return jsonify({'error': f'Invalid numerical value for {numerical_columns}: {str(e)}'}), 400

        # Ensure the input DataFrame has the same columns and order as training data
        X = input_df[required_columns]
        X = X.reindex(columns=model.feature_names_in_, fill_value=0)

        # Make prediction
        prediction = model.predict(X)[0]
        probability = float(model.predict_proba(X)[:, 1][0])

        # Prepare result
        prediction_text = "Yes, the person is predicted to survive." if prediction == 1 else "No, the person is not predicted to survive."
        result = {
            'prediction': prediction_text,
            'probability_survived': probability
        }

        return jsonify(result), 200

    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.getenv('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=True)