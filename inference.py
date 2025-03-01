from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import logging
import os

app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load model, encoders, and scaler with logging
logger.info("Loading model...")
model = joblib.load("NigeriaHeartAttack.pkl")
logger.info("Model loaded successfully")
label_encoders = joblib.load("label_encoders.pkl")
logger.info("Label encoders loaded successfully")
scaler = joblib.load("scaler.pkl")
logger.info("Scaler loaded successfully")

categorical_columns = ['State', 'Age_Group', 'Gender', 'Smoking_Status', 'Alcohol_Consumption',
                      'Exercise_Frequency', 'Hypertension', 'Diabetes', 'Cholesterol_Level',
                      'Family_History', 'Stress_Level', 'Diet_Type', 'Heart_Attack_Severity',
                      'Hospitalized', 'Income_Level', 'Urban_Rural', 'Employment_Status']

@app.route('/ping', methods=['GET'])
def ping():
    logger.info("Ping received")
    return jsonify({'status': 'healthy'}), 200

@app.route('/invocations', methods=['POST'])
def predict():
    logger.info("Invocation received")
    try:
        form_data = request.get_json()
        if not form_data:
            logger.error("No JSON data received")
            return jsonify({'error': 'No data received'}), 400
        input_df = pd.DataFrame([form_data])
        required_columns = ['State', 'Age_Group', 'Gender', 'BMI', 'Smoking_Status', 'Alcohol_Consumption',
                          'Exercise_Frequency', 'Hypertension', 'Diabetes', 'Cholesterol_Level',
                          'Family_History', 'Stress_Level', 'Diet_Type', 'Heart_Attack_Severity',
                          'Hospitalized', 'Income_Level', 'Urban_Rural', 'Employment_Status']
        missing_columns = [col for col in required_columns if col not in input_df or pd.isna(input_df[col].iloc[0])]
        if missing_columns:
            return jsonify({'error': f'Missing columns: {missing_columns}'}), 400
        if 'BMI' in input_df and (pd.isna(input_df['BMI'].iloc[0]) or not str(input_df['BMI'].iloc[0]).replace('.', '').isdigit()):
            return jsonify({'error': 'Invalid BMI value'}), 400
        for column in categorical_columns:
            if column in input_df:
                if pd.isna(input_df[column].iloc[0]) or input_df[column].iloc[0] == 'None':
                    input_df[column] = 'No_Alcohol' if column == 'Alcohol_Consumption' else 'No_' + column
                try:
                    input_df[column] = label_encoders[column].transform([input_df[column].iloc[0]])[0]
                except (ValueError, KeyError) as e:
                    logger.error(f"Encoding error for {column}: {e}")
                    input_df[column] = -1
        numerical_columns = ['BMI']
        if numerical_columns and any(col in input_df.columns for col in numerical_columns):
            input_df[numerical_columns] = scaler.transform(input_df[numerical_columns])
        X = input_df[required_columns].reindex(columns=model.feature_names_in_, fill_value=0)
        logger.info("Predicting...")
        prediction = model.predict(X)[0]
        probability = float(model.predict_proba(X)[:, 1][0])
        logger.info(f"Prediction: {prediction}, Probability: {probability}")
        return jsonify({
            'prediction': "Yes, the person is predicted to survive." if prediction == 1 else "No, the person is not predicted to survive.",
            'probability_survived': probability
        }), 200
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.getenv('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=True)