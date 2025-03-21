from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import logging
import os  # Add this import for accessing environment variables

app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load the trained model
model_path = "NigeriaHeartAttack.pkl"  # Update this path if the model is in a different location
try:
    model = joblib.load(model_path)
    print("Model loaded successfully!")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    raise

# Load the fitted encoders and scaler
try:
    label_encoders_path = "label_encoders.pkl"  # Update this path to match where you saved it
    scaler_path = "scaler.pkl"  # Update this path to match where you saved it
    label_encoders = joblib.load(label_encoders_path)
    scaler = joblib.load(scaler_path)
    print("Encoders and scaler loaded successfully!")
except Exception as e:
    logger.error(f"Error loading encoders or scaler: {e}")
    raise

# Define categorical columns (must match training)
categorical_columns = ['State', 'Age_Group', 'Gender', 'Smoking_Status', 'Alcohol_Consumption',
                      'Exercise_Frequency', 'Hypertension', 'Diabetes', 'Cholesterol_Level',
                      'Family_History', 'Stress_Level', 'Diet_Type', 'Heart_Attack_Severity',
                      'Hospitalized', 'Income_Level', 'Urban_Rural', 'Employment_Status']

# Home route to display the form (served from templates folder)
@app.route('/')
def home():
    return render_template('index.html')  # Serve index.html from the templates folder

# Route to handle the prediction (via POST from the form)
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from the request (since fetch sends application/json)
        form_data = request.json
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
                # Handle "None" or missing values in categorical columns
                if pd.isna(input_df[column].iloc[0]) or input_df[column].iloc[0] == 'None':
                    input_df[column] = 'No_Alcohol' if column == 'Alcohol_Consumption' else 'No_' + column
                # Encode categorical variables using the fitted LabelEncoder, handling unseen labels
                try:
                    # Get the classes the encoder was fitted on
                    fitted_classes = label_encoders[column].classes_
                    value = input_df[column].iloc[0]
                    if value in fitted_classes:
                        input_df[column] = label_encoders[column].transform([value])[0]
                    else:
                        logger.warning(f"Unseen label '{value}' for column {column}. Using default value -1.")
                        input_df[column] = -1  # Use -1 or another default value for unseen labels
                except ValueError as e:
                    logger.error(f"Value error for column {column}: {str(e)}")
                    return jsonify({'error': f'Value error for {column}: {str(e)}'}), 500
                except Exception as e:
                    logger.error(f"Unexpected error for column {column}: {str(e)}")
                    return jsonify({'error': f'Unexpected error for {column}: {str(e)}'}), 500

        # Scale numerical columns (e.g., BMI)
        numerical_columns = ['BMI']  # Add other numerical columns if any (e.g., Stress_Level if numeric)
        if numerical_columns and any(col in input_df.columns for col in numerical_columns):
            try:
                input_df[numerical_columns] = scaler.transform(input_df[numerical_columns])
            except ValueError as e:
                logger.error(f"Scaling error for numerical columns: {str(e)}")
                return jsonify({'error': f'Invalid numerical value for {numerical_columns}: {str(e)}'}), 400

        # Ensure the input DataFrame has the same columns and order as training data
        X = input_df[required_columns]
        X = X.reindex(columns=model.feature_names_in_, fill_value=0)  # Align with training feature order

        # Make prediction
        prediction = model.predict(X)[0]
        probability = float(model.predict_proba(X)[:, 1][0])  # Convert numpy.float32 to standard Python float

        # Prepare result for rendering in template
        prediction_text = "Yes, the person is predicted to survive." if prediction == 1 else "No, the person is not predicted to survive."
        probability_text = f" (Probability of survival: {probability:.2f})"
        result = f"{prediction_text}{probability_text}"

        # Return the prediction as JSON (since fetch expects JSON)
        return jsonify({
            'prediction': prediction_text,
            'probability_survived': probability  # Now a standard Python float, JSON-serializable
        }), 200

    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Run the Flask app
if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))  # Use PORT environment variable, default to 5000 if not set
    app.run(debug=True, host='0.0.0.0', port=port)