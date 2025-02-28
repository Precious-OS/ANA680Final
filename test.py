import requests
url = 'http://localhost:8080/invocations'  # Matches current port
sample_data = {
    "State": "Benue",
    "Age_Group": "Adult",
    "Gender": "Female",
    "BMI": 34.8,
    "Smoking_Status": "Non-Smoker",
    "Alcohol_Consumption": "Low",
    "Exercise_Frequency": "Occasionally",
    "Hypertension": "Yes",
    "Diabetes": "No",
    "Cholesterol_Level": "Normal",
    "Family_History": "No",
    "Stress_Level": "Moderate",
    "Diet_Type": "Mixed",
    "Heart_Attack_Severity": "Moderate",
    "Hospitalized": "Yes",
    "Income_Level": "Medium",
    "Urban_Rural": "Urban",
    "Employment_Status": "Employed"
}
try:
    response = requests.post(url, json=sample_data, timeout=10)
    response.raise_for_status()  # Raise an error for bad status codes
    print("Prediction Response:", response.json())
except requests.exceptions.RequestException as e:
    print(f"Error: {e}")