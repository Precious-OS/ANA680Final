**ANA680 Final Project: Heart Attack Predictor**



**Overview**

This repository contains a machine learning solution developed for ANA680, designed to predict the likelihood of heart attacks based on various health and lifestyle factors using the XGBoost algorithm on Amazon SageMaker. The project leverages a dataset of Nigerian youth and adult health records to train and deploy a predictive model, providing actionable insights for healthcare applications.

This project demonstrates end-to-end machine learning workflows, including data preprocessing, model training, and deployment as a real-time inference endpoint on SageMaker, without the use of Docker containers.

**Features**
- Data Preprocessing: Handles categorical and numerical features with label encoding and standardization.
- Model Training: Utilizes XGBoost for robust prediction of heart attack risk.
- Real-Time Inference: Deploys the model as a SageMaker endpoint for live predictions.
- Scalability: Leverages AWS SageMaker for scalable training and deployment.
- Health Check: Includes a ping endpoint for monitoring model health.



**Dataset**

The project uses a synthetic dataset (heart_attack_youth_vs_adult_nigeria.csv) stored in an S3 bucket (s3://sagemaker-bucket-heart/), containing features such as:

State

Age 

Group

Gender

BMI

Smoking Status

Alcohol Consumption

Exercise Frequency

Hypertension

Diabetes

Cholesterol Level

Family History

Stress Level

Diet Type

Heart Attack Severity

Hospitalized

Income Level

Urban/Rural

Employment Status


The target variable is a binary classification indicating the presence or absence of a heart attack risk.
