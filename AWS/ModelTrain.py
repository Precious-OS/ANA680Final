# /home/ec2-user/SageMaker/ModelTrain.py
import argparse
import joblib
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import xgboost as xgb

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=6)
    args = parser.parse_args()

    # Load data from S3
    train_data_path = os.path.join(args.train, "heart_attack_youth_vs_adult_nigeria.csv")
    if not os.path.exists(train_data_path):
        raise FileNotFoundError(f"Training data not found at {train_data_path}")
    train_data = pd.read_csv(train_data_path, na_values=[''])
    print(f"Dataset loaded successfully! Shape: {train_data.shape}")

    # Preprocessing
    X = train_data.drop(columns=['Survived'])
    y = train_data['Survived']

    # Encode the target variable 'Survived'
    le_target = LabelEncoder()
    y = le_target.fit_transform(y)  # Convert 'No' and 'Yes' to 0 and 1
    print(f"LabelEncoder classes: {le_target.classes_}")  # Verify classes
    print(f"LabelEncoder type before saving: {type(le_target)}")  # Debug type

    # Encode categorical features
    label_encoders = {}
    for column in X.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        X[column] = le.fit_transform(X[column])
        label_encoders[column] = le

    # Scale numerical features
    scaler = StandardScaler()
    numerical_cols = X.select_dtypes(include=['float64', 'int64']).columns
    X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

    # Verify shapes before split
    print(f"X shape before split: {X.shape}, y shape before split: {y.shape}")

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")

    # Train XGBoost model
    model = xgb.XGBClassifier(
        eval_metric='logloss',
        random_state=42,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        objective="binary:logistic"
    )
    model.fit(X_train, y_train)

    # Save the model, label encoder, and scaler
    os.makedirs(args.model_dir, exist_ok=True)
    model_path = os.path.join(args.model_dir, "NigeriaHeartAttack.pkl")
    joblib.dump(model, model_path)
    joblib.dump(le_target, os.path.join(args.model_dir, "label_encoder.pkl"))  # Save the correct object
    joblib.dump(scaler, os.path.join(args.model_dir, "scaler.pkl"))
    print(f"Model, label encoder, and scaler saved successfully to {args.model_dir}")
    print(f"LabelEncoder type after saving: {type(joblib.load(os.path.join(args.model_dir, 'label_encoder.pkl')))}")  # Verify saved type