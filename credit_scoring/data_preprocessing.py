# data_preprocessing.py
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer


def load_and_preprocess_data(filepath=None):
    # Fallback default path if none is provided
    if filepath is None:
        filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "credit_data.csv")

    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Could not find the dataset at: {filepath}. Please ensure 'credit_data.csv' is in the folder.")

    # Load raw dataset
    df = pd.read_csv(filepath)

    # 1. Parse Dates safely
    df['OpenDate'] = pd.to_datetime(df['OpenDate'], errors='coerce')
    df['ClosedDate'] = pd.to_datetime(df['ClosedDate'], errors='coerce')

    # Define an anchor date for active accounts (current reference point)
    current_date = pd.to_datetime('2026-07-18')

    # 2. Feature Engineering
    # Calculate account age in days
    df['AccountTenureDays'] = (df['ClosedDate'].fillna(current_date) - df['OpenDate']).dt.days
    df['AccountTenureDays'] = df['AccountTenureDays'].fillna(0)  # Handle edge cases if OpenDate is missing

    # Binary flag if account has a closed date
    df['IsClosed'] = df['ClosedDate'].notna().astype(int)

    # Clean up column names to avoid trailing whitespaces
    df.columns = df.columns.str.strip()

    # 3. Dynamic Target Mapping (Fixes the Single-Class ValueError)
    unique_statuses = df['Status'].dropna().unique()

    # Look for common default/risk indicator keywords case-insensitively
    risk_keywords = ['default', 'delinquent', 'charged', 'risk', 'late', 'closed']
    risk_classes = [status for status in unique_statuses if any(kw in str(status).lower() for kw in risk_keywords)]

    # Fallback: If no explicit risk names are matched, assign the minority class as the target risk (1)
    if len(risk_classes) == 0 or len(risk_classes) == len(unique_statuses):
        minority_class = df['Status'].value_counts().index[-1]
        risk_classes = [minority_class]

    # Map target: 1 for Risk/Default, 0 for Good Standing
    y = df['Status'].apply(lambda x: 1 if x in risk_classes else 0)

    # Diagnostic Output to console
    print(f"[Data Check] Identified Risk Classes in 'Status': {risk_classes}")
    print(f"[Class Distribution] Class 0 (Good): {sum(y == 0)} | Class 1 (Risk): {sum(y == 1)}")

    if len(np.unique(y)) < 2:
        raise ValueError("CRITICAL ERROR: The processed target variable still contains only 1 class. "
                         "Please manually adjust the mapping rule in data_preprocessing.py.")

    # Drop unique identifier strings that have no predictive capacity
    X = df.drop(columns=['AccountID', 'CustomerID', 'RegistrationID', 'OpenDate', 'ClosedDate', 'Status'],
                errors='ignore')

    # 4. Train/Test Split with Stratification (preserves class ratios)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Identify processing columns dynamically based on types
    numeric_features = ['Balance', 'AccountTenureDays']
    categorical_features = ['AccountType', 'IsClosed']

    # 5. Build Pipeline Transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', ColumnTransformer([
                ('imputer', SimpleImputer(strategy='median'), numeric_features),
                ('scaler', StandardScaler(), numeric_features)
            ]), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), categorical_features)
        ]
    )

    # Apply processing matrices
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    return X_train_processed, X_test_processed, y_train, y_test


if __name__ == "__main__":
    print("Pre-processing configuration finalized.")
