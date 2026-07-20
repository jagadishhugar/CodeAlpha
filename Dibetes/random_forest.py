import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# 1. Load Dataset
df = pd.read_csv('diabetes.csv')

# 2. Data Preprocessing
zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in zero_cols:
    df[col] = df[col].replace(0, np.nan)
    df[col] = df[col].fillna(df[col].median())

# 3. Split Features and Target
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Tree ensembles don't strictly require feature scaling
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 4. Model Training
model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=6)
model.fit(X_train, y_train)

# 5. Evaluation
y_pred = model.predict(X_test)
print("=== RANDOM FOREST MODEL ===")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 6. Bonus: View diagnostic driver weights
importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nTop Predictor Features:")
print(importances)