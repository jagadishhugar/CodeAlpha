# model_logistic.py
from data_preprocessing import load_and_preprocess_data
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score


def run_logistic_regression():
    print("--- Running Logistic Regression Credit Scoring Model ---")

    # Load the synchronized preprocessed data
    X_train, X_test, y_train, y_test = load_and_preprocess_data()

    # Initialize Logistic Regression with balanced weights to handle minority default class
    model = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)

    # Train
    model.fit(X_train, y_train)

    # Predict
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    # Evaluate
    print("\n[Classification Metrics]")
    print(classification_report(y_test, preds))
    print(f"ROC-AUC Performance Score: {roc_auc_score(y_test, probs):.4f}")


if __name__ == "__main__":
    # Ensure you have credit_data.csv in the same directory before running
    run_logistic_regression()
    #C:\Users\JAGADISH H\credit_scoring