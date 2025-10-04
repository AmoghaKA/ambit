# retrain_model.py
"""
Retrain a 10-feature breast cancer classifier (mean_* features).
Saves model as model_pipeline_10.pkl
"""

import joblib
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, classification_report

def main():
    print("Loading dataset...")
    data = load_breast_cancer(as_frame=True)
    df = data.frame  # dataframe including 'target' and feature columns

    # 10 realistic mean features (note the spaces — matches sklearn's column names)
    features_10 = [
        "mean radius", "mean texture", "mean perimeter", "mean area", "mean smoothness",
        "mean compactness", "mean concavity", "mean concave points", "mean symmetry", "mean fractal dimension"
    ]

    # verify columns exist
    missing = [f for f in features_10 if f not in df.columns]
    if missing:
        raise RuntimeError(f"Expected features missing in dataset: {missing}")

    X = df[features_10]
    y = df["target"]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # pipeline with scaler + XGBoost classifier
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", XGBClassifier(
            use_label_encoder=False,
            eval_metric="logloss",
            n_estimators=200,
            learning_rate=0.2,
            max_depth=3,
            subsample=1.0,
            colsample_bytree=1.0,
            reg_alpha=0.1,
            reg_lambda=5,
            random_state=42
        ))
    ])

    print("Training...")
    pipeline.fit(X_train, y_train)

    print("Evaluating...")
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f"Accuracy: {acc*100:.2f}%")
    print(f"ROC AUC: {roc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Save model
    joblib.dump(pipeline, "model_pipeline_10.pkl")
    print("\n✅ Saved model as model_pipeline_10.pkl")

if __name__ == "__main__":
    main()
