"""Create and save a small sklearn pipeline to model/model.pkl
This helps ensure the FastAPI server can load a valid model file.
"""
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib, os

def main():
    X, y = load_breast_cancer(return_X_y=True)
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=1000))])
    pipe.fit(X, y)
    os.makedirs("model", exist_ok=True)
    joblib.dump(pipe, "model/model.pkl")
    print("Saved model to model/model.pkl")

if __name__ == '__main__':
    main()
