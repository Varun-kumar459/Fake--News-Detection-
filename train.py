"""Train, evaluate and save the model:  python train.py"""
import json

import sklearn

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB  # noqa: F401  (try as an alternative)
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier  # noqa: F401

import config
from src.data_loader import load_dataset
from src.model import build_pipeline


def main():
    df = load_dataset()
    print("Class balance:\n", df["label"].value_counts().rename({0: "Real", 1: "Fake"}))

    X = pd.DataFrame({"clean_text": df["clean_text"], "raw_text": df["text"]})
    y = df["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, stratify=y
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    print(f"\nAccuracy: {acc:.4f} | F1 (Fake): {f1:.4f}\n")
    print(classification_report(y_test, pred, target_names=["Real", "Fake"]))
    cm = confusion_matrix(y_test, pred)
    print("Confusion matrix [rows=true, cols=pred]:\n", cm)

    config.MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(pipeline, config.MODEL_PATH)
    config.METRICS_PATH.write_text(json.dumps(
        {"accuracy": round(acc, 4), "f1": round(f1, 4), "sklearn_version": sklearn.__version__,
         "train_size": len(X_train), "test_size": len(X_test)}, indent=2))
    print(f"\nModel saved to {config.MODEL_PATH}")


if __name__ == "__main__":
    main()
