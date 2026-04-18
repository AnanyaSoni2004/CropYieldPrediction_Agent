"""
Train and persist the Random Forest crop classifier.

Run once before starting the API:
    python models/train_model.py
"""
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "crop_data.csv")
MODEL_DIR = os.path.dirname(__file__)

FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]


def train_and_save():
    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)

    X = df[FEATURES].values
    y = df["label"].values

    encoder = LabelEncoder()
    y_enc = encoder.fit_transform(y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )

    print("Training Random Forest classifier...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=encoder.classes_))

    cv_scores = cross_val_score(model, X_scaled, y_enc, cv=5, scoring="accuracy")
    print(f"5-Fold CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Feature importance
    importances = model.feature_importances_
    feat_imp = sorted(zip(FEATURES, importances), key=lambda x: x[1], reverse=True)
    print("\nFeature Importances:")
    for feat, imp in feat_imp:
        print(f"  {feat:15s}: {imp:.4f}")

    # Persist artifacts
    model_path = os.path.join(MODEL_DIR, "crop_model.pkl")
    scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
    encoder_path = os.path.join(MODEL_DIR, "label_encoder.pkl")

    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    with open(encoder_path, "wb") as f:
        pickle.dump(encoder, f)

    print(f"\nArtifacts saved to {MODEL_DIR}/")
    return model, scaler, encoder


if __name__ == "__main__":
    train_and_save()
