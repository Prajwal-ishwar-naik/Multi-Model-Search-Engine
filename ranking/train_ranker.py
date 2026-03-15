import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

FEATURES_PATH = "data/processed/ranking_features.csv"
MODEL_DIR = "data/models"
MODEL_PATH = os.path.join(MODEL_DIR, "ranker.pkl")


def main():
    print("📥 Loading ranking features...")

    if not os.path.exists(FEATURES_PATH):
        raise FileNotFoundError(
            "❌ data/processed/ranking_features.csv not found. "
            "Run feature_engineering.py first."
        )

    df = pd.read_csv(FEATURES_PATH)

    print(f"📊 Training samples: {len(df)}")

    X = df[
        ["clip_similarity", "text_length", "image_brightness"]
    ]
    y = df["label"]

    # 🔑 If dataset is too small, don't stratify
    stratify = y if y.value_counts().min() >= 2 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2 if len(df) > 5 else 0.3,
        random_state=42,
        stratify=stratify
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("\n📈 Training Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print(f"\n✅ Ranking model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
