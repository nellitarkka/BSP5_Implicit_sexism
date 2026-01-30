import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from pathlib import Path

from augmentation import augment_text


def train_task1_classifier_dda_cse(
    train_path: str,
    test_path: str,
    model_name: str = "all-MiniLM-L6-v2"
):
    #Load processed data
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    #Apply DDA + CSE ONLY to sexist training samples
    train_df["augmented_text"] = train_df.apply(
        lambda row: augment_text(row["text"])
        if row["task1_label"] == 1
        else row["text"],
        axis=1
    )

    X_train_text = train_df["augmented_text"].tolist()
    y_train = train_df["task1_label"].values

    X_test_text = test_df["text"].tolist()
    y_test = test_df["task1_label"].values

    print("Loading Sentence-BERT model...")
    encoder = SentenceTransformer(model_name)

    print("Encoding training data (DDA + CSE)...")
    X_train_emb = encoder.encode(X_train_text, show_progress_bar=True)

    print("Encoding test data...")
    X_test_emb = encoder.encode(X_test_text, show_progress_bar=True)

    print("Training Logistic Regression classifier...")
    clf = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

    clf.fit(X_train_emb, y_train)

    print("\nTask 1 – DDA + CSE Results:")
    report = classification_report(
        y_test,
        clf.predict(X_test_emb),
        target_names=["Not Sexist", "Sexist"]
    )

    print(report)

    results_path = Path("results/task1_dda_cse_metrics.txt")
    with open(results_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    train_task1_classifier_dda_cse(
        train_path="data/processed/task1_train.csv",
        test_path="data/processed/task1_test.csv"
    )
