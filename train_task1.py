import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from pathlib import Path
import joblib



def train_task1_classifier(
    train_path: str,
    test_path: str,
    model_name: str = "all-MiniLM-L6-v2"
):
    #Load processed data
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train_text = train_df["text"].tolist()
    y_train = train_df["task1_label"].values

    X_test_text = test_df["text"].tolist()
    y_test = test_df["task1_label"].values

    #Load sentence embedding model
    print("Loading Sentence-BERT model...")
    encoder = SentenceTransformer(model_name)

    #Encode texts
    print("Encoding training data...")
    X_train_emb = encoder.encode(X_train_text, show_progress_bar=True)

    print("Encoding test data...")
    X_test_emb = encoder.encode(X_test_text, show_progress_bar=True)

    #Train classifier
    print("Training Logistic Regression classifier...")
    clf = LogisticRegression(max_iter=2000)
    clf.fit(X_train_emb, y_train)

    #Evaluate
    print("\nTask 1 – Implicit Sexism Detection Results:")
    report = classification_report(
        y_test,
        clf.predict(X_test_emb),
        target_names=["Not Sexist", "Sexist"]
    )

    print(report)
    return clf, encoder


    #Save results
    results_path = Path("results/task1_baseline_metrics.txt")
    results_path.parent.mkdir(exist_ok=True)

    with open(results_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    train_task1_classifier(
        train_path="data/processed/task1_train.csv",
        test_path="data/processed/task1_test.csv"
    )
# Save model and encoder
if __name__ == "__main__":
    clf, encoder = train_task1_classifier(
    train_path="data/processed/task1_train.csv",
    test_path="data/processed/task1_test.csv"
)

    import joblib
    joblib.dump(clf, "results/task1_baseline_model.joblib")
    joblib.dump(encoder, "results/task1_sentencebert.joblib")

    print("Models saved successfully.")

