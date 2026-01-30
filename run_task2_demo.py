import joblib
import pandas as pd

from sentence_transformers import SentenceTransformer

from retrieval import SexismRetriever
from explain import generate_explanation

MODEL_PATH = "results/task1_baseline_model.joblib"
VECTORIZER_PATH = "results/task1_sentencebert.joblib"
TOP_K = 3


def main():
    #Example input 
    input_text = "I will go for a run with my dogs today"

    print("=" * 60)
    print("TASK 2 – EXPLAINABLE IMPLICIT SEXISM DETECTION")
    print("=" * 60)
    print(f"Input: {input_text}")
    print()

    # Load classifier & encoder
    print("Loading classifier and encoder...")
    classifier = joblib.load(MODEL_PATH)
    encoder = joblib.load(VECTORIZER_PATH)

    #Encode input
    input_embedding = encoder.encode([input_text])

    #Predict
    prediction = int(classifier.predict(input_embedding)[0])

    #Retrieval
    retriever = SexismRetriever()
    retrieved_items = retriever.retrieve(input_text, top_k=TOP_K)
    #Explanation
    explanation = generate_explanation(
        text=input_text,
        prediction=prediction,
        retrieved_items=retrieved_items
    )

    print(explanation)
    print("=" * 60)


if __name__ == "__main__":
    main()
