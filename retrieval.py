from pathlib import Path
from typing import List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class SexismRetriever:
    def __init__(
        self,
        corpus_path: str = "data/raw/definitions_sexism.txt",
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        """
        Semantic retriever for sexism definitions/examples.
        """
        self.corpus_path = Path(corpus_path)
        self.model = SentenceTransformer(model_name)

        self.corpus = self._load_corpus()
        self.corpus_embeddings = self.model.encode(
            self.corpus,
            convert_to_numpy=True,
            show_progress_bar=False
        )

    def _load_corpus(self) -> List[str]:
        """
        Load definitions/examples from text file.
        Each non-empty line is treated as a retrievable unit.
        """
        with open(self.corpus_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f.readlines()]
        return [line for line in lines if len(line) > 0]

    def retrieve(
        self,
        query: str,
        top_k: int = 3
    ) -> List[Tuple[str, float]]:
        """
        Retrieve top-k most similar corpus entries.
        Returns (text, similarity_score).
        """
        query_emb = self.model.encode(
            [query],
            convert_to_numpy=True
        )

        similarities = cosine_similarity(query_emb, self.corpus_embeddings)[0]
        top_indices = np.argsort(similarities)[::-1][:top_k]

        return [
            (self.corpus[idx], float(similarities[idx]))
            for idx in top_indices
        ]
