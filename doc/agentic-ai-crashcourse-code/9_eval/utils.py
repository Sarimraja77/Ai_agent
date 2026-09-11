from sentence_transformers import SentenceTransformer
import numpy as np
import logging
from dotenv import load_dotenv
load_dotenv()

logging.getLogger("transformers").setLevel(logging.ERROR)

_model = SentenceTransformer("all-MiniLM-L6-v2")


def cosine_similarity(sentence1: str, sentence2: str) -> float:
    embeddings = _model.encode([sentence1, sentence2])
    a, b = embeddings[0], embeddings[1]
    similarity = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.clip(similarity, 0.0, 1.0))


if __name__ == "__main__":
    pairs = [
        ("The cat sat on the mat.", "A feline rested on the rug."),
        ("The cat sat on the mat.", "The dog barked at the mailman."),
        ("Python is great.", "I love programming in Python."),
    ]
    for s1, s2 in pairs:
        sim = cosine_similarity(s1, s2)
        print(f"{sim:.4f}  |  \"{s1}\" vs \"{s2}\"")
