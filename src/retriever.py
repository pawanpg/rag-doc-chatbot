import json, pathlib, pandas as pd, numpy as np
import faiss
from sentence_transformers import SentenceTransformer

class Retriever:
    def __init__(self, index_path: pathlib.Path, db_path: pathlib.Path, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.index = faiss.read_index(str(index_path))
        self.df = pd.read_json(db_path, lines=True)
        self.model = SentenceTransformer(model_name)

    def search(self, query: str, k: int = 5):
        qv = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(qv)
        D, I = self.index.search(qv, k)
        hits = self.df.iloc[I[0]].copy()
        hits["score"] = D[0]
        return hits[["id", "path", "chunk", "score"]].to_dict(orient="records")
