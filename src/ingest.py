import argparse, pathlib, json, hashlib
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import faiss
from PyPDF2 import PdfReader

def read_text(path):
    if path.suffix.lower() in {".txt", ".md"}:
        return path.read_text(encoding="utf-8", errors="ignore")
    if path.suffix.lower() == ".pdf":
        text = []
        try:
            reader = PdfReader(str(path))
            for page in reader.pages:
                text.append(page.extract_text() or "")
            return "\n".join(text)
        except Exception as e:
            return ""
    return ""

def chunk_text(text, chunk_size=600, overlap=120):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i+chunk_size])
        if chunk.strip():
            chunks.append(chunk)
        i += (chunk_size - overlap)
    return chunks

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs_dir", type=pathlib.Path, required=True)
    ap.add_argument("--index_path", type=pathlib.Path, required=True)
    ap.add_argument("--db_path", type=pathlib.Path, required=True)
    ap.add_argument("--model_name", default="sentence-transformers/all-MiniLM-L6-v2")
    args = ap.parse_args()

    args.index_path.parent.mkdir(parents=True, exist_ok=True)
    args.db_path.parent.mkdir(parents=True, exist_ok=True)

    model = SentenceTransformer(args.model_name)

    records = []
    for file in tqdm(list(args.docs_dir.rglob("*.*"))):
        if file.suffix.lower() not in {".txt", ".md", ".pdf"}:
            continue
        text = read_text(file)
        for ch in chunk_text(text):
            uid = hashlib.md5((str(file)+ch).encode("utf-8")).hexdigest()
            records.append({"id": uid, "path": str(file), "chunk": ch})

    import numpy as np
    embeddings = model.encode([r["chunk"] for r in records], batch_size=64, show_progress_bar=True, convert_to_numpy=True)

    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)

    faiss.write_index(index, str(args.index_path))
    with open(args.db_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Indexed {len(records)} chunks.")

if __name__ == "__main__":
    main()
