from fastapi import FastAPI
from pydantic import BaseModel
import pathlib
from retriever import Retriever
from llm import synthesize_answer

app = FastAPI(title="RAG Doc Chatbot")
ret = None

class AskReq(BaseModel):
    query: str
    k: int = 5

@app.on_event("startup")
def _load():
    global ret
    idx = pathlib.Path("outputs/faiss.index")
    db = pathlib.Path("outputs/docstore.pkl")
    if idx.exists() and db.exists():
        ret = Retriever(idx, db)
    else:
        ret = None

@app.post("/ask")
def ask(req: AskReq):
    if ret is None:
        return {"error": "Index not built. Run ingest first."}
    hits = ret.search(req.query, k=req.k)
    answer = synthesize_answer(req.query, hits)
    return {"answer": answer, "sources": hits}
