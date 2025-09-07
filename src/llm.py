# Optional: wire up an OpenAI-compatible client or local LLM here.
# For now, we provide a cheap extractive summarizer as a fallback.
def synthesize_answer(query: str, hits: list[dict]) -> str:
    """Simple extractive summary: return the most relevant chunks stitched together.
    Replace with an LLM call if available.
    """
    if not hits:
        return "No relevant context found."
    contexts = [h["chunk"] for h in hits]
    # Naive strategy: show top 2 chunks
    return "\n\n".join(contexts[:2])
