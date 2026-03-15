from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import List
from vector_store.search_with_ranker import MultimodalSearchEngine

# -------------------------
# App
# -------------------------
app = FastAPI(
    title="Multimodal Search API",
    description="CLIP + FAISS + Learning-to-Rank",
    version="1.0"
)

# -------------------------
# Load engine (once)
# -------------------------
engine = MultimodalSearchEngine()

# -------------------------
# Response schema
# -------------------------
class SearchResult(BaseModel):
    rank: int
    image: str
    description: str
    clip_similarity: float
    rank_score: float


# -------------------------
# Search endpoint
# -------------------------
@app.get("/search", response_model=List[SearchResult])
def search(
    q: str = Query(..., description="Search query"),
    top_k: int = Query(5, ge=1, le=10)
):
    """
    Run multimodal search with re-ranking
    """
    results = engine.search(q)
    return results[:top_k]
