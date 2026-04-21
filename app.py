from fastapi import FastAPI
from pydantic import BaseModel
from query import query_rag  # your existing function

app = FastAPI()


# Request body
class QueryRequest(BaseModel):
    query: str


# Health check
@app.get("/")
def health():
    return {"status": "running"}


# Main endpoint
@app.post("/ask")
def ask_question(req: QueryRequest):
    answer = query_rag(req.query, stream=False)
    return {"answer": answer}