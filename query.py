# query.py

from sentence_transformers import SentenceTransformer
import faiss
import pickle
from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

# ✅ GLOBAL CACHES
query_cache = {}
embedding_cache = {}

model = SentenceTransformer("all-MiniLM-L6-v2")
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


# 🔹 Embedding cache
def get_embedding(text):
    if text in embedding_cache:
        return embedding_cache[text]

    emb = model.encode([text])[0]
    embedding_cache[text] = emb
    return emb


# 🔹 LLM re-ranking
def llm_re_rank_chunks(query, chunks, client, top_n=3):
    scored_chunks = []

    for chunk in chunks:
        prompt = f"""
You are a relevance judge.

Given the query and a chunk of text, rate how useful this chunk is for answering the query.

Query:
{query}

Chunk:
{chunk}

Respond with ONLY a number from 1 to 10.
"""

        response = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[{"role": "user", "content": prompt}],
        )

        score_text = response.choices[0].message.content.strip()

        try:
            score = float(score_text)
        except:
            score = 0

        scored_chunks.append((chunk, score))

    scored_chunks.sort(key=lambda x: x[1], reverse=True)

    return [chunk for chunk, _ in scored_chunks[:top_n]]


# 🔹 Main RAG function
def query_rag(user_query, top_k=10):

    # ✅ Cache check
    if user_query in query_cache:
        print("⚡ Cache hit!")
        print(query_cache[user_query])
        return query_cache[user_query]

    # Load index
    index = faiss.read_index("embeddings/index.faiss")

    # Load chunks
    with open("embeddings/chunks.pkl", "rb") as f:
        chunks = pickle.load(f)

    # ✅ Use cached embedding
    query_embedding = get_embedding(user_query).reshape(1, -1)

    # Search
    distances, indices = index.search(query_embedding, top_k)

    # Map indices → chunks
    retrieved_chunks = [chunks[i] for i in indices[0]]

    # Remove duplicates + garbage
    retrieved_chunks = list(dict.fromkeys(retrieved_chunks))
    retrieved_chunks = [c for c in retrieved_chunks if len(c.strip()) > 20]

    print("\n--- Retrieved Chunks ---")
    for i, chunk in enumerate(retrieved_chunks):
        print(f"\nChunk {i+1}:\n{chunk}")

    # 🔥 LLM Re-ranking
    retrieved_chunks = llm_re_rank_chunks(user_query, retrieved_chunks, client, top_n=3)

    if not retrieved_chunks:
        return "I don't know"

    print("\n--- Re-ranked Chunks ---")
    for i, chunk in enumerate(retrieved_chunks):
        print(f"\nChunk {i+1}:\n{chunk}")

    # Create context
    context = "\n".join(retrieved_chunks)

    prompt = f"""
You are a strict AI assistant.

Use ONLY the context below to answer.
Do NOT use prior knowledge.
If the answer is not explicitly present, say "I don't know".

Context:
{context}

Question:
{user_query}
"""

    # 🔥 STREAMING RESPONSE
    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": prompt}],
        stream=True
    )

    print("\n⚡ Generating response...\n")

    answer = ""

    for chunk in response:
        if chunk.choices[0].delta.content:
            token = chunk.choices[0].delta.content
            print(token, end="", flush=True)
            answer += token

    print("\n")

    # ✅ Store in cache
    query_cache[user_query] = answer

    return answer