from sentence_transformers import SentenceTransformer
import faiss
import pickle
from utils import chunk_text

model = SentenceTransformer("all-MiniLM-L6-v2")

def create_vector_store(file_path):
    with open(file_path, "r") as f:
        text = f.read()

    chunks = chunk_text(text)

    embeddings = model.encode(chunks)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)

    index.add(embeddings)

    # Save FAISS index
    faiss.write_index(index, "embeddings/index.faiss")

    # Save chunks
    with open("embeddings/chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)

    print("✅ Ingestion complete")