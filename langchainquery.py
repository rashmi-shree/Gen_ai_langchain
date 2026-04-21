import os
from dotenv import load_dotenv

from langchain_core.retrievers import BaseRetriever
from typing import List
from langchain_core.documents import Document
from langchain_core.callbacks import StreamingStdOutCallbackHandler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

load_dotenv()

class CustomRetriever(BaseRetriever):
    vectorstore: any
    llm: any
    k: int = 10

    def _get_relevant_documents(self, query: str) -> List[Document]:
        # 🔹 Semantic search
        docs = self.vectorstore.similarity_search(query, k=self.k)

        # 🔹 Keyword search
        docs_keyword = keyword_search(query, docs, top_k=5)

        # 🔹 Combine
        combined = docs + docs_keyword

        # 🔹 Deduplicate
        seen = set()
        unique_docs = []
        for doc in combined:
            if doc.page_content not in seen:
                unique_docs.append(doc)
                seen.add(doc.page_content)

        # 🔹 Re-rank
        ranked_docs = llm_re_rank_docs(query, unique_docs, self.llm, top_n=3)

        return ranked_docs
# 🔹 Load embeddings (same as ingestion)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 🔹 Load vectorstore
vectorstore = FAISS.load_local("lc_vectorstore", embeddings, allow_dangerous_deserialization=True)



# 🔹 LLM (Groq via OpenAI-compatible API)
llm = ChatOpenAI(
    model="openai/gpt-oss-120b",
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("GROQ_API_KEY"),
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()]
)

# 🔹 Create retriever
# retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
retriever = CustomRetriever(vectorstore=vectorstore, llm=llm, k=10)
def keyword_search(query, docs, top_k=5):
    texts = [doc.page_content for doc in docs]

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)

    query_vec = vectorizer.transform([query])

    scores = cosine_similarity(query_vec, tfidf_matrix).flatten()

    top_indices = scores.argsort()[-top_k:][::-1]

    return [docs[i] for i in top_indices]

def llm_re_rank_docs(query, docs, llm, top_n=3):
    scored_docs = []

    for doc in docs:
        prompt = f"""
You are a relevance judge.

Query:
{query}

Document:
{doc.page_content}

Rate relevance from 1 to 10.
Only return a number.
"""

        response = llm.invoke(prompt)

        try:
            score = float(response.content.strip())
        except:
            score = 0

        scored_docs.append((doc, score))

    scored_docs.sort(key=lambda x: x[1], reverse=True)

    return [doc for doc, _ in scored_docs[:top_n]]

# 🔹 Query loop
while True:
    query = input("Ask: ")

    if query.lower() in ["exit", "quit"]:
        break

    # 🔹 Semantic search
    docs_semantic = retriever.invoke(query)

    # 🔹 Keyword search (on same docs)
    docs_keyword = keyword_search(query, docs_semantic, top_k=5)

    # 🔹 Combine
    docs = docs_semantic + docs_keyword

    # 🔹 Remove duplicates
    unique_docs = []
    seen = set()

    for doc in docs:
        if doc.page_content not in seen:
            unique_docs.append(doc)
            seen.add(doc.page_content)

    docs = unique_docs

    print("\n--- Retrieved Docs ---")
    for i, doc in enumerate(docs):
        print(f"\nDoc {i+1}:\n{doc.page_content}")

    # 🔥 Apply YOUR re-ranking
    docs = llm_re_rank_docs(query, docs, llm, top_n=3)

    print("\n--- Re-ranked Docs ---")
    for i, doc in enumerate(docs):
        print(f"\nDoc {i+1}:\n{doc.page_content}")

    print("\n--- Retrieved Docs ---")
    for i, doc in enumerate(docs):
        print(f"\nDoc {i+1}:\n{doc.page_content}")

    context = "\n".join([doc.page_content for doc in docs])

    prompt = f"""
You are a strict AI assistant.

Use ONLY the context below.
If not found, say "I don't know".

Context:
{context}

Question:
{query}
"""

    print("\n⚡ Generating response...\n")

    llm.invoke(prompt)

    print("\n")  # newline after streaming