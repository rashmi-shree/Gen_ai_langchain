# main.py

from ingest import create_vector_store
from query import query_rag

# Step 1: Run once
create_vector_store("data/sample.txt")

# Step 2: Ask questions
while True:
    q = input("Ask: ")
    if q.lower() == "exit":
        break

    answer = query_rag(q)
    print("Answer:", answer)