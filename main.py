from sentence_transformers import SentenceTransformer
import chromadb
from openai import OpenAI


# =========================
# 1. CONNECT TO LOACAL LLM
# =========================



client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)




# =========================
# 2. LOAD DOCUMENT
# =========================

with open("data/document.txt", "r", encoding="utf-8") as file:
    text = file.read()


# =========================
# 3. CHUNKING
# =========================

chunks = text.split("\n\n")

print("Number of chunks:", len(chunks))


# =========================
# 4. CREATE EMBEDDINGS
# =========================

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

embeddings = embedding_model.encode(chunks)


# =========================
# 5. STORE IN CHROMADB
# =========================

chroma_client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = chroma_client.get_or_create_collection(
    name="rag_documents"
)

collection.add(
    documents=chunks,
    embeddings=embeddings.tolist(),
    ids=[str(i) for i in range(len(chunks))]
)


# =========================
# 6. USER QUESTION
# =========================

question = input("\nAsk a question: ")


# =========================
# 7. EMBED QUESTION
# =========================

question_embedding = embedding_model.encode([question])[0]


# =========================
# 8. RETRIEVE
# =========================

results = collection.query(
    query_embeddings=[question_embedding.tolist()],
    n_results=2
)

retrieved_chunks = results["documents"][0]


print("\nRetrieved information:")
for chunk in retrieved_chunks:
    print("-", chunk)


# =========================
# 9. CREATE PROMPT
# =========================

context = "\n\n".join(retrieved_chunks)

prompt = f"""
Answer the question using only the information provided below.

Context:
{context}

Question:
{question}
"""


# =========================
# 10. CALL local llm
# =========================

response = client.chat.completions.create(
    model="qwen3:8b",
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ],
    temperature=0.2,
    max_tokens=500
)


# =========================
# 11. PRINT ANSWER
# =========================

answer = response.choices[0].message.content

print("\nAnswer:")
print(answer)