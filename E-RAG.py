import pandas as pd
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
import psycopg2
import requests

# === Configuration ===
XLSX_PATH = "C:/Users/91990/Downloads/Masters Thesis/Code/Evaluation.xlsx"  # Your Excel file path
QUESTION_COLUMN = "Questions"
ANSWER_COLUMN = "RAG_answer"

QDRANT_URL = "https://4cc464a6-1527-4ba9-9b45-c5b64f1eab07.europe-west3-0.gcp.cloud.qdrant.io"
QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.wuA5by67TJ-ajEB22EUJa0PkpC0kOxsER5mY7omLpcY"
COLLECTION_NAME = "Master_thesis"

PG_CONFIG = {
    "dbname": "thesis",
    "user": "postgres",
    "password": "Kshema123",
    "host": "localhost",
    "port": "7000"
}

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "phi3"

# === Initialize clients ===
embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
pg_conn = psycopg2.connect(**PG_CONFIG)
pg_cursor = pg_conn.cursor()

# === LLM Call ===
def generate_answer(prompt):
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False
    }
    try:
        response = requests.post(OLLAMA_URL, json=payload)
        if response.status_code == 200:
            return response.json()["response"].strip()
        else:
            return f"Ollama error: {response.text}"
    except Exception as e:
        return f"Ollama failed: {e}"

# === Read Excel ===
df = pd.read_excel(XLSX_PATH)

# Limit to first 10 questions
df = df.head(10)

answers = []

for idx, row in df.iterrows():
    query = row[QUESTION_COLUMN]
    print(f"🔍 Processing Q{idx+1}: {query}")

    try:
        # Step 1: Embed query
        query_vector = embedder.encode(query).tolist()

        # Step 2: Qdrant Search
        search_results = qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            limit=5
        )
        retrieved_ids = [point.payload.get('postgres_id') for point in search_results if point.payload.get('postgres_id')]

        # Step 3: Fetch Chunks from PostgreSQL
        if not retrieved_ids:
            answers.append("No matching chunks found")
            continue

        pg_cursor.execute(
            "SELECT chunk_text FROM new_chunked WHERE id = ANY(%s);",
            (retrieved_ids,)
        )
        chunks = [row[0] for row in pg_cursor.fetchall()]
        context = "\n\n".join(chunks)[:4000]

        # Step 4: Create prompt
        prompt = (
            "You are an academic assistant. Use ONLY the provided context to answer clearly.\n\n"
            f"Question: {query}\n\n"
            f"Context:\n{context}\n\n"
            "Answer:"
        )

        # Step 5: LLM
        answer = generate_answer(prompt)

    except Exception as e:
        answer = f"Error: {e}"

    answers.append(answer)

# Save answers to new column
df[ANSWER_COLUMN] = answers

# Save back to Excel
df.to_excel(XLSX_PATH, index=False)
print(f"✅ Saved answers to {XLSX_PATH}")
