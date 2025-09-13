# import pandas as pd
# from sentence_transformers import SentenceTransformer
# from qdrant_client import QdrantClient
# import psycopg2
# import requests
# from neo4j import GraphDatabase

# # === Configuration ===
# EXCEL_PATH = "C:/Users/91990/Downloads/Masters Thesis/Code/Evaluation.xlsx"  # <- Change this to your actual file
# QDRANT_URL = "https://4cc464a6-1527-4ba9-9b45-c5b64f1eab07.europe-west3-0.gcp.cloud.qdrant.io"
# QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.wuA5by67TJ-ajEB22EUJa0PkpC0kOxsER5mY7omLpcY"
# COLLECTION_NAME = "Master_thesis"

# PG_CONFIG = {
#     "dbname": "thesis",
#     "user": "postgres",
#     "password": "Kshema123",
#     "host": "localhost",
#     "port": "7000"
# }

# NEO4J_URI = "neo4j+s://638eab67.databases.neo4j.io"
# NEO4J_USER = "neo4j"
# NEO4J_PASSWORD = "x3jq4vK4nnr5S3GsqVrqgK5qGaD942poU-TKgL4BEyk"

# OLLAMA_URL = "http://localhost:11434/api/generate"
# OLLAMA_MODEL = "phi3"
# EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# # === Initialize Clients ===
# embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
# qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
# pg_conn = psycopg2.connect(**PG_CONFIG)
# pg_cursor = pg_conn.cursor()
# neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

# def generate_answer_with_ollama(prompt):
#     payload = {"model": OLLAMA_MODEL, "prompt": prompt, "stream": False}
#     try:
#         response = requests.post(OLLAMA_URL, json=payload)
#         if response.status_code == 200:
#             return response.json()["response"].strip()
#         else:
#             return f"⚠ Ollama Error: {response.text}"
#     except Exception as e:
#         return f"⚠ Connection Error: {e}"

# def get_neo4j_context(query_text):
#     with neo4j_driver.session() as session:
#         result = session.run(
#             """
#             MATCH (e1)-[r]->(e2)
#             WHERE e1.name CONTAINS $query
#             RETURN e1.name AS head, type(r) AS relation, e2.name AS tail
#             """,
#             {"query": query_text}
#         )
#         triplets = [f"{row['head']} --[{row['relation']}]--> {row['tail']}" for row in result]
#         return "\n".join(triplets)

# # === Load Questions from Excel ===
# df = pd.read_excel(EXCEL_PATH)
# questions = df["Questions"].tolist()

# # === Process each question ===
# answers = []
# for query in questions:
#     print(f"🔍 Processing: {query}")
#     query_vector = embedder.encode(query).tolist()

#     search_results = qdrant_client.search(
#         collection_name=COLLECTION_NAME,
#         query_vector=query_vector,
#         limit=5
#     )
#     retrieved_ids = [pt.payload.get('postgres_id') for pt in search_results if pt.payload.get('postgres_id')]

#     pg_cursor.execute(
#         "SELECT chunk_text FROM new_chunked WHERE id = ANY(%s);",
#         (retrieved_ids,)
#     )
#     chunks = [row[0] for row in pg_cursor.fetchall()]
#     context_chunks = "\n\n".join(chunks)[:4000]

#     graph_context = get_neo4j_context(query)
#     final_context = f"{context_chunks}\n\nKnowledge Graph:\n{graph_context}"

#     prompt = (
#         "You are an academic assistant. Use ONLY the provided context and graph facts to answer clearly.\n\n"
#         f"Question: {query}\n\n"
#         f"Context:\n{final_context}\n\n"
#         "Answer:"
#     )

#     answer = generate_answer_with_ollama(prompt)
#     answers.append(answer)

# # === Save to Excel ===
# df["GRAG_answers"] = answers
# df.to_excel(EXCEL_PATH, index=False)
# print("✅ Answers saved to Excel.")

import pandas as pd
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
import psycopg2
import requests
from neo4j import GraphDatabase
from sklearn.feature_extraction.text import CountVectorizer

# === Configuration ===
EXCEL_PATH = "C:/Users/91990/Downloads/Masters Thesis/Code/Evaluation.xlsx"  # <- Change this to your actual file
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

NEO4J_URI = "neo4j+s://638eab67.databases.neo4j.io"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "x3jq4vK4nnr5S3GsqVrqgK5qGaD942poU-TKgL4BEyk"

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "phi3"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# === Initialize Clients ===
embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
pg_conn = psycopg2.connect(**PG_CONFIG)
pg_cursor = pg_conn.cursor()
neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

# === Helper Functions ===

def generate_answer_with_ollama(prompt):
    payload = {"model": OLLAMA_MODEL, "prompt": prompt, "stream": False}
    try:
        response = requests.post(OLLAMA_URL, json=payload)
        if response.status_code == 200:
            return response.json()["response"].strip()
        else:
            return f"⚠ Ollama Error: {response.text}"
    except Exception as e:
        return f"⚠ Connection Error: {e}"

def get_neo4j_context(keywords):
    triplets = []
    seen = set()
    with neo4j_driver.session() as session:
        for word in keywords:
            result = session.run(
                """
                MATCH (e1)-[r]->(e2)
                WHERE toLower(e1.name) CONTAINS toLower($word) OR toLower(e2.name) CONTAINS toLower($word)
                RETURN DISTINCT e1.name AS head, type(r) AS relation, e2.name AS tail
                """,
                {"word": word}
            )
            for row in result:
                head = row["head"]
                rel = row["relation"]
                tail = row["tail"]
                triplet = f"{head} --[{rel}]--> {tail}"
                if triplet not in seen:
                    triplets.append(triplet)
                    seen.add(triplet)
    return "\n".join(triplets)

# === Load Questions from Excel ===
df = pd.read_excel(EXCEL_PATH)
questions = df["Questions"].astype(str).tolist()

# === Process each question ===
answers = []

for query in questions:
    print(f"🔍 Processing: {query}")
    query_vector = embedder.encode(query).tolist()

    search_results = qdrant_client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=5
    )

    retrieved_ids = [pt.payload.get('postgres_id') for pt in search_results if pt.payload.get('postgres_id')]

    # Retrieve chunk texts from PostgreSQL
    pg_cursor.execute(
        "SELECT chunk_text FROM new_chunked WHERE id = ANY(%s);",
        (retrieved_ids,)
    )
    chunks = [row[0] for row in pg_cursor.fetchall()]
    context_chunks = "\n\n".join(chunks)[:4000]

    # Extract keywords for graph search
    vectorizer = CountVectorizer(stop_words='english', max_features=5)
    X = vectorizer.fit_transform(chunks)
    keywords = vectorizer.get_feature_names_out().tolist()

    # Get graph context
    graph_context = get_neo4j_context(keywords)

    # Final prompt
    final_context = f"{context_chunks}\n\nKnowledge Graph:\n{graph_context}"
    prompt = (
        "You are an academic assistant. Use ONLY the provided context and graph facts to answer clearly.\n\n"
        f"Question: {query}\n\n"
        f"Context:\n{final_context}\n\n"
        "Answer:"
    )

    answer = generate_answer_with_ollama(prompt)
    answers.append(answer)

# === Save to Excel ===
df["GraphRAG_answers"] = answers
df.to_excel(EXCEL_PATH, index=False)
print("✅ Answers saved to Excel.")
