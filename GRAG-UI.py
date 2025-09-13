import streamlit as st
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
import psycopg2
import pandas as pd
import requests
from neo4j import GraphDatabase

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

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "phi3"

st.title("📚 GraphRAG Knowledge Retreival System")

qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
pg_conn = psycopg2.connect(**PG_CONFIG)
pg_cursor = pg_conn.cursor()
neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

def generate_answer_with_ollama(prompt):
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
            st.error(f"❌ Ollama Error: {response.text}")
            return "⚠ Error generating answer."
    except Exception as e:
        st.error(f"❌ Ollama Connection Error: {e}")
        return "⚠ Ollama connection failed."

def get_neo4j_context(query_text):
    with neo4j_driver.session() as session:
        # result = session.run(
        #     "MATCH (e1)-[r]->(e2) WHERE e1.name CONTAINS $query RETURN e1.name, r.type, e2.name",
        #     {"query": query}
        # )

        result = session.run(
            """
            MATCH (e1)-[r]->(e2)
            WHERE e1.name CONTAINS $query
            RETURN e1.name AS head, type(r) AS relation, e2.name AS tail
            """,
            {"query": query}
        )

        triplets = [f"{row['Entity1']} --[{row['Relation']}]--> {row['Entity2']}" for row in result]
        return "\n".join(triplets)

query = st.text_input("🔍 Enter your academic query:", "")

if st.button("Get Answer"):
    if query.strip() == "":
        st.warning("⚠️ Please enter a query.")
    else:
        with st.spinner("🔎 Embedding and retrieving top chunks from Qdrant..."):
            query_vector = embedder.encode(query).tolist()

            search_results = qdrant_client.search(
                collection_name=COLLECTION_NAME,
                query_vector=query_vector,
                limit=5
            )

            retrieved_ids = [point.payload.get('postgres_id') for point in search_results if point.payload.get('postgres_id')]

            pg_cursor.execute(
                f"SELECT id, chunk_text, source_id FROM new_chunked WHERE id = ANY(%s);",
                (retrieved_ids,)
            )
            rows = pg_cursor.fetchall()
            df = pd.DataFrame(rows, columns=["Chunk ID", "Chunk Text", "Source ID"])

        st.subheader("🔹 Top Retrieved Chunks:")
        st.dataframe(df[["Chunk ID", "Chunk Text"]])

        graph_context = get_neo4j_context(query)
        if graph_context:
            st.subheader("🧠 Related Triplets from Knowledge Graph:")
            st.text(graph_context)

        with st.spinner("💬 Generating answer using Ollama..."):
            context_chunks = "\n\n".join(df["Chunk Text"].tolist())[:4000]
            final_context = f"{context_chunks}\n\nKnowledge Graph:\n{graph_context}"

            prompt = (
                "You are an academic assistant. Use ONLY the provided context and graph facts to answer clearly.\n\n"
                f"Question: {query}\n\n"
                f"Context:\n{final_context}\n\n"
                "Answer:"
            )

            final_answer = generate_answer_with_ollama(prompt)

        st.subheader("✅ Generated Answer:")
        st.write(final_answer)

        st.subheader("📎 Sources (arXiv Papers):")
        source_ids = df["Source ID"].tolist()
        arxiv_ids = [sid.split('v')[0] if 'v' in sid else sid for sid in source_ids]
        unique_arxiv_ids = list(set(arxiv_ids))

        for arxiv_id in unique_arxiv_ids:
            pdf_link = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
            st.markdown(f"- [{arxiv_id}]({pdf_link})", unsafe_allow_html=True)

# import streamlit as st
# from sentence_transformers import SentenceTransformer
# from qdrant_client import QdrantClient
# import psycopg2
# import pandas as pd
# import requests
# from neo4j import GraphDatabase

# # ------------------- CONFIGURATION -------------------

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

# EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# OLLAMA_URL = "http://localhost:11434/api/generate"
# OLLAMA_MODEL = "phi3"

# # ------------------- INITIALIZE COMPONENTS -------------------

# st.title("📚 GraphRAG Knowledge Retrieval System")

# qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
# embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
# pg_conn = psycopg2.connect(**PG_CONFIG)
# pg_cursor = pg_conn.cursor()
# neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

# # ------------------- HELPER FUNCTIONS -------------------

# def generate_answer_with_ollama(prompt):
#     payload = {
#         "model": OLLAMA_MODEL,
#         "prompt": prompt,
#         "stream": False
#     }
#     try:
#         response = requests.post(OLLAMA_URL, json=payload)
#         if response.status_code == 200:
#             return response.json()["response"].strip()
#         else:
#             st.error(f"❌ Ollama Error: {response.text}")
#             return "⚠ Error generating answer."
#     except Exception as e:
#         st.error(f"❌ Ollama Connection Error: {e}")
#         return "⚠ Ollama connection failed."


# def get_neo4j_context(keywords):
#     triplets = []
#     entities = set()

#     with neo4j_driver.session() as session:
#         for word in keywords:
#             result = session.run(
#                 """
#                 MATCH (e1)-[r]->(e2)
#                 WHERE toLower(e1.name) CONTAINS toLower($word) OR toLower(e2.name) CONTAINS toLower($word)
#                 RETURN DISTINCT e1.name AS head, type(r) AS relation, e2.name AS tail
#                 """,
#                 {"word": word}
#             )

#             for row in result:
#                 head = row['head']
#                 relation = row['relation']
#                 tail = row['tail']
#                 triplet_str = f"{head} --[{relation}]--> {tail}"
#                 if triplet_str not in triplets:
#                     triplets.append(triplet_str)
#                     entities.update([head, tail])

#     return triplets, list(entities)

# # ------------------- UI INPUT -------------------

# query = st.text_input("🔍 Enter your academic query:", "")

# if st.button("Get Answer"):
#     if query.strip() == "":
#         st.warning("⚠️ Please enter a query.")
#     else:
#         # --- Qdrant Search ---
#         with st.spinner("🔎 Embedding and retrieving top chunks from Qdrant..."):
#             query_vector = embedder.encode(query).tolist()

#             search_results = qdrant_client.search(
#                 collection_name=COLLECTION_NAME,
#                 query_vector=query_vector,
#                 limit=5
#             )
#             retrieved_ids = [point.payload.get('postgres_id') for point in search_results if point.payload.get('postgres_id')]
#             pg_cursor.execute(
#                 f"SELECT id, chunk_text, source_id FROM new_chunked WHERE id = ANY(%s);",
#                 (retrieved_ids,)
#             )
#             rows = pg_cursor.fetchall()
#             df = pd.DataFrame(rows, columns=["Chunk ID", "Chunk Text", "Source ID"])
#         st.subheader("🔹 Top Retrieved Chunks:")
#         st.dataframe(df[["Chunk ID", "Chunk Text"]])

#         # --- Neo4j Triplet Retrieval ---
#         triplets, entities = get_neo4j_context(query)

#         if triplets:
#             st.subheader(" Related Triplets from Knowledge Graph:")

#             # Split triplets into components
#             triplet_data = []
#             for triplet in triplets:
#                 if '--[' in triplet and ']-->' in triplet:
#                     head = triplet.split(' --[')[0]
#                     relation = triplet.split(' --[')[1].split(']--> ')[0]
#                     tail = triplet.split(']--> ')[1]
#                     triplet_data.append((head, relation, tail))
    
#             triplet_df = pd.DataFrame(triplet_data, columns=["Head", "Relation", "Tail"])
#             st.dataframe(triplet_df)
        

#         # if entities:
#         #     st.subheader("🧩 Extracted Entities:")
#         #     entity_df = pd.DataFrame(entities, columns=["Entity"])
#         #     st.dataframe(entity_df)

#         # --- Answer Generation ---
#         with st.spinner(" Generating answer using Ollama..."):
#             context_chunks = "\n\n".join(df["Chunk Text"].tolist())[:4000]
#             graph_context = "\n".join(triplets)
#             final_context = f"{context_chunks}\n\nKnowledge Graph:\n{graph_context}"

#             # prompt = (
#             #     '''You are an academic assistant. Use ONLY the provided context and graph facts to answer the query clearly.
#             #     Do not include the triplets in the final answer but use them to provide the answer. \n\n'''
#             #     f"Question: {query}\n\n"
#             #     f"Context:\n{final_context}\n\n"
#             #     "Answer:"
#             # )

#             prompt = (
#                 "You are an academic assistant. Use ONLY the provided context and graph facts to answer clearly.\n\n"
#                 f"Question: {query}\n\n"
#                 f"Context:\n{final_context}\n\n"
#                 "Answer:"
#             )



#             final_answer = generate_answer_with_ollama(prompt)

#         st.subheader(" Generated Answer:")
#         st.write(final_answer)

#         # --- Sources ---
#         st.subheader("📎 Sources (arXiv Papers):")
#         source_ids = df["Source ID"].tolist()
#         arxiv_ids = [sid.split('v')[0] if 'v' in sid else sid for sid in source_ids]
#         unique_arxiv_ids = list(set(arxiv_ids))

#         for arxiv_id in unique_arxiv_ids:
#             pdf_link = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
#             st.markdown(f"- [{arxiv_id}]({pdf_link})", unsafe_allow_html=True)
