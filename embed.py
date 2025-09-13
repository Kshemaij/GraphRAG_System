import psycopg2
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models
from tqdm import tqdm

PG_CONFIG = {
    "dbname": "thesis",
    "user": "postgres",
    "password": "Kshema123",
    "host": "localhost",
    "port": 7000
}

# Qdrant Cloud configuration
QDRANT_URL = "https://4cc464a6-1527-4ba9-9b45-c5b64f1eab07.europe-west3-0.gcp.cloud.qdrant.io"  # replace with your cluster URL
QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.wuA5by67TJ-ajEB22EUJa0PkpC0kOxsER5mY7omLpcY"                # replace with your API key
COLLECTION_NAME = "Master_thesis"
MODEL_NAME = "all-MiniLM-L6-v2"


pg_conn = psycopg2.connect(**PG_CONFIG)
pg_cursor = pg_conn.cursor()

# Fetch data from 100_chunked table
pg_cursor.execute("SELECT id, chunk_text FROM new_chunked;")
rows = pg_cursor.fetchall()
print(f"✅ Fetched {len(rows)} rows from new_chunked table.")

# ========================
# 2️⃣ Generate Embeddings
# ========================
MODEL_NAME = "all-MiniLM-L6-v2"
print("🔄 Generating embeddings...")
model = SentenceTransformer(MODEL_NAME)

texts = [row[1] for row in rows]
embeddings = model.encode(texts, batch_size=32, show_progress_bar=True)

# ========================
# 3️⃣ Connect to Qdrant Cloud
# ========================

qdrant_client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)

vector_size = embeddings.shape[1]

# Create collection if it does not exist
try:
    qdrant_client.get_collection(COLLECTION_NAME)
    print(f"✅ Collection '{COLLECTION_NAME}' already exists in Qdrant Cloud.")
except Exception:
    print(f"🔄 Creating collection '{COLLECTION_NAME}'...")
    qdrant_client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=vector_size,
            distance=models.Distance.COSINE
        )
    )

# ========================
# 4️⃣ Upload Embeddings to Qdrant
# ========================

from qdrant_client.http.models import PointStruct
points = []
for row, embedding in zip(rows, embeddings):
    doc_id = row[0]
    metadata = {"postgres_id": doc_id}  

    points.append(
        PointStruct(
            id=doc_id,
            vector=embedding.tolist(),
            payload=metadata
        )
    )

batch_size = 100
print(f"🔄 Uploading embeddings to Qdrant Cloud in batches of {batch_size}...")
for i in tqdm(range(0, len(points), batch_size), desc="Uploading"):
    qdrant_client.upsert(
        collection_name=COLLECTION_NAME,
        points=points[i:i+batch_size]
    )

print("✅ All embeddings uploaded to Qdrant Cloud successfully.")
