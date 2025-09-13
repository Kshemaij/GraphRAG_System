# import psycopg2
# from psycopg2.extras import execute_values

# def chunk_text(text, chunk_size=500, overlap=100):
#     words = text.split()
#     chunks = []
#     i = 0
#     while i < len(words):
#         chunk = words[i:i + chunk_size]
#         chunks.append(" ".join(chunk))
#         i += chunk_size - overlap
#     return chunks

# # --- PostgreSQL DB Config ---
# DB_CONFIG = {
#     'dbname': 'thesis',
#     'user': 'postgres',
#     'password': 'Kshema123',
#     'host': 'localhost',
#     'port': 7000
# }

# def fetch_and_chunk_100_direct():
#     conn = psycopg2.connect(**DB_CONFIG)
#     cur = conn.cursor()

#     # Step 1: Drop and create 100_chunked table
#     cur.execute("""
#         DROP TABLE IF EXISTS "new_chunked";
#         CREATE TABLE "new_chunked" (
#             id SERIAL PRIMARY KEY,
#             source_id TEXT,
#             chunk_index INTEGER,
#             chunk_text TEXT
#         );
#     """)
#     print("✅ Table '100_chunked' created.")

#     # Step 2: Fetch first 100 rows from papers
#     cur.execute("""
#         SELECT id, full_text FROM papers
#         WHERE full_text IS NOT NULL
#         LIMIT 20;
#     """)
#     rows = cur.fetchall()

#     # Step 3: Chunk and prepare data
#     all_chunks = []
#     for row in rows:
#         source_id, full_text = row
#         chunks = chunk_text(full_text)
#         for idx, chunk in enumerate(chunks):
#             all_chunks.append((source_id, idx, chunk))

#     # Step 4: Insert chunks into 100_chunked
#     insert_query = """
#         INSERT INTO "new_chunked" (source_id, chunk_index, chunk_text)
#         VALUES %s
#     """
#     execute_values(cur, insert_query, all_chunks)

#     conn.commit()
#     cur.close()
#     conn.close()
#     print(f"✅ Inserted {len(all_chunks)} chunks into 'new_chunked' successfully.")

# if __name__ == "__main__":
#     fetch_and_chunk_100_direct()













import psycopg2
from psycopg2.extras import execute_values

def chunk_text(text, chunk_size=500, overlap=100):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i + chunk_size]
        chunks.append(" ".join(chunk))
        i += chunk_size - overlap
    return chunks

# --- PostgreSQL DB Config ---
DB_CONFIG = {
    'dbname': 'thesis',
    'user': 'postgres',
    'password': 'Kshema123',
    'host': 'localhost',
    'port': 7000
}

def fetch_and_chunk_all():
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    # Step 1: Drop and create new_chunked table
    cur.execute("""
        DROP TABLE IF EXISTS "new_chunked";
        CREATE TABLE "new_chunked" (
            id SERIAL PRIMARY KEY,
            source_id TEXT,
            chunk_index INTEGER,
            chunk_text TEXT
        );
    """)
    print("✅ Table 'new_chunked' created.")

    # Step 2: Fetch ALL rows from papers where full_text is not null
    cur.execute("""
        SELECT id, full_text 
        FROM papers
        WHERE full_text IS NOT NULL;
    """)
    rows = cur.fetchall()
    print(f"📄 Retrieved {len(rows)} papers to chunk.")

    # Step 3: Chunk and prepare data
    all_chunks = []
    for row in rows:
        source_id, full_text = row
        chunks = chunk_text(full_text)
        for idx, chunk in enumerate(chunks):
            all_chunks.append((source_id, idx, chunk))

    # Step 4: Insert chunks into new_chunked
    insert_query = """
        INSERT INTO "new_chunked" (source_id, chunk_index, chunk_text)
        VALUES %s
    """
    execute_values(cur, insert_query, all_chunks)

    conn.commit()
    cur.close()
    conn.close()
    print(f"✅ Inserted {len(all_chunks)} chunks into 'new_chunked' successfully.")

if __name__ == "__main__":
    fetch_and_chunk_all()
