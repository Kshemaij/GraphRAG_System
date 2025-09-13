import psycopg2
import pandas as pd
import spacy

# ========== CONFIGURATION ==========

# PostgreSQL credentials
PG_CONFIG = {
    'dbname': 'thesis',
    'user': 'postgres',
    'password': 'Kshema123',
    'host': 'localhost',
    'port': '7000'
}

# Table and column names
TABLE_NAME = "new_chunked"
TEXT_COLUMN = "chunk_text"
CHUNK_ID_COLUMN = "id"

# ========== STEP 1: CONNECT TO POSTGRESQL & FETCH CHUNKS ==========

def fetch_chunks():
    conn = psycopg2.connect(**PG_CONFIG)
    query = f"SELECT {CHUNK_ID_COLUMN}, {TEXT_COLUMN} FROM {TABLE_NAME};"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

# ========== STEP 2: EXTRACT RELATIONSHIP TRIPLETS USING SPACY ==========

def extract_triplets_spacy(text):
    doc = nlp(text)
    triplets = []

    for sent in doc.sents:
        subj = ""
        obj = ""
        verb = ""

        for token in sent:
            if "subj" in token.dep_:
                subj = token.text
            if "obj" in token.dep_:
                obj = token.text
            if token.pos_ == "VERB":
                verb = token.lemma_

        if subj and obj and verb:
            triplets.append((subj, verb, obj))

    return triplets

# ========== STEP 3: PROCESS DATA & EXTRACT ALL TRIPLETS ==========

def process_all_chunks(df):
    results = []

    for _, row in df.iterrows():
        chunk_id = row[CHUNK_ID_COLUMN]
        text = row[TEXT_COLUMN]
        triplets = extract_triplets_spacy(text)

        for (subj, rel, obj) in triplets:
            results.append({
                "chunk_id": chunk_id,
                "subject": subj,
                "relation": rel,
                "object": obj
            })

    return pd.DataFrame(results)

# ========== MAIN EXECUTION ==========

if __name__ == "__main__":
    print("🔄 Loading spaCy model...")
    nlp = spacy.load("en_core_web_sm")

    print("🗃️ Fetching chunked text from PostgreSQL...")
    df_chunks = fetch_chunks()

    print("🔍 Extracting (subject, relation, object) triplets...")
    df_triplets = process_all_chunks(df_chunks)

    print(f"💾 Saving {len(df_triplets)} triplets to triplets.csv...")
    df_triplets.to_csv("triplets.csv", index=False)
    print("✅ Done!")
