import psycopg2
from config import load_config

def create_tables():
    """Create papers and topics tables in PostgreSQL"""
    commands = (
        """
        CREATE TABLE IF NOT EXISTS papers (
            id TEXT PRIMARY KEY,
            title TEXT,
            summary TEXT,
            authors TEXT,
            link TEXT,
            pdf_path TEXT,
            full_text TEXT
        )
        """,
        
        """
        CREATE TABLE IF NOT EXISTS chunked_docs (
            id SERIAL PRIMARY KEY,
            source_id TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            chunk_text TEXT NOT NULL
        )
        """
        
    )
    try:
        config = load_config()
        with psycopg2.connect(**config) as conn:
            with conn.cursor() as cur:
                for command in commands:
                    cur.execute(command)
        print("Tables created successfully.")
    except (psycopg2.DatabaseError, Exception) as error:
        print(error)

if __name__ == '__main__':
    create_tables()