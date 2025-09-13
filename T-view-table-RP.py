import sqlite3

def view_all_papers(db_path="papers.db"):
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Get column names
        cursor.execute("PRAGMA table_info(papers)")
        columns = [col[1] for col in cursor.fetchall()]

        # Fetch all rows
        cursor.execute("SELECT * FROM papers")
        rows = cursor.fetchall()

        print(f"\n📄 Found {len(rows)} papers in '{db_path}':\n")

        for i, row in enumerate(rows, start=1):
            print(f"\n--- Paper {i} ---")
            for col_name, value in zip(columns, row):
                if col_name == "full_text":
                    print(f"{col_name}: {value[:500]}... [truncated]")
                else:
                    print(f"{col_name}: {value}")
        
        conn.close()

    except sqlite3.Error as e:
        print("Error reading database:", e)

if __name__ == "__main__":
    view_all_papers()
