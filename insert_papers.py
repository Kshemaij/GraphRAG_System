import psycopg2
import requests
import xml.etree.ElementTree as ET
import fitz  # PyMuPDF
from io import BytesIO
from config import load_config

def fetch_arxiv_papers(query, max_results=1):
    base_url = "http://export.arxiv.org/api/query?"
    search_query = f"search_query=all:{query}&start=0&max_results={max_results}"
    response = requests.get(base_url + search_query)
    root = ET.fromstring(response.text)
    ns = {'atom': 'http://www.w3.org/2005/Atom'}
    papers = []

    for entry in root.findall('atom:entry', ns):
        arxiv_id = entry.find('atom:id', ns).text.split('/')[-1]
        title = entry.find('atom:title', ns).text.strip()
        summary = entry.find('atom:summary', ns).text.strip()
        authors = ', '.join([author.find('atom:name', ns).text for author in entry.findall('atom:author', ns)])
        link = entry.find('atom:id', ns).text
        papers.append({
            'id': arxiv_id,
            'title': title,
            'summary': summary,
            'authors': authors,
            'link': link
        })

    return papers

def extract_text_from_online_pdf(arxiv_id):
    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    response = requests.get(pdf_url)
    if response.status_code == 200:
        try:
            with fitz.open(stream=BytesIO(response.content), filetype="pdf") as doc:
                return "\n".join([page.get_text() for page in doc])
        except Exception as e:
            print(f"Error reading PDF {arxiv_id}: {e}")
            return None
    else:
        print(f"Failed to fetch PDF for {arxiv_id}")
        return None

def save_to_db(conn, paper):
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO papers (id, title, summary, authors, link, full_text)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT (id) DO NOTHING
    """, (
        paper['id'],
        paper['title'],
        paper['summary'],
        paper['authors'],
        paper['link'],
        paper['full_text']
    ))
    conn.commit()

def run_pipeline(max_results=1):
    config = load_config()
    try:
        with psycopg2.connect(**config) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT name FROM topics")
                topics = [row[0] for row in cur.fetchall()]

            for topic in topics:
                print(f"\n Searching for topic: {topic}")
                papers = fetch_arxiv_papers(topic, max_results)
                for paper in papers:
                    print(f" Processing {paper['id']} - {paper['title'][:60]}...")
                    text = extract_text_from_online_pdf(paper['id'])
                    if text:
                        paper['full_text'] = text.replace('\x00', '') if text else ''
                        save_to_db(conn, paper)
                        print(f"✅ Saved {paper['id']} to DB.")
                    else:
                        print(f"❌ Skipped {paper['id']} (no text).")
    except Exception as e:
        print("Database error:", e)

if __name__ == "__main__":
    run_pipeline(max_results=100)  
