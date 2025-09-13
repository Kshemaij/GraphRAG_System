import psycopg2
import requests
import xml.etree.ElementTree as ET
import fitz  # PyMuPDF
from io import BytesIO
import re
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

def is_probable_table(text):
    lines = text.split("\n")
    if len(lines) >= 2:
        numeric_lines = sum(1 for l in lines if re.search(r'\d', l))
        if numeric_lines / len(lines) > 0.6:
            return True
    if text.count('|') > 3 or text.count('\t') > 3:
        return True
    return False

def clean_pdf_text(doc):
    full_text = []
    header_footer_tracker = set()
    for page in doc:
        blocks = page.get_text("dict")["blocks"]
        page_text = []
        for b in blocks:
            if b["type"] != 0:
                continue  
            block_text = " ".join([line["spans"][0]["text"] for line in b["lines"] if line["spans"]])
            block_text = block_text.strip()
            if not block_text or len(block_text.split()) < 3:
                continue
            if re.match(r'^\d+$', block_text.strip()):
                continue  
            if block_text.lower() in header_footer_tracker:
                continue  
            if is_probable_table(block_text):
                continue
            header_footer_tracker.add(block_text.lower())
            page_text.append(block_text)
        if page_text:
            full_text.append("\n".join(page_text))
    return "\n\n".join(full_text)

def extract_text_from_online_pdf(arxiv_id):
    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    response = requests.get(pdf_url)
    if response.status_code == 200:
        try:
            with fitz.open(stream=BytesIO(response.content), filetype="pdf") as doc:
                cleaned = clean_pdf_text(doc)
                return cleaned
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
                        paper['full_text'] = text.replace('\x00', '')
                        save_to_db(conn, paper)
                        print(f"✅ Saved {paper['id']} to DB.")
                    else:
                        print(f"❌ Skipped {paper['id']} (no text).")
    except Exception as e:
        print("Database error:", e)



if __name__ == "__main__":
    run_pipeline(max_results=100)  