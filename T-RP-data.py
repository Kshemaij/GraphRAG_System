import requests
import xml.etree.ElementTree as ET
import os
import fitz  # PyMuPDF
import sqlite3
from pathlib import Path

# Step 1: Setup DB
def init_db(db_path="papers.db"):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute('''
        CREATE TABLE IF NOT EXISTS papers (
            id TEXT PRIMARY KEY,
            title TEXT,
            summary TEXT,
            published TEXT,
            authors TEXT,
            link TEXT,
            pdf_path TEXT,
            full_text TEXT
        )
    ''')
    conn.commit()
    return conn

# Step 2: Fetch papers from arXiv API
def fetch_arxiv_papers(query="graph neural networks", max_results=5):
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
        published = entry.find('atom:published', ns).text
        authors = ', '.join([author.find('atom:name', ns).text for author in entry.findall('atom:author', ns)])
        link = entry.find('atom:id', ns).text
        papers.append({
            'id': arxiv_id,
            'title': title,
            'summary': summary,
            'published': published,
            'authors': authors,
            'link': link
        })

    return papers

# Step 3: Download PDF
def download_pdf(arxiv_id, save_dir="pdfs"):
    Path(save_dir).mkdir(exist_ok=True)
    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    pdf_path = Path(save_dir) / f"{arxiv_id}.pdf"
    response = requests.get(pdf_url)
    if response.status_code == 200:
        with open(pdf_path, "wb") as f:
            f.write(response.content)
        return str(pdf_path)
    else:
        return None

# Step 4: Extract full text from PDF
def extract_text(pdf_path):
    doc = fitz.open(pdf_path)
    return "\n".join([page.get_text() for page in doc])

# Step 5: Save to DB
def save_to_db(conn, paper):
    cur = conn.cursor()
    cur.execute('''
        INSERT OR REPLACE INTO papers
        (id, title, summary, published, authors, link, pdf_path, full_text)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        paper['id'], paper['title'], paper['summary'], paper['published'],
        paper['authors'], paper['link'], paper['pdf_path'], paper['full_text']
    ))
    conn.commit()

# Run the pipeline
def run(query="transformer neural network", max_results=3):
    conn = init_db()
    papers = fetch_arxiv_papers(query, max_results)

    for paper in papers:
        print(f"Processing {paper['id']} - {paper['title']}")
        pdf_path = download_pdf(paper['id'])
        if pdf_path:
            text = extract_text(pdf_path)
            paper['pdf_path'] = pdf_path
            paper['full_text'] = text
            save_to_db(conn, paper)
            print(f"Saved {paper['id']} to DB.\n")
        else:
            print(f"Failed to download PDF for {paper['id']}\n")

# Start
if __name__ == "__main__":
    run()
