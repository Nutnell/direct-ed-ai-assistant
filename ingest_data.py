# ingest_data.py

import os
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ------------------- Setup -------------------
load_dotenv()
print("✅ Environment variables loaded.")

# --- Configuration ---
DATA_SOURCES_PATH = "app/data/gen_ai_track/module_1_llmops_fundamentals"
VECTOR_STORE_PATH = "app/vector_store"

SUPPLEMENTARY_URL = "https://signoz.io/guides/llmops/"
SUPPLEMENTARY_FILE_PATH = os.path.join(
    DATA_SOURCES_PATH, "2_signoz_article.txt"
)

# --- Models & Vector Store ---
print("Loading open-source embedding model 'all-MiniLM-L6-v2'...")
embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vector_store = Chroma(
    persist_directory=VECTOR_STORE_PATH,
    embedding_function=embedding_function
)
print("✅ Setup complete. Database and open-source model are ready.")


# ------------------- Helper Functions -------------------

def scrape_and_save(url: str, file_path: str):
    """Scrape SigNoz LLMOps guide and overwrite local file."""
    print(f"🔍 Scraping content from: {url}")
    try:
        resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        soup = BeautifulSoup(resp.content, "html.parser")

        # look for <article> with class 'prose'
        article = soup.select_one("article.prose")
        if not article:
            print("❌ Unable to locate <article class='prose'> on the page.")
            return

        text = article.get_text(separator="\n", strip=True)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(text)

        size = os.path.getsize(file_path)
        print(f"✅ Saved {size} bytes to {file_path}")

    except Exception as e:
        print(f"❌ Scraping failed: {e}")


def process_and_embed_module(module_path: str):
    """Load or re-scrape .txt sources, split them, and upsert into Chroma."""

    # always re-scrape the SigNoz article to ensure fresh content
    scrape_and_save(SUPPLEMENTARY_URL, SUPPLEMENTARY_FILE_PATH)

    for fname in sorted(os.listdir(module_path)):
        if not fname.endswith(".txt"):
            continue

        path = os.path.join(module_path, fname)
        print(f"\n📄 Processing file: {fname}")

        # load text
        loader = TextLoader(path, encoding="utf-8")
        docs = loader.load()

        # split into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = splitter.split_documents(docs)
        if not chunks:
            print("⚠️  no text chunks found—skipping")
            continue

        # set metadata
        if "1_directed" in fname:
            source_name = "DirectEd Curriculum"
            source_url = "https://directed.example.com/llmops"
        else:
            source_name = "SigNoz Article"
            source_url = SUPPLEMENTARY_URL

        for c in chunks:
            c.metadata.update({
                "source_name": source_name,
                "source_url": source_url,
                "track": "Generative AI",
                "module": "LLMOps Fundamentals"
            })

        # generate unique IDs per chunk
        ids = [f"{path}__{i}" for i in range(len(chunks))]

        # check existing IDs in vector store
        existing = vector_store.get(ids=ids, include=[]).get("ids", [])
        new_items = [
            (cid, chunk)
            for cid, chunk in zip(ids, chunks)
            if cid not in existing
        ]
        if not new_items:
            print("✨ all chunks already ingested")
            continue

        new_ids, new_chunks = zip(*new_items)
        print(f"📦 adding {len(new_ids)} new chunks…")
        vector_store.add_documents(
            documents=list(new_chunks),
            ids=list(new_ids)
        )
        print("✅ chunks added to vector store")


# ------------------- Main Execution -------------------

def main():
    process_and_embed_module(DATA_SOURCES_PATH)
    print("\n🎉 Ingestion complete — knowledge base updated.")


if __name__ == "__main__":
    main()
