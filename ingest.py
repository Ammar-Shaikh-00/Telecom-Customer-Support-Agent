from pathlib import Path
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

DATA_DIR = Path("data/faqs")
CHROMA_DIR = "chroma_db"

def main():
    # Use proper persistent client depending on version
    try:
        client = chromadb.PersistentClient(path=CHROMA_DIR)
    except AttributeError:
        client = chromadb.Client(Settings(persist_directory=CHROMA_DIR))

    collection = client.get_or_create_collection("support_faqs")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    docs, ids, metas = [], [], []
    for i, p in enumerate(sorted(DATA_DIR.glob("*.txt"))):
        text = p.read_text(encoding="utf-8").strip()
        if not text:
            continue
        docs.append(text)
        ids.append(f"doc-{i}")
        metas.append({"source": p.name})

    if not docs:
        print("❌ No .txt files found in data/faqs.")
        return

    embs = model.encode(docs, show_progress_bar=True).tolist()
    collection.upsert(ids=ids, documents=docs, metadatas=metas, embeddings=embs)

    # Safer way to check collection size
    try:
        print(f"✅ Done. Collection size: {collection.count()}")
    except Exception:
        print(f"✅ Done. Inserted {len(ids)} documents.")

if __name__ == "__main__":
    main()
