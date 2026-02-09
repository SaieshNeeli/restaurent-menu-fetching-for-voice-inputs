import sqlite3
import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import ast

# Path to your database
DB_PATH = "production_menu.db"

# Load BGE model (one-time download)
model = SentenceTransformer("BAAI/bge-small-en-v1.5")

def embed_passages(texts):
    embeddings = []
    for text in tqdm(texts):
        emb = model.encode(
            "passage: " + text,
            normalize_embeddings=True
        )
        embeddings.append(emb)
    return embeddings

def fetch_menu_items(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT item_id,text,text_normalized,aliases FROM menu_items")
    rows = cursor.fetchall()
    conn.close()
    return rows

if __name__ == "__main__":
    # Fetch menu items from DB
    menu_rows = fetch_menu_items(DB_PATH)

    texts = []
    item_ids = []

    for row in menu_rows:
        item_id, item, normalized_item, alias = row
        base_text = normalized_item 

        # Convert string representation of list to actual list
        aliases = []
        if alias:
            try:
                aliases = ast.literal_eval(alias)
            except:
                aliases = [alias]

        # Combine normalized item + aliases
        combined_text = " | ".join([base_text.lower().strip()] + aliases)

        texts.append(combined_text)
        item_ids.append(item_id)

    print(f"Generating BGE embeddings for {len(texts)} items...")
    vectors = embed_passages(texts)

    dimension = len(vectors[0])
    print(f"Embedding dimension: {dimension}")

    # Create FAISS index
    index = faiss.IndexFlatIP(dimension)  # Inner Product (BEST for BGE)
    index.add(np.array(vectors).astype("float32"))

    # Save index and item_ids mapping
    faiss.write_index(index, "bge_menu_updated1.index")
    with open("bge_menu_updated_item_ids1.pkl", "wb") as f:
        pickle.dump(item_ids, f)

    print("✅ BGE FAISS index with normalized items + aliases created from DB")
