import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

import sys
import os
# Add parent directory to path so we can import menu_items
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# from menu_items import MENU_ITEMS
from menu_items import MENU_ITEMS

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

if __name__ == "__main__":
    texts = [item["text"].lower().strip() for item in MENU_ITEMS]
    item_ids = [item["item_id"] for item in MENU_ITEMS]

    print(f"Generating BGE embeddings for {len(texts)} items...")
    vectors = embed_passages(texts)

    dimension = len(vectors[0])
    print(f"Embedding dimension: {dimension}")

    index = faiss.IndexFlatIP(dimension)  # Inner Product (BEST for BGE)
    index.add(np.array(vectors).astype("float32"))

    index_path = os.path.join(current_dir, "bge_menu_new1.index")
    faiss.write_index(index, index_path)

    ids_path = os.path.join(current_dir, "bge_menu_new1_item_ids.pkl")
    with open(ids_path, "wb") as f:
        pickle.dump(item_ids, f)

    print("✅ BGE FAISS index and item_id mapping created")
