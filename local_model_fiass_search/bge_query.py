import faiss
import numpy as np
import pickle
import sqlite3
import time
from sentence_transformers import SentenceTransformer
start_full_time = time.time()
# Load BGE model (downloads once, cached locally)
model = SentenceTransformer("BAAI/bge-small-en-v1.5")

def embed_query(text: str):
    return model.encode(
        "query: " + text,
        normalize_embeddings=True
    )

# Load FAISS index and item mapping
try:
    index = faiss.read_index("local_model_fiass_search/bge_menu_new1.index")
    with open("local_model_fiass_search/bge_menu_new1_item_ids.pkl", "rb") as f:
        item_ids = pickle.load(f)
except Exception:
    print("❌ Error loading index or item_ids. Run bge_ingest.py first.")
    index = None
    item_ids = []

def find_item(query: str):
    if index is None:
        raise ValueError("Index not loaded")

    q_vec = embed_query(query)

    D, I = index.search(
        np.array([q_vec]).astype("float32"), 1
    )

    index_pos = I[0][0]
    score = D[0][0]   # similarity score (higher is better)

    return {
        "item_id": item_ids[index_pos],
        "score": score
    }

if __name__ == "__main__":
    query = "Chilli Chicken Gravy"

    start_time = time.time()
    result = find_item(query)
    end_time = time.time()

    print("Query:", query)
    print("Matched item_id:", result["item_id"])
    print("Similarity score:", result["score"])
    print("Time taken:", end_time - start_time)

    # Confidence threshold
    if result["score"] > 0.55:
        conn = sqlite3.connect("menu_given_new1.db")
        cursor = conn.cursor()

        cursor.execute(
            "SELECT * FROM menu_items WHERE item_id = ?",
            (result["item_id"],)
        )

        item = cursor.fetchone()
        conn.close()

        print("item_id:", item[0])
        print("text:", item[1])
        print("category:", item[2])
        print("price:", item[3])
        print("type:", item[4])

        end_full_time = time.time()
        print("Full time taken:", end_full_time - start_full_time)
    else:
        print("⚠️ Low confidence match – ask user to repeat")