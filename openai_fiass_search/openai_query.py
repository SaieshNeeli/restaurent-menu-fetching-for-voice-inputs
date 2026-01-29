import faiss
import numpy as np
import pickle
from dotenv import load_dotenv
import os
from openai import OpenAI
import sqlite3
import time 
load_dotenv()
start_full_time = time.time()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

def embed(text: str):
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

# Load index and mapping
try:
    index = faiss.read_index("openai_fiass_search/openai_menu.index")
    with open("openai_fiass_search/openai_item_ids.pkl", "rb") as f:
        item_ids = pickle.load(f)
except Exception as e:
    print("Error loading index or item_ids. Make sure to run openai_ingest.py first.")
    # Initialize empty to prevent crash on import, though script will fail on run
    index = None
    item_ids = []

def find_item(query: str):
    if index is None:
        raise ValueError("Index not loaded")
        
    q_vec = embed(query)

    D, I = index.search(
        np.array([q_vec]).astype("float32"), 1
    )

    index_pos = I[0][0]
    distance = D[0][0]

    return {
        "item_id": item_ids[index_pos],
        "score": distance
    }


if __name__ == "__main__":
    query = "prawn fry"
    start_time = time.time()
    try:
        result = find_item(query)
        print("Query:", query)
        print("Matched item_id:", result["item_id"])
        print("Distance score:", result["score"])
        end_time = time.time()
        print("Time taken by open_ai:", end_time - start_time)
        
        if result["score"] < 0.7:
            conn = sqlite3.connect("menu_given.db")
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM menu_items WHERE item_id = ?", (result["item_id"],))
            item = cursor.fetchone()
            
            if item:
                print("item_id:", item[0])
                print("text:", item[1])
                print("category:", item[2])
                print("price:", item[3])
                print("type:", item[4])
            else:
                print(f"No item found with item_id: {result['item_id']}")
            
            end_full_time = time.time()
            print("Full time taken:", end_full_time - start_full_time)
            conn.close()
        else:
            print("Distance score is too high for reliable match")
    except Exception as e:
        print(f"Error: {e}")

