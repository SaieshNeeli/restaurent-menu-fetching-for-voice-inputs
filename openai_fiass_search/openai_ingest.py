import faiss
import numpy as np
import pickle
from dotenv import load_dotenv
import os
from openai import OpenAI
import time
from tqdm import tqdm
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv()
from menu_items import MENU_ITEMS
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

def get_embeddings_batched(texts, batch_size=20):
    all_embeddings = []
    
    # Process in batches
    i = 0
    pbar = tqdm(total=len(texts))
    
    while i < len(texts):
        batch = texts[i:i + batch_size]
        try:
            # OpenAI embedding call
            response = client.embeddings.create(
                input=batch,
                model="text-embedding-3-small"
            )
            
            # Extract embeddings in correct order
            batch_embeddings = [data.embedding for data in response.data]
            all_embeddings.extend(batch_embeddings)
            
            pbar.update(len(batch))
            i += batch_size
            
            # Minimal sleep to be safe, though OpenAI limits are usually higher
            time.sleep(0.5) 

        except Exception as e:
            print(f"\n⚠️ Error in batch starting at {i}: {e}")
            print("⏳ Waiting 60 seconds before retrying...")
            time.sleep(60)
            # Retry same batch
                
    pbar.close()
    return all_embeddings

if __name__ == "__main__":
    texts = [item["text"] for item in MENU_ITEMS]
    item_ids = [item["item_id"] for item in MENU_ITEMS]

    print(f"Generating OpenAI embeddings for {len(texts)} items...")
    vectors = get_embeddings_batched(texts)

    if not vectors:
        print("❌ No vectors generated.")
        exit(1)

    dimension = len(vectors[0])
    print(f"Embedding dimension: {dimension}")

    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(vectors).astype("float32"))

    faiss.write_index(index, "openai_fiass_search/openai_menu.index")

    with open("openai_fiass_search/openai_item_ids.pkl", "wb") as f:
        pickle.dump(item_ids, f)

    print("✅ OpenAI FAISS index and item_id mapping created")
