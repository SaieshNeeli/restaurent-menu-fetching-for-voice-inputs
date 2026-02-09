import os
import time
import pickle
import sqlite3
import numpy as np
import faiss
import re
from fastapi import APIRouter, HTTPException,FastAPI
from pydantic import BaseModel
from metaphone import doublemetaphone
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import process, fuzz
from collections import defaultdict
import json
import sys
router = APIRouter(prefix="/search", tags=["RAG Search"])

# BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# print("base_dir",BASE_DIR)
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
DB_PATH = os.path.join( "menu_updated.db")
INDEX_PATH = os.path.join( "bge_menu_updated1.index")
PKL_PATH = os.path.join( "bge_menu_updated_item_ids1.pkl")



app = FastAPI(title="Vector Search API")

print(f"Database Path: {DB_PATH}")

print("Loading BGE model...")
bge_model = SentenceTransformer("BAAI/bge-small-en-v1.5")
print("BGE model loaded")

# Initialize variables to avoid NameError
bge_index = None
bge_item_ids = []

# Load BGE Index
try:
    if os.path.exists(INDEX_PATH) and os.path.exists(PKL_PATH):
        bge_index = faiss.read_index(INDEX_PATH)
        with open(PKL_PATH, "rb") as f:
            bge_item_ids = pickle.load(f)
        print("BGE Index loaded")
    else:
        print(f"Index or ID file not found")
except Exception as e:
    print(f"Error loading BGE index: {e}")

# FastAPI request model
class SearchQuery(BaseModel):
    query: str
    top_k: int = 3  # number of results to return

class CategoryQuery(BaseModel):
    query: str
    top_k: int = 200  # number of results to return

def get_db_details(item_id: str):
    if not os.path.exists(DB_PATH):
        print(f"Database not found at {DB_PATH}")
        return None
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM menu_items WHERE item_id = ?", (item_id,))
        item = cursor.fetchone()
        conn.close()
        return item
    except Exception as e:
        print(f"Database error: {e}")
        return None

def get_db_details_by_name(name: str):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM menu_items WHERE text = ?", (name,))
    item = cursor.fetchone()
    conn.close()
    if item:
        return item
    return None

print("Building phonetic index...")

menu_item_names = []

if bge_item_ids:
    for item_id in bge_item_ids:

        item = get_db_details(item_id)
        if not item:
            continue

        # -------- text_normalized --------
        if item[2]:
            menu_item_names.append(item[2])

        # -------- aliases from DB --------
        db_aliases = item[11]  # aliases column

        if db_aliases:
            if isinstance(db_aliases, str):
                for alias in db_aliases.split(","):
                    alias = alias.strip().lower()
                    if alias:
                        print("alias for insert", alias)
                        menu_item_names.append(alias)


# Remove duplicates
menu_item_names = list(set(menu_item_names))

print(f"Total vocab names: {len(menu_item_names)}")


menu_vocab = set()
VALID_BIGRAMS = set()

for name in menu_item_names:
    name = name.lower()
    words = re.findall(r"[a-z]+|[0-9]+", name)


    for word in words:
        menu_vocab.add(word)

    for i in range(len(words) - 1):
        VALID_BIGRAMS.add((words[i], words[i+1]))

menu_vocab = list(menu_vocab)

menu_vocab1 = set()

print("\n\nmenu_vocab\n\n", menu_vocab)
print("\n\nVALID_BIGRAMS\n\n", VALID_BIGRAMS)
PHONETIC_INDEX = defaultdict(set)

for word in menu_vocab:
    p, s = doublemetaphone(word)
    if p:
        PHONETIC_INDEX[p].add(word)
    if s:
        PHONETIC_INDEX[s].add(word)
print("\n\nPHONETIC_INDEX\n\n", PHONETIC_INDEX)


def get_phonetic_candidates(
    word: str,
    phonetic_index: dict,
    metaphone_threshold=70,
    word_limit=5
):
    primary, secondary = doublemetaphone(word)

    if not primary and not secondary:
        return []

    all_keys = list(phonetic_index.keys())
    candidate_words = set()

    # 🔹 Fuzzy match PRIMARY metaphone
    if primary:
        key_matches = process.extract(
            primary,
            all_keys,
            scorer=fuzz.ratio,
            score_cutoff=metaphone_threshold
        )
        for key, _, _ in key_matches:
            candidate_words.update(phonetic_index[key])

    # 🔹 Fuzzy match SECONDARY metaphone
    if secondary:
        key_matches = process.extract(
            secondary,
            all_keys,
            scorer=fuzz.ratio,
            score_cutoff=metaphone_threshold
        )
        for key, _, _ in key_matches:
            candidate_words.update(phonetic_index[key])

    if not candidate_words:
        return []

    # 🔹 Final fuzzy ranking on actual words
    matches = process.extract(
        word,
        list(candidate_words),
        scorer=fuzz.ratio,
        limit=word_limit
    )

    return matches

def choose_best_word(original, candidates, next_word_candidates, valid_bigrams):
    if not candidates:
        return original, 0.0

    best_bigram_match = None
    best_score = 0.0

    if next_word_candidates:
        print(f"  [Context Check] '{original}' candidates: {[c[0] for c in candidates]}")
        print(f"  [Context Check] Next word candidates: {[c[0] for c in next_word_candidates]}")
        for cand, score, _ in candidates:
            for next_cand, _, _ in next_word_candidates:
                if (cand, next_cand) in valid_bigrams:
                    confidence = score / 100.0
                    if confidence > best_score:
                        print(f"  [Context Match] Potential bigram found: ({cand}, {next_cand}) (Score: {score})")
                        best_bigram_match = cand
                        best_score = confidence

        if best_bigram_match:
            print(f"  [Context Match] Final selection from bigram: {best_bigram_match}")
            return best_bigram_match, best_score

    # Fallback to best phonetic match
    print("\n candidates for fallback (choose best word candidates) \n", candidates)
    best_match = candidates[0]
    print(f"  [Phonetic Fallback] No bigram match for '{original}'. Selecting best match: {best_match[0]}")
    return best_match[0], best_match[1] / 100.0


def correct_query(query: str, valid_bigrams: set):
    words = re.findall(r"[a-z]+|[0-9]+", query.lower())
    # words = query.lower().split()
    corrected_words = []
    confidences = []

    # Get all candidates first to allow look-ahead
    all_word_candidates = [get_phonetic_candidates(w, PHONETIC_INDEX) for w in words]
    print("\n all_word_candidates \n", all_word_candidates)
    for i in range(len(words)):
        original = words[i]
        candidates = all_word_candidates[i]
        
        # Look ahead for next word candidates
        next_word_candidates = all_word_candidates[i+1] if i + 1 < len(words) else None
        
        print(f"Correcting word [{i}]: '{original}'")
        corrected, conf = choose_best_word(original, candidates, next_word_candidates, valid_bigrams)
        corrected_words.append(corrected)
        confidences.append(conf)

    avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
    final_corrected = " ".join(corrected_words)
    print(f"Final Corrected Query: '{final_corrected}' (Confidence: {avg_confidence:.2f})")
    return {
        "original": query,
        "corrected": " ".join(corrected_words),
        "confidence": float(round(avg_confidence, 2))
    }



def should_use_correction(original, corrected, confidence):
    # If identical, always accept
    if original == corrected:
        return True

    # Confidence gate
    if confidence < 0.6:
        return False

    # Lexical similarity gate
    fuzz_score = fuzz.ratio(original, corrected)
    if fuzz_score < 65:
        return False

    return True
def fetch_cat():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("select cat,cat_alias1 from menu_items ")
    rows = cursor.fetchall()
    full_cat = set()

    for row in rows:
        full_cat.add(row[0])

        aliases = json.loads(row[1])   # ✅ convert string → list

        for alias in aliases:
            full_cat.add(alias)
    # print(full_cat)

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()
cursor.execute("select cat,cat_alias1 from menu_items ")
rows = cursor.fetchall()
full_cat = set()

for row in rows:
    full_cat.add(row[0])

    aliases = json.loads(row[1])   # ✅ convert string → list

    for alias in aliases:
        full_cat.add(alias)
print(full_cat)
# full_cat = fetch_cat()

cat_vocab = set()
cat_bigrams = set()
for name in full_cat:
    cat_vocab.update(name.split())
    words = name.split()
    for i in range(len(words) - 1):
        cat_bigrams.add((words[i], words[i+1]))
print("\n\ncat_bigrams\n\n", cat_bigrams)
# print(cat_vocab)
from collections import defaultdict
from metaphone import doublemetaphone
PHONETIC_INDEX_cat = defaultdict(set)

for word in cat_vocab:
    p, s = doublemetaphone(word)
    if p:
        PHONETIC_INDEX_cat[p].add(word)
    if s:
        PHONETIC_INDEX_cat[s].add(word)
# print("\n\nPHONETIC_INDEX\n\n", PHONETIC_INDEX)


MENU_CATEGORIES = [
    'DOSA', 'UTTAPAM', 'BREADS', 'KEBABS', 'INDO-CHINESE (VEG)',
    'INDO-CHINESE (NON-VEG / EGG)', 'VEG STARTERS', 'BIRYANI',
    'VEG CURRIES', 'CHICKEN CURRIES', 'GOAT CURRIES', 'SEAFOOD CURRIES',
    'DESSERTS', 'DRINKS (COLD)', 'DRINKS (HOT)', 'FRY APPETIZERS (NON-VEG)',
    'SEAFOOD FRY', 'SHRIMP & CRAB', 'EGG DISHES'
]

def get_db_details_by_tokens1(query: str):
    print("\nquery\n", query)

    tokens = query.lower().split()

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    conditions = []
    params = []

    for token in tokens:
        # Each token must match either text_normalized OR aliases
        conditions.append("(text_normalized LIKE ? OR aliases LIKE ?)")
        params.extend([f"%{token}%", f"%{token}%"])

    sql = f"""
        SELECT *
        FROM menu_items
        WHERE {" AND ".join(conditions)}
    """

    cursor.execute(sql, params)

    items = cursor.fetchall()
    conn.close()

    print("\nitems\n", items)
    return items


# ==================== ENDPOINTS ====================

@router.post("/correct_query_multi")
async def correct_query_multi_endpoint(search: SearchQuery):
    start_time = time.time()

    if bge_index is None:
        raise HTTPException(
            status_code=500,
            detail="Vector index not loaded"
        )

    raw_queries = [
        q.strip().lower()
        for q in search.query.split(",")
        if q.strip()
    ]

    final_results = {}
    top_match = min(search.top_k, bge_index.ntotal)

    for q in raw_queries:

        # -------------------------------------------------
        # STEP 1: PHONETIC + FUZZY QUERY CORRECTION
        # -------------------------------------------------
        corrected_info = correct_query(q, VALID_BIGRAMS)
        corrected_q = corrected_info["corrected"]

        if not should_use_correction(
            q, corrected_q, corrected_info["confidence"]
        ):
            corrected_q = q

        if not corrected_q:
            continue

        # -------------------------------------------------
        # STEP 2: VEG INTENT DETECTION
        # -------------------------------------------------
        is_veg = (
            re.search(r"\bveg\b", q)
            and not re.search(r"\bnon[\s-]?veg\b", q)
        )

        query_matches = []

        # -------------------------------------------------
        # STEP 3: DIRECT DB TOKEN MATCH (FAST EXIT)
        # -------------------------------------------------
        matched_items = get_db_details_by_tokens1(corrected_q)
        print("matched_items", matched_items)
        if matched_items and len(matched_items) == 1:
            item = matched_items[0]
            print("item", item)

            if not (is_veg and item[9] != 1):
                query_matches.append({
                    "item_id": item[0],
                    "item_name": item[1],
                    "category": item[6],
                    "price": item[7],
                    "type": item[8],
                    "description": item[12],
                    "score": 1.0,
                    "match_type": "exact"
                })

                final_results[q] = {
                    "original_query": q,
                    "corrected_query": corrected_q,
                    "confidence": float(corrected_info["confidence"]),
                    "description": "Exact DB match",
                    "matches": query_matches
                }
                continue

        # -------------------------------------------------
        # STEP 4: BGE SEMANTIC SEARCH (CANDIDATE GENERATION)
        # -------------------------------------------------
        q_vec = bge_model.encode(
            "query: " + corrected_q,
            normalize_embeddings=True
        )
        q_vec = np.array([q_vec], dtype="float32")
        faiss.normalize_L2(q_vec)

        candidate_k = min(top_match * 10, bge_index.ntotal)
        D, I = bge_index.search(q_vec, candidate_k)

        # -------------------------------------------------
        # STEP 5: FUZZY + ALIAS RE-RANKING
        # -------------------------------------------------
        for score, idx in zip(D[0], I[0]):
            item_id = bge_item_ids[int(idx)]
            item = get_db_details(item_id)
            if not item:
                continue

            # Veg enforcement
            if is_veg and item[9] != 1:
                continue

            texts_to_match = [item[2]]  # text_normalized
            if item[11]:
                texts_to_match.extend(
                    a.strip() for a in item[11].split(",")
                )

            best_fuzz = max(
                fuzz.token_set_ratio(corrected_q, t)
                for t in texts_to_match
            )

            if best_fuzz == 100 and len(corrected_q.split()) == len(item[2].split()):
                print("best_fuzz query", corrected_q,item[2])
                query_matches = [{
                    "item_id": item[0],
                    "item_name": item[1],
                    "category": item[6],
                    "price": item[7],
                    "type": item[8],
                    "description": item[12],
                    "score": 1.0,
                    "fuzz_score": best_fuzz,
                    "match_type": "exact"
                }]
                break

            # Drop weak candidates
            if best_fuzz < 65 or score < 0.55:
                continue

            final_score = round(
                (0.6 * best_fuzz / 100) + (0.4 * score),
                4
            )

            query_matches.append({
                "item_id": item[0],
                "item_name": item[1],
                "category": item[6],
                "price": item[7],
                "type": item[8],
                "description": item[12],
                "score": float(final_score),
                "bge_score": float(score),
                "fuzz_score": float(best_fuzz),
                "match_type": "semantic"
            })

            if len(query_matches) >= top_match:
                break

        # -------------------------------------------------
        # STEP 6: FINALIZE RESULT
        # -------------------------------------------------
        query_matches = sorted(
            query_matches,
            key=lambda x: x["score"],
            reverse=True
        )[:top_match]

        final_results[q] = {
            "original_query": q,
            "corrected_query": corrected_q,
            "confidence": float(corrected_info["confidence"]),
            "description": (
                "Matches found" if query_matches else "No matches found"
            ),
            "matches": query_matches
        }

    return {
        "results": final_results,
        "time": round(time.time() - start_time, 6)
    }

@router.post("/category")
def get_category(query: CategoryQuery):
    """
    Advanced category search endpoint that uses:
    1. Phonetic correction using Double Metaphone
    2. Bigram-based context-aware word selection
    3. Token-based search across category aliases
    
    Args:
        query: Search query for category
        top_k: Number of top matches to return (default: 3)
    
    Returns:
        JSON with corrected query and matched items
    """
    import time
    start_time = time.time()
    
    try:
        # Step 1: Phonetic correction with bigram context
        words = query.query.lower().split()
        all_matches = [get_phonetic_candidates(w, PHONETIC_INDEX_cat) for w in words]
        
        corrected_words = []
        confidences = []
        
        for i in range(len(words)):
            original = words[i]
            candidates = all_matches[i]
            
            # Look ahead for next word candidates
            next_word_candidates = all_matches[i+1] if i + 1 < len(words) else None
            
            corrected, conf = choose_best_word(original, candidates, next_word_candidates, cat_bigrams)
            corrected_words.append(corrected)
            confidences.append(conf)
        
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        final_corrected = " ".join(corrected_words)
        
        # Detect Veg/Non-Veg Intent
        import re
        query_lower = query.query.lower()
        is_veg_only = False
        is_non_veg_only = False
        
        # Check for "veg" without "non" prefix
        if re.search(r'\bveg\b', query_lower) and not re.search(r'\bnon[\s-]?veg\b', query_lower):
            is_veg_only = True
        # Check for "non veg" or "non-veg" or "nonveg"
        elif re.search(r'\bnon[\s-]?veg\b', query_lower):
            is_non_veg_only = True
        
        # Step 2: Search using corrected query
        tokens = final_corrected.lower().split()
        
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        conditions = []
        params = []
        
        for token in tokens:
            # Each token must match either cat OR cat_alias1
            conditions.append("(cat LIKE ? OR cat_alias1 LIKE ?)")
            params.extend([f"%{token}%", f"%{token}%"])
        
        sql = f"""
            SELECT item_id, text, cat, price, type
            FROM menu_items
            WHERE {" AND ".join(conditions)}
            LIMIT ?
        """
        
        params.append(query.top_k * 50)  # Get more results to ensure we have enough after filtering
        cursor.execute(sql, params)
        
        items = cursor.fetchall()
        conn.close()
        
        # Filter results based on veg/non-veg intent
        filtered_items = []
        for item in items:
            item_type = item[4]  # type field
            
            if is_veg_only and item_type == "Veg":
                filtered_items.append(item)
            elif is_non_veg_only and item_type != "Veg":
                filtered_items.append(item)
            elif not is_veg_only and not is_non_veg_only:
                filtered_items.append(item)
            
            # Stop once we have enough results
            if len(filtered_items) >= query.top_k:
                break
        
        # Format results
        results = [
            {
                "item_id": item[0],
                "name": item[1],
                "category": item[2],
                "price": item[3],
                "type": item[4]
            } for item in filtered_items[:query.top_k]
        ]
        
        return {
            "original_query": query.query,
            "corrected_query": final_corrected,
            "confidence": round(avg_confidence, 2),
            "filter_applied": "veg_only" if is_veg_only else ("non_veg_only" if is_non_veg_only else "none"),
            "total_items_found": len(results),
            "items": results,
            "time_taken": round(time.time() - start_time, 4)
        }
        
    except Exception as e:
        return {
            "error": f"Error in category search: {str(e)}",
            "original_query": query.query
        }

