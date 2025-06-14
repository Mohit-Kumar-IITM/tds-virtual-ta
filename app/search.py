import json
import base64
import numpy as np
from compute_embeddings import compute_embeddings
import heapq

# Cosine similarity
def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# Main search function
def search(query, top_k=3, image_base64=None):
    if not image_base64:
        query_embedding = compute_embeddings(query)
    else:
        query_embedding = compute_embeddings(question=query, image_base64=image_base64)
     
    chunk_paths = [
    "chunks_embedding.json",
    "discourse_chunks_with_embeddings.json"
    ] 
    # Use a min-heap to keep track of top-k similar chunks
    heap = []
    for path in chunk_paths:
        with open(path, "r", encoding="utf-8") as f:
            for chunk in json.load(f):
                if not chunk.get("embedding"):
                    continue
                sim = cosine_similarity(query_embedding, chunk["embedding"])
                item = (sim, chunk)
                if len(heap) < top_k:
                    heapq.heappush(heap, item)
                else:
                    heapq.heappushpop(heap, item)

    # Sort final top-k results in descending order
    top_chunks = sorted(heap, key=lambda x: x[0], reverse=True)
    return top_chunks

### testing the search function ###
# with open("sample_question_image.webp", "rb") as image_file:
#         base64_str = base64.b64encode(image_file.read()).decode("utf-8")

# print(search(query="Should I use gpt-4o-mini which AI proxy supports, or gpt3.5 turbo?", image_base64=base64_str))