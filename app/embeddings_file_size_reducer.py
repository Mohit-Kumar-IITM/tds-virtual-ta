import json

with open("discourse_chunks_with_embeddings.json", "r", encoding="utf-8") as f:
    data = json.load(f)

with open("discourse_chunks_compact.json", "w", encoding="utf-8") as f:
    json.dump(data, f, separators=(",", ":"))  # no spaces, minimal file size
