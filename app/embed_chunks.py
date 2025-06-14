import json
from compute_embeddings import compute_embeddings

def tds_course_content_embedding():
    chunks = [] # this will hold the loaded chunks
    # Load the chunks from the JSON file
    with open("../Data/tools-in-data-science-public/chunks.json") as f:
        for line in f:
            chunks.append(json.loads(line)) # each line is a separate JSON object

    # # Extract contents for embedding
    # texts = [chunk["content"] for chunk in chunks]

    # # Compute embeddings using your function (e.g., via Jina API)
    # embeddings = compute_embeddings(texts)  # should return list of vectors

    embeddings = []
    for chunk in chunks:
        embedding = compute_embeddings(chunk["content"])
        embeddings.append(embedding)
        print(f"Processed chunk ID: {chunk['id']} with embedding length: {len(embedding)}")

    # Combine with IDs and content
    embedded_chunks = []
    for chunk, emb in zip(chunks, embeddings):
        embedded_chunks.append({
            "id": chunk["id"],
            "content": chunk["content"],
            "embedding": emb
        })

    # Save to JSON file
    with open("../Data/chunks_embedding.json", "w", encoding="utf-8") as f:
        json.dump(embedded_chunks, f, indent=2)

def discourse_content_embedding():

    # Load discourse chunks
    with open("../Data/discourse_chunks.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    for chunk in data:
        text = f"{chunk['topic_title']} {chunk['content']}"
        embedding = compute_embeddings(text)
        chunk["embedding"] = embedding
        print(f"Processed chunk ID: {chunk['id']} with embedding length: {len(embedding)}")

# Save updated JSON
    with open("../Data/discourse_chunks_with_embeddings.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

# discourse_content_embedding() ## discourse chunks embedding are already computed

# tds_course_content_embedding()

## computed embeddings of both tds course and discourse chunks