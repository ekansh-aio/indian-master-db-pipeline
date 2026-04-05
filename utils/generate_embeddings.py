import json
from sentence_transformers import SentenceTransformer

# Load model (384-dim vectors)
model = SentenceTransformer("all-MiniLM-L6-v2")

input_file = "legal_data.json"
output_file = "legal_data_with_embeddings.json"

# Load your cleaned legal data
with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

# What to embed (text only)
texts = [doc["text"] for doc in data]

# Generate embeddings
embeddings = model.encode(texts, show_progress_bar=True)

# Attach embeddings to each document
for doc, emb in zip(data, embeddings):
    doc["embedding"] = emb.tolist()

# Save new file
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2)

print("Embeddings generated and saved to legal_data_with_embeddings.json")
