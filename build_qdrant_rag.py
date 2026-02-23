import sys
import os
import json
import torch
import hashlib
from tqdm import tqdm
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJECT_ROOT)

from benchmark.models import get_model

# Configuration
CHUNKS_FILES = [
    "RAG/dermnet_chunks_cleaned.json",
    "RAG/mayo_chunks_cleaned.json"
]
# Use localhost as requested, assuming server is running and using qdrant_server_storage
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "derm_rag"
MODEL_ID = "Qwen/Qwen3-Embedding-8B"
BATCH_SIZE = 32  # 8B model is large, keep batch size moderate
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def build_rag_index():
    print(f"Running on device: {DEVICE}")
    
    # Initialize Model
    print(f"Loading Embedding model ({MODEL_ID})...")
    try:
        model = get_model("qwen3-embedding", device=DEVICE)
    except Exception as e:
        print(f"Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Determine vector size
    print("Determining vector size...")
    dummy_emb = model.encode_texts(["test"])
    vector_size = dummy_emb.shape[-1]
    print(f"Vector size detected: {vector_size}")

    # Initialize Qdrant
    print(f"Initializing Qdrant client ({QDRANT_HOST}:{QDRANT_PORT})...")
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=120)

    # Re-create collection
    if client.collection_exists(COLLECTION_NAME):
        print(f"Collection '{COLLECTION_NAME}' exists. Deleting to rebuild...")
        client.delete_collection(COLLECTION_NAME)

    print(f"Creating collection '{COLLECTION_NAME}'...")
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
    )

    success_count = 0
    error_count = 0

    for json_file in CHUNKS_FILES:
        full_path = os.path.join(PROJECT_ROOT, json_file)
        print(f"Processing {full_path}...")
        
        if not os.path.exists(full_path):
            print(f"File not found: {full_path}")
            continue

        with open(full_path, 'r', encoding='utf-8') as f:
            chunks = json.load(f)

        print(f"Loaded {len(chunks)} chunks from {json_file}")
        
        for i in tqdm(range(0, len(chunks), BATCH_SIZE), desc=f"Indexing {json_file}"):
            batch_chunks = chunks[i:i+BATCH_SIZE]
            
            texts = [c.get("text_for_embedding", "") for c in batch_chunks]
            
            # Skip empty texts
            valid_indices = [j for j, t in enumerate(texts) if t.strip()]
            if not valid_indices:
                continue
                
            batch_texts = [texts[j] for j in valid_indices]
            
            # Encode
            try:
                embeddings = model.encode_texts(batch_texts)
            except Exception as e:
                print(f"Error encoding batch at index {i}: {e}")
                error_count += 1
                continue

            points = []
            for k, original_idx in enumerate(valid_indices):
                chunk = batch_chunks[original_idx]
                
                # Use existing ID or generate one
                point_id = chunk.get("id")
                if not point_id:
                     point_id = hashlib.md5(batch_texts[k].encode()).hexdigest()
                
                # Payload
                payload = {
                    "text_for_embedding": batch_texts[k],
                    "original_content": chunk.get("original_content", ""),
                    "metadata": chunk.get("metadata", {}),
                    "source_file": json_file
                }
                
                points.append(PointStruct(
                    id=point_id,
                    vector=embeddings[k].tolist(),
                    payload=payload
                ))

            if points:
                try:
                    client.upsert(
                        collection_name=COLLECTION_NAME,
                        points=points
                    )
                    success_count += len(points)
                except Exception as e:
                    print(f"Error upserting batch at index {i}: {e}")
                    error_count += 1

    print("-" * 50)
    print("Indexing complete!")
    print(f"Successfully indexed: {success_count} chunks")
    print(f"Batches with errors: {error_count}")

if __name__ == "__main__":
    build_rag_index()
