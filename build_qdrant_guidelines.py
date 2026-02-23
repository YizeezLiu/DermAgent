"""
Build Qdrant Guidelines Database with Qwen3-Embedding-8B

This script loads the Qwen3-Embedding-8B text embedding model from HuggingFace,
computes embeddings for dermatology guidelines (dermnet + mayo), and stores them
in the local Qdrant server.

Usage:
    python build_qdrant_guidelines.py

Prerequisites:
    - Qdrant server running at localhost:6333
    - Sufficient GPU memory (~16GB+ for 8B model, or use 4-bit quantization)
"""

import os
import sys
import json
import torch
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm

# Check dependencies
try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import VectorParams, Distance, PointStruct
except ImportError:
    print("Error: qdrant-client not found. Please install: pip install qdrant-client")
    sys.exit(1)

try:
    from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig
except ImportError:
    print("Error: transformers not found. Please install: pip install transformers")
    sys.exit(1)

# =============================================================================
# Configuration
# =============================================================================
PROJECT_ROOT = Path(__file__).parent.resolve()

# Data paths
DERMNET_JSON = PROJECT_ROOT / "RAG" / "dermnet_chunks_cleaned.json"
MAYO_JSON = PROJECT_ROOT / "RAG" / "mayo_chunks_cleaned.json"

# Qdrant configuration
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "guidelines"

# Model configuration
EMBEDDING_MODEL_NAME = "Qwen/Qwen3-Embedding-8B"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4  # Small batch for 8B model to avoid OOM
USE_4BIT = True  # Enable 4-bit quantization to reduce memory usage

# Qwen3-Embedding-8B hidden size
VECTOR_SIZE = 4096


def check_gpu_status():
    """Check GPU availability and memory status."""
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"Found {gpu_count} GPU(s)")
        for i in range(gpu_count):
            props = torch.cuda.get_device_properties(i)
            total_mem = props.total_memory / 1e9
            allocated = torch.cuda.memory_allocated(i) / 1e9
            print(f"  GPU {i}: {props.name}, {total_mem:.1f}GB total, {allocated:.1f}GB allocated")
        return True
    else:
        print("No GPU available, using CPU (will be very slow)")
        return False


def load_chunks(json_path: Path, source_name: str) -> List[Dict[str, Any]]:
    """
    Load chunks from a JSON file.
    
    Args:
        json_path: Path to the JSON file
        source_name: Source identifier (e.g., "dermnet", "mayo")
        
    Returns:
        List of chunk dictionaries with source field added
    """
    if not json_path.exists():
        print(f"Error: File not found: {json_path}")
        return []
    
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    chunks = []
    for item in data:
        # Extract text_for_embedding field
        text = item.get("text_for_embedding")
        chunk_id = item.get("id")
        
        if text is None or chunk_id is None:
            continue
        
        chunks.append({
            "id": str(chunk_id),
            "text_for_embedding": text,
            "original_content": item.get("original_content", ""),
            "metadata": item.get("metadata", {}),
            "source": source_name
        })
    
    return chunks


def load_embedding_model(use_4bit: bool = True):
    """
    Load Qwen3-Embedding-8B model and tokenizer.
    
    Args:
        use_4bit: Whether to use 4-bit quantization to save memory
        
    Returns:
        Tuple of (tokenizer, model)
    """
    print(f"Loading model: {EMBEDDING_MODEL_NAME}")
    print(f"Device: {DEVICE}, 4-bit quantization: {use_4bit}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        EMBEDDING_MODEL_NAME,
        padding_side="left",
        trust_remote_code=True
    )
    
    # Configure quantization if enabled
    if use_4bit and DEVICE == "cuda":
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )
        model = AutoModel.from_pretrained(
            EMBEDDING_MODEL_NAME,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True
        )
    else:
        model = AutoModel.from_pretrained(
            EMBEDDING_MODEL_NAME,
            torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
            trust_remote_code=True
        )
        if DEVICE == "cuda":
            model = model.to(DEVICE)
    
    model.eval()
    print(f"Model loaded successfully. Hidden size: {model.config.hidden_size}")
    
    return tokenizer, model


def compute_embeddings(
    texts: List[str],
    tokenizer,
    model,
    max_length: int = 8192
) -> torch.Tensor:
    """
    Compute embeddings for a batch of texts using last-token pooling.
    
    Args:
        texts: List of input texts
        tokenizer: HuggingFace tokenizer
        model: HuggingFace model
        max_length: Maximum token length
        
    Returns:
        Normalized embedding tensor of shape (batch_size, hidden_size)
    """
    # Tokenize
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    
    # Move to device
    if hasattr(model, 'device'):
        device = model.device
    else:
        device = next(model.parameters()).device
    
    encoded = {k: v.to(device) for k, v in encoded.items()}
    
    # Forward pass
    with torch.no_grad():
        outputs = model(**encoded)
        last_hidden = outputs.last_hidden_state  # (batch, seq_len, hidden_size)
        
        # Last-token pooling: get embedding from the last valid token
        attention_mask = encoded["attention_mask"]
        seq_lengths = attention_mask.sum(dim=1) - 1  # Last valid token position
        batch_size = last_hidden.size(0)
        batch_indices = torch.arange(batch_size, device=device)
        
        # Extract last token embedding for each sample
        pooled = last_hidden[batch_indices, seq_lengths]  # (batch, hidden_size)
        
        # L2 normalize
        embeddings = torch.nn.functional.normalize(pooled, p=2, dim=1)
    
    return embeddings.cpu()


def build_guidelines_index():
    """Main function to build the Qdrant guidelines index."""
    print("=" * 60)
    print("Building Qdrant Guidelines Index with Qwen3-Embedding-8B")
    print("=" * 60)
    
    # Check GPU status
    check_gpu_status()
    
    # Load data
    print("\nLoading guideline chunks...")
    dermnet_chunks = load_chunks(DERMNET_JSON, source_name="dermnet")
    mayo_chunks = load_chunks(MAYO_JSON, source_name="mayo")
    all_chunks = dermnet_chunks + mayo_chunks
    
    print(f"  Dermnet chunks: {len(dermnet_chunks)}")
    print(f"  Mayo chunks: {len(mayo_chunks)}")
    print(f"  Total chunks: {len(all_chunks)}")
    
    if not all_chunks:
        print("Error: No chunks loaded. Exiting.")
        sys.exit(1)
    
    # Load embedding model
    print("\nLoading embedding model...")
    tokenizer, model = load_embedding_model(use_4bit=USE_4BIT)
    
    # Initialize Qdrant client
    print(f"\nConnecting to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}...")
    try:
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        # Test connection
        client.get_collections()
        print("Connected to Qdrant successfully.")
    except Exception as e:
        print(f"Error connecting to Qdrant: {e}")
        print("Please ensure Qdrant server is running at localhost:6333")
        sys.exit(1)
    
    # Create or recreate collection
    if client.collection_exists(COLLECTION_NAME):
        print(f"Collection '{COLLECTION_NAME}' exists. Deleting to rebuild...")
        client.delete_collection(COLLECTION_NAME)
    
    print(f"Creating collection '{COLLECTION_NAME}'...")
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=VECTOR_SIZE,
            distance=Distance.COSINE
        )
    )
    
    # Process chunks in batches
    print(f"\nProcessing {len(all_chunks)} chunks in batches of {BATCH_SIZE}...")
    total_batches = (len(all_chunks) + BATCH_SIZE - 1) // BATCH_SIZE
    success_count = 0
    error_count = 0
    
    for i in tqdm(range(0, len(all_chunks), BATCH_SIZE), total=total_batches, desc="Indexing"):
        batch = all_chunks[i:i + BATCH_SIZE]
        
        try:
            # Extract texts for embedding
            texts = [chunk["text_for_embedding"] for chunk in batch]
            
            # Compute embeddings
            embeddings = compute_embeddings(texts, tokenizer, model)
            embeddings_list = embeddings.tolist()
            
            # Prepare points
            points = []
            for j, chunk in enumerate(batch):
                # Use source_id as unique point ID
                point_id = f"{chunk['source']}_{chunk['id']}"
                
                # Build payload
                payload = {
                    "id": chunk["id"],
                    "text": chunk["text_for_embedding"],
                    "original_content": chunk["original_content"],
                    "source": chunk["source"],
                    "disease": chunk["metadata"].get("disease", ""),
                    "header": chunk["metadata"].get("header", ""),
                    "source_url": chunk["metadata"].get("source_url", ""),
                    "category": chunk["metadata"].get("category", "")
                }
                
                points.append(PointStruct(
                    id=point_id,
                    vector=embeddings_list[j],
                    payload=payload
                ))
            
            # Upsert to Qdrant
            client.upsert(
                collection_name=COLLECTION_NAME,
                points=points
            )
            success_count += len(points)
            
        except torch.cuda.OutOfMemoryError:
            print(f"\nGPU OOM at batch {i // BATCH_SIZE}. Try reducing BATCH_SIZE.")
            torch.cuda.empty_cache()
            error_count += 1
            continue
        except Exception as e:
            print(f"\nError processing batch {i // BATCH_SIZE}: {e}")
            error_count += 1
            continue
    
    # Summary
    print("\n" + "=" * 60)
    print("Indexing Complete!")
    print("=" * 60)
    print(f"Successfully indexed: {success_count} documents")
    print(f"Batches with errors: {error_count}")
    print(f"Collection name: {COLLECTION_NAME}")
    print(f"Qdrant server: {QDRANT_HOST}:{QDRANT_PORT}")
    
    # Verify collection
    collection_info = client.get_collection(COLLECTION_NAME)
    print(f"Collection points count: {collection_info.points_count}")


if __name__ == "__main__":
    build_guidelines_index()
