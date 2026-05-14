"""
Build Qdrant Text RAG Database

Loads Qwen3-Embedding-8B, computes embeddings for dermatology guideline chunks
(DermNet + Mayo Clinic), and stores them in the local Qdrant server under the
'derm_rag' collection used by TextRAGTool.

Usage:
    python scripts/build_qdrant_rag.py

Prerequisites:
    - Qdrant server running at localhost:6333
    - RAG/dermnet_chunks_cleaned.json and RAG/mayo_chunks_cleaned.json present
    - ~16GB VRAM (or ~8GB with 4-bit quantization)
"""

import os
import sys
import json
import torch
import hashlib
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import VectorParams, Distance, PointStruct
except ImportError:
    print("Error: qdrant-client not found. Install with: pip install qdrant-client")
    sys.exit(1)

try:
    from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig
except ImportError:
    print("Error: transformers not found. Install with: pip install transformers")
    sys.exit(1)

# =============================================================================
# Configuration
# =============================================================================
PROJECT_ROOT = Path(__file__).parent.parent.resolve()

CHUNK_FILES = [
    ("RAG/dermnet_chunks_cleaned.json", "dermnet"),
    ("RAG/mayo_chunks_cleaned.json", "mayo"),
]

QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "derm_rag"

EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-8B"
VECTOR_SIZE = 4096          # Qwen3-Embedding-8B hidden size
BATCH_SIZE = 4              # Conservative for 8B model; increase if VRAM allows
USE_4BIT = True             # 4-bit quantization to save ~8GB VRAM
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =============================================================================
# Helpers
# =============================================================================

def check_gpu_status() -> None:
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            total = props.total_memory / 1e9
            used = torch.cuda.memory_allocated(i) / 1e9
            print(f"  GPU {i}: {props.name}, {total:.1f}GB total, {used:.1f}GB used")
    else:
        print("  No GPU found — using CPU (will be very slow)")


def load_chunks(json_path: Path, source: str) -> List[Dict[str, Any]]:
    if not json_path.exists():
        print(f"  Warning: {json_path} not found, skipping.")
        return []
    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    chunks = []
    for item in raw:
        text = item.get("text_for_embedding")
        chunk_id = item.get("id")
        if not text or chunk_id is None:
            continue
        chunks.append({
            "id": str(chunk_id),
            "text_for_embedding": text,
            "original_content": item.get("original_content", ""),
            "metadata": item.get("metadata", {}),
            "source": source,
        })
    return chunks


def load_model(use_4bit: bool):
    print(f"  Model: {EMBEDDING_MODEL}")
    print(f"  Device: {DEVICE}, 4-bit: {use_4bit}")
    tokenizer = AutoTokenizer.from_pretrained(
        EMBEDDING_MODEL, padding_side="left", trust_remote_code=True
    )
    if use_4bit and DEVICE == "cuda":
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModel.from_pretrained(
            EMBEDDING_MODEL, quantization_config=quant_cfg,
            device_map="auto", trust_remote_code=True
        )
    else:
        model = AutoModel.from_pretrained(
            EMBEDDING_MODEL,
            torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
            trust_remote_code=True,
        )
        if DEVICE == "cuda":
            model = model.to(DEVICE)
    model.eval()
    return tokenizer, model


def encode(texts: List[str], tokenizer, model, max_length: int = 8192) -> torch.Tensor:
    """Mean pooling with L2 normalisation — matches TextRAGTool._encode_text()."""
    encoded = tokenizer(
        texts, padding=True, truncation=True,
        max_length=max_length, return_tensors="pt"
    )
    dev = next(model.parameters()).device
    encoded = {k: v.to(dev) for k, v in encoded.items()}
    with torch.no_grad():
        out = model(**encoded)
        last_hidden = out.last_hidden_state                    # (B, L, H)
        mask = encoded["attention_mask"].unsqueeze(-1).float() # (B, L, 1)
        pooled = (last_hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        return torch.nn.functional.normalize(pooled, p=2, dim=1).cpu()


# =============================================================================
# Main
# =============================================================================

def build_rag_index() -> None:
    print("=" * 60)
    print("Building Qdrant Text RAG Index  (collection: derm_rag)")
    print("=" * 60)

    # GPU status
    print("\nGPU status:")
    check_gpu_status()

    # Load chunks
    print("\nLoading guideline chunks...")
    all_chunks: List[Dict[str, Any]] = []
    for rel_path, source in CHUNK_FILES:
        chunks = load_chunks(PROJECT_ROOT / rel_path, source)
        print(f"  {source}: {len(chunks)} chunks")
        all_chunks.extend(chunks)
    if not all_chunks:
        print("Error: no chunks loaded. Exiting.")
        sys.exit(1)
    print(f"  Total: {len(all_chunks)} chunks")

    # Load model
    print("\nLoading embedding model...")
    tokenizer, model = load_model(use_4bit=USE_4BIT)

    # Connect to Qdrant
    print(f"\nConnecting to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}...")
    try:
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=120)
        client.get_collections()
        print("  Connected.")
    except Exception as e:
        print(f"  Error: {e}")
        print("  Make sure Qdrant is running: docker run -p 6333:6333 qdrant/qdrant")
        sys.exit(1)

    # Recreate collection
    if client.collection_exists(COLLECTION_NAME):
        print(f"  Collection '{COLLECTION_NAME}' exists — deleting to rebuild...")
        client.delete_collection(COLLECTION_NAME)
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )
    # Create full-text index on 'text_for_embedding' for keyword search mode in TextRAGTool
    from qdrant_client.models import TextIndexParams, TokenizerType
    client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="text_for_embedding",
        field_schema=TextIndexParams(type="text", tokenizer=TokenizerType.WORD, min_token_len=2),
    )
    print(f"  Collection '{COLLECTION_NAME}' created (with full-text index on 'text_for_embedding').")

    # Index
    print(f"\nIndexing {len(all_chunks)} chunks (batch size {BATCH_SIZE})...")
    success, errors = 0, 0
    total_batches = (len(all_chunks) + BATCH_SIZE - 1) // BATCH_SIZE

    for i in tqdm(range(0, len(all_chunks), BATCH_SIZE), total=total_batches):
        batch = all_chunks[i : i + BATCH_SIZE]
        texts = [c["text_for_embedding"] for c in batch]

        try:
            embeddings = encode(texts, tokenizer, model)

            points = []
            for j, chunk in enumerate(batch):
                point_id = hashlib.md5(
                    f"{chunk['source']}_{chunk['id']}".encode()
                ).hexdigest()
                points.append(PointStruct(
                    id=point_id,
                    vector=embeddings[j].tolist(),
                    payload={
                        "id": chunk["id"],
                        # Key name must match TextRAGTool keyword search filter and reranker fallback
                        "text_for_embedding": chunk["text_for_embedding"],
                        "original_content": chunk["original_content"],
                        # source_file shown in TextRAGTool._format_results() as "Data Source"
                        "source_file": chunk["source"],
                        # metadata nested dict: TextRAGTool reads payload.get("metadata", {})
                        "metadata": chunk["metadata"],
                    },
                ))

            client.upsert(collection_name=COLLECTION_NAME, points=points)
            success += len(points)

        except torch.cuda.OutOfMemoryError:
            print(f"\n  GPU OOM at batch {i // BATCH_SIZE} — try reducing BATCH_SIZE.")
            torch.cuda.empty_cache()
            errors += 1
        except Exception as e:
            print(f"\n  Error at batch {i // BATCH_SIZE}: {e}")
            errors += 1

    # Summary
    info = client.get_collection(COLLECTION_NAME)
    print("\n" + "=" * 60)
    print(f"Done. Indexed: {success}  Errors: {errors}")
    print(f"Collection '{COLLECTION_NAME}' — {info.points_count} points in Qdrant.")
    print("=" * 60)


if __name__ == "__main__":
    build_rag_index()
