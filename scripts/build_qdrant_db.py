"""
Build Qdrant Image RAG Database

Encodes Derm1M images and captions using DermLIP (PanDerm), and stores them
in the local Qdrant server under the 'derm1m' collection used by RAGTool.

Usage:
    python scripts/build_qdrant_db.py

Prerequisites:
    - Qdrant server running at localhost:6333
    - Derm1M dataset at datasets/Derm1M/
    - Derm1M open_clip fork at Derm1M/src/ (provides BEiT architecture support)
"""

import sys
import os
import torch
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from typing import List
import hashlib

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import VectorParams, Distance, PointStruct
except ImportError:
    print("Error: qdrant-client not found. Install with: pip install qdrant-client")
    sys.exit(1)

# Configuration
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
CSV_PATH = "datasets/Derm1M/Derm1M_v2_pretrain.csv"
IMAGE_ROOT = "datasets/Derm1M"
COLLECTION_NAME = "derm1m"
MODEL_ID = "redlessone/DermLIP_PanDerm-base-w-PubMed-256"
BATCH_SIZE = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_dermlip_model():
    """Load DermLIP using the Derm1M open_clip fork (required for BEiT architecture)."""
    derm1m_src = PROJECT_ROOT / "Derm1M" / "src"
    if not derm1m_src.exists():
        print(f"Error: Derm1M source not found at {derm1m_src}")
        print("Please ensure the Derm1M repository is placed at Derm1M/ in the project root.")
        sys.exit(1)

    # Temporarily prioritise Derm1M's open_clip to avoid conflicts with MAKE's version
    original_path = sys.path.copy()
    sys.path = [p for p in sys.path if 'MAKE' not in p]
    if str(derm1m_src) not in sys.path:
        sys.path.insert(0, str(derm1m_src))

    # Force reimport to get the Derm1M version
    for mod in [k for k in list(sys.modules) if k.startswith('open_clip')]:
        del sys.modules[mod]

    import open_clip
    model, preprocess = open_clip.create_model_from_pretrained(
        f'hf-hub:{MODEL_ID}', device=DEVICE
    )
    tokenizer = open_clip.get_tokenizer(f'hf-hub:{MODEL_ID}')
    model.eval()

    sys.path = original_path
    return model, preprocess, tokenizer


def build_qdrant_index():
    print(f"Running on device: {DEVICE}")
    print("Initializing Qdrant client at localhost:6333...")
    client = QdrantClient(host="localhost", port=6333)

    print(f"Loading data from {CSV_PATH}...")
    if not os.path.exists(CSV_PATH):
        print(f"Error: CSV file not found at {CSV_PATH}")
        sys.exit(1)

    df = pd.read_csv(CSV_PATH)

    # Filter out VQA images
    initial_count = len(df)
    df['filename'] = df['filename'].astype(str)
    df = df[~df['filename'].str.startswith('VQA/')]
    print(f"Filtered {initial_count - len(df)} VQA images. Remaining: {len(df)}")

    # Load model
    print(f"Loading DermLIP model ({MODEL_ID})...")
    try:
        model, preprocess, tokenizer = load_dermlip_model()
    except Exception as e:
        print(f"Failed to load model: {e}")
        sys.exit(1)

    # Determine vector size
    vector_size = 512  # Known size for DermLIP_PanDerm-base-w-PubMed-256
    print(f"Vector size: {vector_size}")

    # Re-create collection
    if client.collection_exists(COLLECTION_NAME):
        print(f"Collection '{COLLECTION_NAME}' exists. Deleting to rebuild...")
        client.delete_collection(COLLECTION_NAME)

    print(f"Creating collection '{COLLECTION_NAME}'...")
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            "image_embedding": VectorParams(size=vector_size, distance=Distance.COSINE),
            "caption_embedding": VectorParams(size=vector_size, distance=Distance.COSINE),
        }
    )
    
    # Processing loop
    total_batches = (len(df) + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"Starting indexing of {len(df)} items in {total_batches} batches...")
    
    success_count = 0
    error_count = 0
    
    for i in tqdm(range(0, len(df), BATCH_SIZE), total=total_batches, desc="Indexing"):
        batch_df = df.iloc[i:i+BATCH_SIZE]
        
        image_paths = []
        valid_indices = []
        captions = []
        
        # Prepare Batch
        for idx, row in batch_df.iterrows():
            img_rel_path = row['filename']
            img_path = os.path.join(IMAGE_ROOT, img_rel_path)
            
            # Check if file exists
            if not os.path.exists(img_path):
                continue
                
            image_paths.append(img_path)
            # Use truncated_caption for embedding
            caption_for_embedding = str(row['truncated_caption']) if pd.notna(row['truncated_caption']) else ""
            captions.append(caption_for_embedding)
            valid_indices.append(idx)

        if not image_paths:
            continue

        try:
            # 1. Image Embeddings
            imgs = [preprocess(Image.open(p).convert("RGB")) for p in image_paths]
            img_tensor = torch.stack(imgs).to(DEVICE)
            with torch.no_grad():
                image_features = model.encode_image(img_tensor)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            image_embeddings = image_features.cpu().numpy()

            # 2. Caption Embeddings — tokenizer may return dict (HF) or tensor (OpenCLIP)
            tokenized = tokenizer(captions)
            if isinstance(tokenized, dict):
                text_tokens = tokenized['input_ids'].to(DEVICE)
            else:
                text_tokens = tokenized.to(DEVICE)
            with torch.no_grad():
                text_features = model.encode_text(text_tokens)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            text_embeddings = text_features.cpu().numpy()
                
            # 3. Prepare Points
            points = []
            for j, df_idx in enumerate(valid_indices):
                row = df.loc[df_idx]
                
                # Payload construction
                payload = {
                    "filename": str(row['filename']),
                    "original_caption": str(row['caption']) if pd.notna(row['caption']) else "",
                    "truncated_caption": str(row['truncated_caption']) if pd.notna(row['truncated_caption']) else "",
                    "disease_label": str(row['disease_label']) if pd.notna(row['disease_label']) else "",
                    "hierarchical_disease_label": str(row['hierarchical_disease_label']) if pd.notna(row['hierarchical_disease_label']) else "",
                    "skin_concept": str(row['skin_concept']) if pd.notna(row['skin_concept']) else "",
                    "body_location": str(row['body_location']) if pd.notna(row['body_location']) else ""
                }
                
                # Create stable ID
                point_id = hashlib.md5(str(row['filename']).encode()).hexdigest()
                
                points.append(PointStruct(
                    id=point_id,
                    vector={
                        "image_embedding": image_embeddings[j].tolist(),
                        "caption_embedding": text_embeddings[j].tolist()
                    },
                    payload=payload
                ))
            
            # 4. Upsert
            if points:
                client.upsert(
                    collection_name=COLLECTION_NAME,
                    points=points
                )
                success_count += len(points)
            
        except Exception as e:
            print(f"Error processing batch starting at index {i}: {e}")
            error_count += 1
            continue

    print("-" * 50)
    print("Indexing complete!")
    print(f"Successfully indexed: {success_count} documents")
    print(f"Batches with errors: {error_count}")
    print("Qdrant server: localhost:6333")

if __name__ == "__main__":
    build_qdrant_index()