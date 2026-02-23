import sys
import os
import torch
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from typing import List
import hashlib

# Check if Qdrant client is installed
try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import VectorParams, Distance, PointStruct
except ImportError:
    print("Error: qdrant-client not found. Please install it with: pip install qdrant-client")
    sys.exit(1)

# Add project root to path so we can import benchmark modules
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJECT_ROOT)

# Import DermLIP model
try:
    from benchmark.models.dermlip import DermLIPModel
except ImportError as e:
    print(f"Error importing DermLIPModel: {e}")
    print("Ensure you are running this script from the project root and that benchmark/models/dermlip.py exists.")
    sys.exit(1)

# Configuration
CSV_PATH = "datasets/Derm1M/Derm1M_v2_pretrain.csv"
IMAGE_ROOT = "datasets/Derm1M"
QDRANT_PATH = "./qdrant_storage"
COLLECTION_NAME = "derm1m"
MODEL_ID = "redlessone/DermLIP_PanDerm-base-w-PubMed-256"
BATCH_SIZE = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def build_qdrant_index():
    print(f"Running on device: {DEVICE}")
    print(f"Initializing Qdrant client at {QDRANT_PATH}...")
    # Initialize persistent local storage
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
    
    # Initialize Model
    print(f"Loading DermLIP model ({MODEL_ID})...")
    try:
        model_wrapper = DermLIPModel(model_id=MODEL_ID, device=DEVICE)
    except Exception as e:
        print(f"Failed to load model: {e}")
        sys.exit(1)
    
    # Determine vector size dynamically
    print("Determining vector size...")
    try:
        dummy_text_input = ["test"]
        # Robust tokenization for vector size check
        try:
            # DermLIP wrapper (HFTokenizer) handles padding/truncation/tensors internally
            tokens = model_wrapper.tokenizer(dummy_text_input)
            
            if isinstance(tokens, dict):
                dummy_text = tokens['input_ids'].to(DEVICE)
            else:
                dummy_text = tokens.to(DEVICE)
        except TypeError:
            # Fallback for simple tokenizers
            dummy_text = model_wrapper.tokenizer(dummy_text_input).to(DEVICE)

        with torch.no_grad():
            dummy_emb = model_wrapper.model.encode_text(dummy_text)
            vector_size = dummy_emb.shape[-1]
        print(f"Vector size detected: {vector_size}")
    except Exception as e:
        print(f"Error determining vector size: {e}")
        # Fallback to known size for this model
        if "PubMed-256" in MODEL_ID:
            print("Fallback: Using known vector size 512 for PubMed-256 model")
            vector_size = 512
        else:
            sys.exit(1)

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
            image_embeddings_tensor = model_wrapper.encode_images(image_paths, batch_size=len(image_paths))
            image_embeddings = image_embeddings_tensor.cpu().numpy()
            
            # 2. Caption Embeddings
            # FIXED: Robust tokenization logic handling both HF Dictionaries and CLIP Tensors
            try:
                # Attempt HuggingFace style tokenization (standard for DermLIP/PubMedBERT)
                # Note: DermLIP's HFTokenizer wrapper does not accept kwargs like return_tensors
                # It handles them internally.
                tokenized_output = model_wrapper.tokenizer(captions)
                
                # If output is a dictionary (HF), extract input_ids
                if isinstance(tokenized_output, dict):
                    text_tokens = tokenized_output['input_ids'].to(DEVICE)
                else:
                    # If output is already a tensor (OpenCLIP SimpleTokenizer or HFTokenizer wrapper)
                    text_tokens = tokenized_output.to(DEVICE)
                    
            except TypeError:
                # Fallback if arguments like 'return_tensors' are not supported
                text_tokens = model_wrapper.tokenizer(captions).to(DEVICE)

            with torch.no_grad():
                text_features = model_wrapper.model.encode_text(text_tokens)
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
    print(f"Qdrant storage location: {os.path.abspath(QDRANT_PATH)}")

if __name__ == "__main__":
    build_qdrant_index()