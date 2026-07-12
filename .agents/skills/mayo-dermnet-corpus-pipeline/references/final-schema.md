# Final JSON schema

Both `RAG/dermnet_chunks_cleaned.json` and `RAG/mayo_chunks_cleaned.json` are JSON arrays whose records use the same contract.

## Top-level fields

| Field | Type | Required | DermAgent use |
|---|---|---:|---|
| `id` | string | yes | Unique 32-character lowercase hexadecimal chunk ID; combined with the source name to create the Qdrant point ID. |
| `text_for_embedding` | string | yes | Contextualized text embedded and indexed by `scripts/build_qdrant_rag.py`. |
| `original_content` | string | yes | Cleaned passage returned to the agent and used by the reranker. |
| `metadata` | object | yes | Attribution and filtering metadata consumed by `TextRAGTool`. |

## Metadata fields

| Field | Type | Required | DermAgent use |
|---|---|---:|---|
| `disease` | string | yes | Disease or topic context shown with retrieval results. |
| `header` | string | yes | Section context shown with retrieval results; it may be empty for text before the first heading. |
| `source_url` | string | yes | Canonical attribution URL shown with retrieval results. |
| `category` | string | yes | Normalized section category shown with retrieval results. |

Allowed `category` values:

- `Overview`
- `Causes`
- `Symptoms`
- `Complications`
- `Diagnosis`
- `Treatment`
- `Other`

Do not add `images`, `image_paths`, `content`, `hierarchy`, `title`, `original_header`, or `section_category` to final records.

## Synthetic example

This fabricated record demonstrates structure only and contains no real source content:

```json
[
  {
    "id": "0123456789abcdef0123456789abcdef",
    "text_for_embedding": "Example Condition - Management:\nThis synthetic sentence demonstrates the final schema.",
    "original_content": "This synthetic sentence demonstrates the final schema.",
    "metadata": {
      "disease": "Example Condition",
      "header": "Management",
      "source_url": "https://example.invalid/conditions/example",
      "category": "Treatment"
    }
  }
]
```
