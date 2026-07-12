---
name: mayo-dermnet-corpus-pipeline
description: Reproduce the Mayo Clinic and DermNet guideline corpora consumed by DermAgent Text RAG. Use when Codex needs to collect source pages, clean Markdown, create heading-based chunks, filter and deduplicate content, generate the final unified JSON records under RAG/, or validate compatibility with scripts/build_qdrant_rag.py and TextRAGTool.
---

# Mayo and DermNet Corpus Pipeline

Reproduce the two guideline corpora required by DermAgent without redistributing scraper source or copyrighted corpus content in this skill.

## Read the references

1. Read `references/pipeline.md` for the Python environment and complete pipeline.
2. Read `references/final-schema.md` for the only supported output contract and synthetic example.

## Rules

- Work in a separate directory and never overwrite an existing corpus without an explicit backup.
- Preserve the canonical source URL in every final record.
- Respect each source site's terms, robots guidance, and request-rate limits.
- Do not add scraper implementations, URL manifests, downloaded images, article text, or generated corpus JSON to this skill or repository.
- Expose only the final JSON schema; do not document intermediate JSON schemas.
- Write the validated outputs to `RAG/dermnet_chunks_cleaned.json` and `RAG/mayo_chunks_cleaned.json`.

## Completion criteria

- Both sources pass collection, Markdown cleaning, chunking, finalization, and validation.
- Every output record matches `references/final-schema.md` exactly.
- IDs are unique within each source corpus.
- `text_for_embedding` and `original_content` are non-empty.
- `metadata.source_url` belongs to the expected source domain.
- `metadata.category` belongs to the seven-category vocabulary.
- No final record contains image or intermediate-only fields.
- `scripts/build_qdrant_rag.py` loads both files without skipping valid records.
