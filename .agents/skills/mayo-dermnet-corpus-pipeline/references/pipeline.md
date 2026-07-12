# Final corpus pipeline

## Python environment

Use CPython 3.10 or newer, matching the DermAgent project requirement. Keep corpus generation in an isolated environment because it does not require the GPU packages used by the main agent.

Create a lightweight Conda environment:

```bash
conda create -n dermagent-corpus python=3.10
conda activate dermagent-corpus
python -m pip install --upgrade pip
python -m pip install "requests>=2.31.0" "beautifulsoup4>=4.12.0" "lxml>=4.9.0" "markdownify>=0.11.0" "pandas>=2.0.0" "numpy>=1.24.0" "python-dotenv>=1.0.0" "tqdm>=4.66.0"
```

Record the resolved versions with `python -m pip freeze` when publishing reproduction metadata.

## Inputs

### DermNet

Use the bundled `dermnet_terms.csv` manifest with:

- `term`: topic name used as the disease label
- `url`: canonical DermNet page URL

The manifest contains 226 topics with unique, non-empty terms and DermNet URLs.

### Mayo Clinic

Use the bundled `mayo_diseases.csv` manifest containing a single `disease` column. Each row is already unique and non-empty.

The manifest contains 71 disease queries in first-occurrence order from the source guideline inventory.

## Stage 1: collect source pages

### DermNet

1. Fetch each curated URL with a normal browser user agent, bounded timeout, error logging, and a polite delay.
2. Select the main article and remove scripts, styles, navigation, advertisements, sharing widgets, breadcrumbs, and unrelated page furniture.
3. Convert the remaining headings and body content to Markdown.
4. Write one topic directory and one Markdown file per topic.
5. Put the topic title and canonical source URL at the top of the document.

### Mayo Clinic

1. Search Mayo Clinic for each unique disease query and resolve the most relevant disease page.
2. Collect the symptoms/causes page.
3. Follow and append the diagnosis/treatment page when available.
4. Remove scripts, styles, navigation, advertisements, forms, and unrelated page furniture.
5. Convert the combined content to Markdown.
6. Write one disease directory and one Markdown file per successful query, retaining the resolved source URL.

## Stage 2: clean Markdown

### DermNet

- Remove category-navigation links, duplicate top titles, footer sections, bibliography/navigation blocks, and empty link artifacts.
- Normalize horizontal rules, repeated blank lines, and surrounding whitespace.
- Preserve clinical headings, article text, and source attribution.

### Mayo Clinic

- Remove product/service promotions, newsletters, appointment prompts, print controls, associated-procedure blocks, page identifiers, and duplicated section-navigation links.
- Remove request-form errors and UI-only fragments.
- Normalize horizontal rules, repeated blank lines, and surrounding whitespace.
- Preserve clinical headings, article text, and source attribution.

## Stage 3: create heading-based chunks

1. Parse Markdown headings from H1 through H6.
2. Maintain the current heading hierarchy while scanning each document.
3. Emit one chunk for the text between adjacent headings.
4. Carry forward the disease label, current header, heading hierarchy, and source URL.
5. Generate a deterministic 32-character lowercase hexadecimal ID from the relative Markdown path and chunk content.

Preserve the same relative directory layout when stable IDs are required across reruns.

## Stage 4: finalize records

### Common finalization

1. Remove empty, very short, navigation-only, link-only, image-only, and UI-only chunks.
2. Remove Markdown image syntax and unwrap Markdown links while retaining visible text.
3. Store the cleaned clinical passage in `original_content`.
4. Assign one category from `Overview`, `Causes`, `Symptoms`, `Complications`, `Diagnosis`, `Treatment`, or `Other` using the section header and hierarchy.
5. Remove image paths and all intermediate-only fields.
6. Serialize each record with the exact contract in `final-schema.md`.

For DermNet, format `text_for_embedding` as:

```text
{title-cased disease} - {header}:
{original_content}
```

For Mayo Clinic, use the joined heading hierarchy when available and fall back to the disease label:

```text
{hierarchy joined with " > " or disease} - {header}:
{original_content}
```

### DermNet-specific filtering

- Exclude reference-only sections.
- Exclude chunks that become too short after link and image cleanup.

### Mayo-specific filtering and deduplication

- Exclude `Care at Mayo Clinic` and `More Information` sections.
- Exclude page-error blocks and print, separator, or UI fragments.
- Deduplicate using disease, header, heading hierarchy, and cleaned content together.

## Stage 5: validate and integrate with DermAgent

Validate each final file:

- Top level is a JSON array.
- Every record has exactly the four final top-level fields.
- Every `metadata` object has exactly the four final metadata fields.
- IDs are unique and match 32 lowercase hexadecimal characters.
- Both text fields are non-empty strings.
- DermNet URLs use the DermNet domain and Mayo URLs use the Mayo Clinic domain.
- Categories use only the seven allowed values.
- No `images`, `image_paths`, `content`, `hierarchy`, or source-specific temporary fields remain.

Place the validated files at:

- `RAG/dermnet_chunks_cleaned.json`
- `RAG/mayo_chunks_cleaned.json`

From the DermAgent repository root, run `python scripts/build_qdrant_rag.py` to load both files into the `derm_rag` Qdrant collection. The builder embeds `text_for_embedding`; `TextRAGTool` returns `original_content` and reads `disease`, `header`, `category`, and `source_url` from `metadata`.
