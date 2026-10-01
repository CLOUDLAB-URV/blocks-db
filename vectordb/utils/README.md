# utils — Shared Utilities

| File | Description |
|------|-------------|
| `s3_client.py` | Singleton S3 client (shared across modules) |
| `s3_utils.py` | S3 Express bucket detection helpers |
| `dataset_ops.py` | Upload, delete, update datasets (CSV in S3) |
| `index_ops.py` | Save, load, delete, reindex FAISS index configs |
| `query_ops.py` | `get_vectors_by_id`, `list_vectors`, `list_vectors_paginated` |
| `vector_utils.py` | CSV parsing: load vectors with/without IDs/tags |
| `vector_tracking.py` | `VectorIndexTracker` — DynamoDB + S3 tracking for pending/indexed vectors |
| `hybrid_search.py` | `brute_force_search` + `merge_search_results` for hybrid queries |
| `parquet.py` | `inspect()` — column layout, vector dimension and rows per row group of a parquet file, from its footer (and its first vector when the schema does not fix the length); `iter_ranges()` — reads row ranges into float32 vectors, scaling `owi-v2` vectors to unit length, and counts the rows it rejects |
| `idmap.py` | `select()` — the `(record_id, chunk_idx)` of vector ids, from the id map parts of a parquet index; `idmap_prefix()` — the S3 prefix of those parts |
| `waiting.py` | `collect` — wait for the functions of a map, giving up when some never start and none starts or finishes within the wait window |
