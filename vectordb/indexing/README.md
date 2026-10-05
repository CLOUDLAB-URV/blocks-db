# indexing — Indexing Pipeline Orchestration

The initial index build via Lithops (`initialize-database`). Pipeline stages:

1. **Preprocess** (`Preprocessor` ABC): optionally chunk the source CSV into smaller pieces for parallel ingestion.
2. **Distribute** (`Partitioner` ABC): partition vector data across workers.
3. **Build** (`IndexBuilder` ABC): each Lithops worker builds a FAISS IVF index for assigned centroids using `index_factory(features, "IVF{k},Flat")`.
4. **Post-process** (client-side): write per-centroid tag files, DDB centroid tag records, CSV blocks for fast ID lookup, initialize DDB auto-indexer state and ID tracker, optionally update Lambda threshold.

A parquet build (`initialize-database --format parquet`) has no Preprocess or Distribute stage: the client plans the blocks from the file footers, and the Build stage runs one function per block. It writes no tags, CSV blocks or auto-indexer state, and seeds the ID tracker before the build.

| File | Description |
|------|-------------|
| `indexator.py` | `initialize_database()` — drives the full indexing pipeline via Lithops; `initialize_from_plan()` — builds the blocks of a parquet plan, one function per block; `check_payload()` — refuses a plan whose task arguments exceed `lithops.data_limit` |
| `planner.py` | `plan()` — splits the rows of parquet files into exactly `num_index` contiguous blocks, from their footers; the id of a row is its position across the files in sorted path order |
| `prepare.py` | `expand_sources()` — the files under a directory or `s3://` prefix whose name matches a pattern (default `*.parquet`); `prepare_build()` — reads the footers, plans the blocks and returns the plan with the index config to save; `check_ephemeral_storage()` — refuses a plan whose largest block does not fit the disk of a function |

Called from `ServerlessVectorDB.indexing()` which is called from `VectorDBClient.index_dataset()`. The parquet build is called from `ServerlessVectorDB.indexing_from_plan()`, which is called from `VectorDBClient.index_parquet_dataset()`.

See also: `vectordb/core/README.md` for the ABC contracts.
