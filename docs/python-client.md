# Blocks-DB Python Client

## Initialization

```python
from vectordb.client import VectorDBClient

client = VectorDBClient(bucket="your-bucket", region="us-east-1")
# With SQS:
client = VectorDBClient(bucket="your-bucket", region="us-east-1", sqs_queue_url="https://sqs...")
# With a DynamoDB table other than BlocksDB-default:
client = VectorDBClient(bucket="your-bucket", region="us-east-1", dynamodb_table_name="your-table")
# Seconds a build or a query waits for functions that never start, counted from
# the last start or finish (default: the function timeout + 60; 0 waits forever):
client = VectorDBClient(bucket="your-bucket", region="us-east-1", wait_timeout=1200)
```

## Dataset Management

| Method | Description |
|--------|-------------|
| `create_dataset(name, csv_path)` | Upload a local CSV file as a new dataset |
| `delete_dataset(name)` | Delete dataset and all its data from S3 and DynamoDB, including the parquet copies a build uploaded (never a source read in place) |
| `list_datasets()` | List all datasets in the bucket |
| `save_index_config(name, config)` | Save index configuration to S3 |
| `delete_index_configs(name)` | Delete all saved config.json files for a dataset |
| `list_indexes(name)` | List available index configs for a dataset |

## Putting Vectors

| Method | Description |
|--------|-------------|
| `put_vectors(dataset_name, vectors, tags, per_vector_tags)` | Add vectors to pending storage (batch); refused on a parquet-built index, which is immutable |
| `put_vector(dataset_name, vector_id, vector, tags, per_vector_tags)` | Add a single vector to pending storage; refused on a parquet-built index |
| `get_pending_vectors(dataset_name)` | Get all pending (unindexed) vectors |
| `has_pending_vectors(dataset_name)` | Check if there are pending vectors |
| `mark_vectors_indexed(dataset_name, indexed_ids)` | Mark vectors as indexed |

## Querying

| Method | Description |
|--------|-------------|
| `query(dataset_name, vector, k, hybrid, batch_size, filter_tags, filter_mode)` | Single vector query (hybrid by default) |
| `query_batch(dataset_name, vectors, k, hybrid, batch_size, filter_tags, filter_mode)` | Multi-vector query; `filter_tags` is refused on a parquet-built index |
| `query_hybrid(dataset_name, vectors, k, batch_size, filter_tags, filter_mode)` | Explicit hybrid query (alias for `query_batch` with `hybrid=True`) |
| `query_indexed_only(dataset_name, vector, vectors, k, batch_size, filter_tags, filter_mode)` | Query only the FAISS index (skip pending) |
| `query_from_file(dataset_name, csv_path, hybrid, k, batch_size, filter_tags, filter_mode)` | Query all vectors from a CSV file |
| `get_vector_ids_by_tags(dataset_name, filter_tags, limit)` | Get vector IDs matching ALL filter tags (refused on a parquet-built index, which has no tags) |
| `get_vectors(dataset_name, ids)` | Get vectors by their IDs (refused on a parquet-built index; use `provenance`) |
| `list_vectors(dataset_name, limit)` | List first N vectors (refused on a parquet-built index) |
| `list_vectors_paginated(dataset_name, start, limit)` | List vectors with pagination (refused on a parquet-built index) |

## Index Management

| Method | Description |
|--------|-------------|
| `index_dataset(dataset_name, config, num_workers, save_config, track_indexed, setup_auto_indexer, csv_blocks)` | Run full indexing pipeline |
| `index_parquet_dataset(dataset_name, sources, config, save_config, replace)` | Build an immutable index from parquet sources (see the parquet section of the README); requires `features`, `num_index` and `k` in the config; refuses a name that already holds an index unless `replace=True`, which deletes it first |
| `provenance(dataset_name, ids, implementation)` | Map result ids back to `(record_id, chunk_idx)` for a parquet-built index |
| `reindex_pending(dataset_name, config, num_workers)` | Rebuild all indexes with pending vectors included (refused on a parquet-built index, before anything is deleted) |
| `index_pending_separate(dataset_name, config)` | Mark pending vectors as indexed without rebuilding |
| `get_indexed_ids(dataset_name)` | Get all indexed vector IDs |
| `get_indexed_count(dataset_name)` | Get count of indexed vectors from DynamoDB counter |
| `is_vector_indexed(dataset_name, vector_id)` | Check if a specific vector is indexed |

## Credentials

| Method | Description |
|--------|-------------|
| `refresh_credentials()` | Refresh AWS credentials in Lithops config from `~/.aws/credentials` |

## Query Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `k` | `int` | `10` | Number of results per query |
| `hybrid` | `bool` | `True` | Include pending vectors in search |
| `batch_size` | `int` | `None` | Override `query_batch_size` (centroid .ann per map worker) |
| `filter_tags` | `dict` | `None` | Tag filters (e.g. `{"source": "web"}`) |
| `filter_mode` | `str` | `"post"` | `"post"` (overfetch + loop) or `"pre"` (reverse-index + IDSelector) |
| `auto_index` | `bool` | `False` | Trigger auto-indexing if threshold reached |
| `tags` | `dict` | `None` | Batch-level tags for `put_vectors` |
| `per_vector_tags` | `list[dict]` | `None` | Per-vector tags (3rd CSV column) |

## Errors

| Exception | Raised when |
|-----------|-------------|
| `NoIndex` (a `ValueError`) | A query names a dataset that has no index; a hybrid query raises it only when there are no pending vectors either |
| `QueryMismatch` (a `ValueError`) | The query vectors do not have the index's dimension, the batch is empty, or `batch_size` is below 1 |
| `FunctionsTimedOut` (a `TimeoutError`) | Functions of a build or a query never started, and none started or finished within the wait timeout |

## Examples

### Create dataset and index

```python
client.create_dataset("mydata", "vectors.csv")
config = {
    "features": 96,
    "num_index": 16,
    "k": 512,
    "n_probe": 32,
    "implementation": "blocks",
}
client.index_dataset("mydata", config, num_workers=16)
```

### Query with tag filter

```python
results, times = client.query(
    "mydata",
    [0.1, 0.2, 0.3],
    k=10,
    filter_tags={"source": "web"},
    filter_mode="pre",
)
```

### Add vectors with tags

```python
client.put_vectors(
    "mydata",
    [(1, [0.1, 0.2]), (2, [0.3, 0.4])],
    tags={"source": "ingest"},
)
```
