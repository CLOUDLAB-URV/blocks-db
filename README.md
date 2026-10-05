<div align="center">

# Blocks-DB: Serverless Vector Database

<p>
  <img src="https://img.shields.io/badge/🐍Python-3.10-4ecdc4?style=for-the-badge&logo=python&logoColor=white">
  <img src="https://img.shields.io/github/stars/Oct-HI/blocks-db-wip?style=for-the-badge&logo=github&logoColor=white">
  <a href="https://doi.org/10.1145/3769769"><img src="https://img.shields.io/badge/📄Paper-10.1145/3769769-ff6b6b?style=for-the-badge"></a>
</p>

<div align="center" style="margin: 30px 0;">
  <img src="./README.assets/blocks-db-scheme.png" alt="Blocks-DB Scheme" style="max-width: 100%; height: auto;">
</div>

</div>

---

Blocks-DB is a modular serverless vector database built on Lithops and AWS Lambda. It supports block-based indexing with FAISS, distributed querying, and a simple CLI or Python client interface.

---

## 📋 Requirements

- **Python 3.10** (venv use recommended)
- **Docker** & **Docker Hub account**
- **AWS account** with access to: S3, Lambda, DynamoDB, ECR, IAM
- **Existing S3 bucket** for storing vectors and indexes

---

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/Oct-HI/blocks-db-wip
cd blocks-db

# (Optional) Create virtual environment
python -m venv venv
source venv/bin/activate

# Install the package
pip install .
```

### Development

```bash
# Install with the test runner
pip install -e ".[dev]"

# Run the unit tests: they use fakes and a local temporary directory, no AWS account needed
pytest tests/unit

# Run the integration tests: Lithops on this machine, no AWS account needed
pytest tests/integration
```

---

## 🚀 Quickstart (minimal workflow)

```bash
# 1. Save default bucket and region
blocks-db configure --bucket my-bucket --region us-east-1

# 2. Create AWS infrastructure (Lambda, DynamoDB, ECR, S3 triggers)
blocks-db setup --bucket my-bucket

# 3. Index a dataset (needs an index config JSON — see below)
blocks-db initialize-database my-dataset vectors.csv --config config.json

# 4. Add more vectors incrementally
blocks-db put my-dataset new_vectors.csv

# 5. Search (hybrid: index + pending by default)
blocks-db query my-dataset --file queries.csv --k 10

# 6. Check status
blocks-db status my-dataset -v
```

Each command is explained in detail below.

---

## ⚙️ Initial Setup

### 1. Configure AWS credentials

> **Recommended:** Configure AWS credentials using `~/.aws/credentials` file:
> ```bash
> aws configure
> ```

This is the recommended way. Blocks-DB will automatically read credentials from your AWS config.

Alternatively, you can set environment variables:
```bash
export AWS_ACCESS_KEY_ID=your-key
export AWS_SECRET_ACCESS_KEY=your-secret
export AWS_DEFAULT_REGION=us-east-1
```

### 2. Save default bucket and region

```bash
blocks-db configure --bucket your-s3-bucket --region us-east-1
```

For SQS-based auto-indexer:
```bash
blocks-db configure --bucket your-s3-bucket --region us-east-1 --sqs
```

This saves configuration to `~/.blocks-db-config/backend_config.json`.

The client keeps its tracking state in the DynamoDB table `BlocksDB-default`.
When several deployments share one AWS account, give each one its own table:
`--table-name` on `configure` saves it, and the global `--table-name` flag or
the `SVDB_DYNAMODB_TABLE` environment variable override it for one command.
`setup --table-name` also saves the name it creates.

A build or a query gives up when some functions never start and none has
started or finished for the function timeout plus a minute (see
`vectordb/README.md`, Common Pitfalls). To change that window:
`--wait-timeout` on `configure` saves it, and the global `--wait-timeout`
flag or the `SVDB_WAIT_TIMEOUT` environment variable override it for one
command; `0` waits forever. `setup` keeps a saved value, and `configure` without
`--wait-timeout` drops it.

---

## 🚦 Auto-Indexer Modes

Blocks-DB supports three auto-indexer configurations, selected at `setup`:

| Mode | `setup` flag | `configure` flag | Use case |
|------|:------------:|:----------------:|----------|
| **S3 Triggers** (default) | *(none)* | *(none)* | Standard S3 buckets with notification support |
| **SQS** | `--sqs` | `--sqs` | Buckets without S3 notification support, or when you prefer queue-based triggers |
| **S3 Express One Zone** | `--s3express` | *(auto-detected)* | S3 Express One Zone buckets (name ends in `--x-s3`). Auto-enables SQS since Express buckets don't support S3 notifications |

### How it works

- **S3 Triggers**: Uploading a CSV to `pending/<dataset>/` fires a Lambda notification directly.
- **SQS**: The CLI sends a message to an SQS queue after each `put`. Lambda polls the queue. Includes a DLQ for failed messages.
- **S3 Express**: Same as SQS, plus adds `s3express:CreateSession` permission to the Lambda role and auto-detects the availability zone from the bucket name.

---

## 🏃 Quick Start

### Setup: Create infrastructure

```bash
# Default — S3 Triggers
blocks-db setup --bucket your-s3-bucket

# SQS mode
blocks-db setup --bucket your-s3-bucket --sqs

# S3 Express One Zone mode (auto-enables SQS)
blocks-db setup --bucket your-bucket--use1-az6--x-s3 --s3express
```

Creates:
- Lambda Layer with FAISS
- Lambda function for auto-indexing
- DynamoDB table for index tracking
- S3 triggers **or** SQS queue (+ DLQ) for auto-indexing
- Lithops runtime in ECR

**Customize names:**

```bash
blocks-db setup --bucket your-s3-bucket \
  --runtime-name mi-runtime \
  --function-name mi-autoindexer \
  --table-name mi-tabla \
  --layer-name mi-layer \
  --role-name mi-rol
```

**Other options:**
```bash
# Custom threshold (bytes) for auto-indexer block size
blocks-db setup --bucket your-bucket --threshold 10485760

# Skip DynamoDB table or Lithops runtime (if already built)
blocks-db setup --bucket your-bucket --skip-vector-table --skip-runtime
```

> **Warning:** Ensure these resources do not already exist, or the setup will fail or update existing resources.

**Rebuild the runtime:**

The runtime image holds a copy of the `vectordb` package and its dependencies, `pyarrow` among them. When the package changes, build the image again and delete the deployed functions, which are created from the new image on next use:

```bash
# From the repository root: build the image and push it to ECR. Lambda refuses an
# image that carries the attestation manifests Docker adds by default, so turn them off
BUILDX_NO_DEFAULT_ATTESTATIONS=1 lithops runtime build -f vectordb/infra/Dockerfile.lambda -b aws_lambda blocks-db-runtime

# Delete the deployed functions
lithops runtime delete blocks-db-runtime -b aws_lambda
```

`blocks-db-runtime` is the name `setup` uses unless `--runtime-name` sets another. A new name works too: build under it and set it as `runtime` in the `aws_lambda` section of the Lithops configuration (`~/.lithops/config`, written by `setup`).

### Initialize database

```bash
blocks-db initialize-database mydataset vectors.csv --config config.json --workers 16
```

This requires an **index config file**. Example (`config.json`):

```json
{
  "features": 96,
  "num_vectors": -1,
  "k_search": 10,
  "k_result": 10,
  "skip_init": false,
  "skip_kmeans": false,
  "kmeans_version": "unbalanced",
  "implementation": "blocks",

  "replication": 1.0,
  "num_index": 16,
  "num_centroids_search": 4,
  "k": 512,
  "n_probe": 32,
  "query_batch_size": 4,

  "index_mem": 10240,
  "search_map_mem": 8192,
  "search_reduce_mem": 2048
}
```

**Parameters:**

| Parameter | Description |
|-----------|-------------|
| `features` | Vector dimensionality |
| `implementation` | "blocks" (default) — block-based indexing |
| `num_index` | Number of index blocks (default: 4) |
| `k` | FAISS IVF k (default: 4096) |
| `n_probe` | FAISS IVF n_probe (default: 1024) |
| `index_mem` | Index Lambda memory in MB (default: 8192) |
| `search_map_mem` | Search map Lambda memory in MB (default: 9216) |
| `search_reduce_mem` | Search reduce Lambda memory in MB (default: 2048) |
| `replication` | Replication factor |
| `num_vectors` | Total vectors in dataset (-1 = auto-detect) |
| `k_search` | K for search calculation (default: 5) |
| `k_result` | K for final results (default: 5) |
| `query_batch_size` | Query batch size (default: 16) |
| `num_centroids_search` | Centroids to search |
| `skip_init` | Skip initialization |
| `skip_kmeans` | Skip k-means clustering |
| `kmeans_version` | K-means version |

**Options:**

```bash
# Skip auto-update of Lambda threshold
blocks-db initialize-database mydataset vectors.csv --config config.json --no-update-threshold

# Build csv_blocks from local file (skip S3 re-download during tracking)
blocks-db initialize-database mydataset vectors.csv --config config.json --build-local

# Custom CSV block size for optimized vector reads (gets)
# Auto-calculated from index config if not set
blocks-db initialize-database mydataset vectors.csv --config config.json --csv-block-size 1000000

# Skip DynamoDB state init and vector tracking (benchmark purity)
blocks-db initialize-database mydataset vectors.csv --config config.json --skip-auto-indexer
```

After initializing, the threshold is automatically configured based on the initial config (num_index, features) and estimated vector size. This threshold controls the size of each block for auto-indexing — the system tries to get as close as possible to this size.

**From parquet files:**

```bash
# One file
blocks-db initialize-database mydata vectors.parquet --format parquet --config config.json

# Every *.parquet file under a directory, at any depth
blocks-db initialize-database mydata /path/to/parquet --format parquet --config config.json

# Every *.parquet object under an S3 prefix (without the final "/", the URI is one object)
blocks-db initialize-database mydata s3://my-bucket/path/to/parquet/ --format parquet --config config.json
```

The files must use one of the layouts in [Parquet vectors](#parquet-vectors). Local files are uploaded to the bucket under `datasets/<dataset>/source/`, and `delete-dataset` removes them; `s3://` files are read in place and left untouched. The functions read them with their execution role, so a source in another bucket must be readable by that role.

Example (`config.json`):

```json
{
  "features": 1024,
  "num_index": 16,
  "k": 512,
  "n_probe": 32,

  "index_mem": 10240,
  "search_map_mem": 8192,
  "search_reduce_mem": 2048
}
```

`features`, `num_index` and `k` are required. When `k` is missing, the error suggests a value from the rows of the smallest block: 4·√rows, at most rows / 39.

**Parquet options:**

```bash
# Under a directory or an S3 prefix, read only the files whose name matches a shell pattern (default: *.parquet)
blocks-db initialize-database mydata /path/to/parquet --format parquet --config config.json --files '*_embeddings.parquet'

# Delete the index already stored under this name, then build
blocks-db initialize-database mydata /path/to/parquet --format parquet --config config.json --replace
```

`--workers`, `--build-local`, `--csv-block-size`, `--skip-auto-indexer` and `--no-update-threshold` belong to a CSV build and are refused with `--format parquet`; `--files` and `--replace` are refused without it.

Without `--replace`, a name that already holds an index is refused. A name that holds a CSV dataset (its `source.csv` or pending vectors) is refused even with `--replace`, since its queries would still search the pending vectors: remove it with `delete-dataset` first. A build that fails or is interrupted while its functions run removes the blocks written so far; with `--replace`, the previous index is already deleted by then. Functions still running on AWS Lambda finish and may write their blocks afterwards; `--replace` or `delete-dataset` removes them.

**Checks before the build:**

The build plans exactly `num_index` blocks from the file footers and runs one function per block. It checks the plan before it deletes, uploads or writes anything, and two of the checks depend on the Lithops configuration:

- The largest block must fit the disk of a function: on AWS Lambda, `ephemeral_storage` in the `aws_lambda` section (in MB: 512 unless set, at most 10240). Raise it, or raise `num_index`. A function keeps the size it was created with, so after a change delete the runtime (`lithops runtime delete <runtime-name> -b aws_lambda`).
- The arguments of all the build tasks, together, must stay under `data_limit` in the `lithops` section (in MiB: 4 unless set). Raise it, or build from fewer files.

The first write of the build is the id counter of the dataset in the DynamoDB table; if the table refuses it, the build stops there. The functions run the `vectordb` package of the runtime image, so rebuild the image before the first parquet build on an existing setup (see **Rebuild the runtime** above).

**What a parquet index does not support:**

An index built from parquet is immutable and has no tags. These stop with an error that says why:

- `put`
- `get` by id or with `--limit`
- `get-by-tags`
- `query --filter`
- `initialize-database` without `--format parquet` over the same name
- `reindex_pending()` in the Python client

To go from a result id to its source record, use `provenance()` in the Python client (see [Python Client](#-python-client)):

```python
client.provenance("mydata", [12, 4096])   # {12: ("doc-3", 1), 4096: ("doc-987", 0)}
```

### Add more vectors

```bash
blocks-db put mydataset new_vectors.csv
```

Vectors are stored as "pending" and included in searches automatically.

**With metadata tags:**
```bash
blocks-db put mydataset new_vectors.csv --tags '{"source":"web","category":"news"}'
```

Tags are stored per batch and can be used to filter searches later. Each vector in the batch inherits the same tags.

**Per-vector tags (3rd CSV column):**

Alternatively, each row in the CSV can carry its own tags as a JSON third
column. This is more granular than batch-level `--tags`:

```
1,0.1 0.2 ...,{"source":"web"}
2,0.4 0.5 ...,"{""source"":""api"",""priority"":""low""}"
```

The format is described under [File Formats](#-file-formats).

When `--tags` and per-vector tags are both present, per-vector tags take
precedence.

**Single-vector mode:**
```bash
blocks-db put mydataset single_vector.csv --single
```

### Update threshold (after initialize-database)

If you used `--no-update-threshold` or want to adjust manually:

```bash
# Manual value
blocks-db update-threshold 10485760

# Auto-calculate from dataset config
blocks-db update-threshold --dataset mydataset --bucket your-bucket
```

### Query

```bash
# Single vector
blocks-db query mydataset --vector "0.1 0.2 0.3 ..."

# From CSV file
blocks-db query mydataset --file queries.csv --k 10
```

By default searches in index + pending. For index-only search:
```bash
blocks-db query mydataset --file queries.csv --indexed-only
```

A query that cannot be answered ends with a message and exit code 1 instead of
an empty result: a dataset with no index and no pending vectors (with
`--indexed-only`, no index at all), a query file with no vectors, a vector
whose dimension is not the index's, or a `--batch-size` below 1.

**Filtered search (requires tags on put):**
```bash
# Only search centroids and pending files matching ALL specified tags
blocks-db query mydataset --file queries.csv --k 10 --filter '{"source":"web"}'
```

**Filter modes:**

| Mode | Flag | Behavior |
|------|------|----------|
| Post-filter (default) | `--filter-mode post` | Overfetches k×2 per centroid, discards non-matching. Fast, default. |
| Pre-filter | `--filter-mode pre` | Uses reverse index per centroid to restrict FAISS search to only matching IDs. May find results post-filter misses. |

Post-filter is the default and performs well for most use cases. Pre-filter
may find more results since it doesn't rely on overfetching, at the cost of
loading a reverse index per centroid.

**Get vectors by tags:**

```bash
blocks-db get-by-tags mydataset --filter '{"priority":"high"}' --limit 20
```

Lists vector IDs whose tags match the filter. Useful for inspection.

**Custom batch size** (number of centroid `.ann` files per map worker):
```bash
blocks-db query mydataset --file queries.csv --batch-size 2
```

### Status

```bash
blocks-db status mydataset

# With details
blocks-db status mydataset -v
```

For an index built from parquet, `status` shows the source format, the
vectors kept and the rows rejected, with no pending section; `-v` adds the
number of blocks and source files.

---

## 📖 Commands Reference

| Command | Description | Usage |
|---------|-------------|-------|
| `setup` | Create infrastructure (Lambda, DynamoDB, SQS/S3 triggers) | `blocks-db setup --bucket <b> [--sqs \| --s3express]` |
| `configure` | Save default bucket, region and DynamoDB table | `blocks-db configure --bucket <b> --region <r> [--sqs] [--table-name <t>]` |
| `refresh-credentials` | Refresh AWS credentials in Lithops config | `blocks-db refresh-credentials` |
| `update-threshold` | Update auto-indexer block size threshold | `blocks-db update-threshold [bytes] --dataset <name>` |
| `initialize-database` | Build the initial index from CSV or parquet | `blocks-db initialize-database <n> <src> --config <j> [--build-local \| --format parquet [--files <p>] [--replace]]` |
| `put` | Add vectors to pending storage | `blocks-db put <name> <csv> [--tags <json>] [--single]` |
| `query` | Search vectors (indexed + pending by default) | `blocks-db query <n> --file <csv> --k <N> [--filter <j>] [--filter-mode post\|pre] [--batch-size <N>]` |
| `status` | Show dataset status and index info | `blocks-db status <name> [-v]` |
| `get` | Retrieve vectors by ID, list vectors, or show pending | `blocks-db get <n> <id>... [--limit N] [--pending]` |
| `get-by-tags` | List vector IDs matching tags | `blocks-db get-by-tags <n> --filter <json> [--limit N]` |
| `delete-dataset` | Delete dataset and all its data | `blocks-db delete-dataset <n> [--yes]` |

### `get` command usage

```bash
# Get specific vectors by ID
blocks-db get mydataset 1 2 3

# List first N vectors from dataset
blocks-db get mydataset --limit 100

# Show pending vectors
blocks-db get mydataset --pending
```

---

## 📄 File Formats

### Vectors CSV

The ID (integer), a comma, then the values separated by spaces.

```
1,0.1 0.2 0.3 ...
2,0.4 0.5 0.6 ...
```

### Vectors CSV with Per-Vector Tags

Add a third column with a JSON object for per-vector tags. A JSON object
with more than one key holds commas, so it is quoted as a CSV field, with
its double quotes doubled:

```
1,0.1 0.2 ...,{"source":"web"}
2,0.4 0.5 ...,"{""source"":""api"",""priority"":""low""}"
```

When the third column is present, `initialize-database` and the auto-indexer
Lambda store the tags alongside each vector in the index. A vector without a
third column, put without `--tags`, has no tags: a query with `--filter` on an
indexed dataset does not return it, unless the filter matches no block and no
pending file; the query then searches every pending vector without the filter.

### Queries CSV

The file of `query --file` holds one query per line: its values separated by
spaces, with no ID.

```
0.1 0.2 0.3 ...
0.4 0.5 0.6 ...
```

### Parquet vectors

A parquet build recognizes two column layouts by their column names; any
other column is ignored:

| Layout | Columns |
|--------|---------|
| canonical | `vector` (list of numbers), optional `id` (integer) |
| owi-v2 | `record_id` (string), `chunk_idx` (integer), `embedding` (list of float16) |

owi-v2 is the layout of the embeddings files the Open Web Index publishes. The
records files published beside them hold no vectors, so a directory of both is
built with `--files '*_embeddings.parquet'`.

All the files of a build must use the same layout, and the dimension of each
file (from its schema, or from its first vector) must be `features`. A row
whose vector has another length, or no vector, is rejected, and so is an
owi-v2 row that cannot be scaled to unit length (all zeros, or a non-finite
value).

A file with no rows is skipped. A file with rows whose columns match neither
layout, or both, stops the build with an error that names it. The
configuration saved with the index (`indexes/<dataset>/blocks/config.json`)
counts the rejected rows in `rejected` and the skipped files in
`source_files_skipped`.

The id of a vector in the index is the position of its row, counted from 0
across the files in sorted path order, whatever the `id` column of a canonical
file holds; rejected rows leave gaps in the ids.

`provenance()` maps an id back to `(record_id, chunk_idx)`: for an owi-v2 row,
its own columns; for a canonical row, its `id` as text (the index id when the
file has no `id` column) and 0.

owi-v2 vectors are scaled to unit length when read, and the saved
configuration records `unit_norm: true`: for a unit-length query, the
returned distance is 2 − 2·cosine. Canonical vectors are indexed as written, so
their values must be finite: a NaN or infinite value makes FAISS refuse to
train its block, and the build fails.

---

## 🐍 Python Client

```python
from blocks_db.client import VectorDBClient

client = VectorDBClient(bucket="your-bucket", region="us-east-1")

# Query
vector = [0.1, 0.2, ...]
results, times = client.query("mydataset", vector)
```

> 📖 Full API reference: [docs/python-client.md](docs/python-client.md)

---

## 🔐 AWS Permissions

Blocks-DB needs the following permissions (created automatically by `setup`):

- **S3**: read/write in your bucket
- **Lambda**: create functions, layers
- **DynamoDB**: create table and write
- **ECR**: push images
- **IAM**: roles for Lambda
- **SQS** (if `--sqs` or `--s3express`): create queues, event source mappings
- **S3 Express** (if `--s3express`): `s3express:CreateSession` on the bucket

---

## Architecture

See [`vectordb/README.md`](vectordb/README.md#architecture) for the full architecture
documentation, including project structure, S3 layout, DynamoDB schema, tag system,
filter modes, auto-indexer Lambda, indexing/search pipelines, and common pitfalls.

---

## 📖 Citation

Based on: *Building Stateless Serverless Vector DBs via Block-based Data Partitioning*  
Daniel Barcelona-Pons, Raúl Gracia-Tinedo, Albert Cañadilla-Domingo, Xavier Roca-Canals, Pedro García-López  
Proc. ACM Manag. Data 3, 6 (SIGMOD), Article 304 (December 2025)  
DOI: [10.1145/3769769](https://doi.org/10.1145/3769769)
