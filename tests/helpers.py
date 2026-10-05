"""Synthetic parquet sources for the tests, and a one-range reader."""

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from vectordb.utils.parquet import Rows, iter_ranges


def read_rows(uri, row_group, dimension, start=0, end=None) -> Rows:
    """Decode rows ``[start, end)`` of one row group."""
    (rows,) = list(iter_ranges(uri, dimension, [(row_group, start, end)]))
    return rows


def write_owi(path, rows, dimension=4, row_group_size=None, seed=0, bad_rows=()):
    """An OWI v2.0.0-shaped file: record_id, chunk_idx, embedding (float16).

    ``bad_rows`` are row positions whose vector gets a wrong length.
    Returns what the reader must give back: the float16 vectors in
    float32, scaled to unit length.
    """
    rng = np.random.default_rng(seed)
    vectors = rng.random((rows, dimension)).astype(np.float16)
    embedding = [
        list(vectors[i][: dimension - 1]) if i in bad_rows else list(vectors[i])
        for i in range(rows)
    ]
    table = pa.table(
        {
            "record_id": [f"doc-{i // 2}" for i in range(rows)],
            "chunk_idx": pa.array([i % 2 for i in range(rows)], pa.int32()),
            "embedding": pa.array(embedding, pa.list_(pa.float16())),
        }
    )
    pq.write_table(table, path, row_group_size=row_group_size or rows)
    stored = vectors.astype(np.float32)
    return stored / np.linalg.norm(stored, axis=1, keepdims=True)


def write_canonical(path, rows, dimension=4, row_group_size=None, seed=1, with_ids=True):
    """A canonical file: vector (float32) and, unless ``with_ids`` is
    false, id (int64) counting from 100."""
    rng = np.random.default_rng(seed)
    vectors = rng.random((rows, dimension)).astype(np.float32)
    columns = {"vector": pa.array([list(v) for v in vectors], pa.list_(pa.float32()))}
    if with_ids:
        columns["id"] = pa.array(range(100, 100 + rows), pa.int64())
    table = pa.table(columns)
    pq.write_table(table, path, row_group_size=row_group_size or rows)
    return vectors
