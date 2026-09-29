"""Synthetic parquet sources for the tests."""

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


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


def write_canonical(path, rows, dimension=4, row_group_size=None, seed=1):
    rng = np.random.default_rng(seed)
    vectors = rng.random((rows, dimension)).astype(np.float32)
    table = pa.table(
        {
            "id": pa.array(range(100, 100 + rows), pa.int64()),
            "vector": pa.array([list(v) for v in vectors], pa.list_(pa.float32())),
        }
    )
    pq.write_table(table, path, row_group_size=row_group_size or rows)
    return vectors
