"""Provenance of parquet-built blocks: vector id -> (record_id, chunk_idx).

Each block writes ``idmap/block_{i}.parquet`` beside its ``.ann`` with
the columns ``id``, ``record_id`` and ``chunk_idx``.
:func:`select` is pure and works on the bytes of any number of those
files; callers fetch them with whatever storage client they have.
"""

from __future__ import annotations

import io
from typing import Iterable

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


def idmap_prefix(dataset: str, implementation: str) -> str:
    return f"indexes/{dataset}/{implementation}/idmap/"


def select(parts: Iterable[bytes], ids: Iterable[int]) -> dict[int, tuple[str, int]]:
    """The provenance of ``ids`` found in ``parts`` (idmap file contents).

    Ids that no part holds are simply absent from the result; the caller
    decides whether that is an error.
    """
    wanted = {int(value) for value in ids}
    found: dict[int, tuple[str, int]] = {}
    if not wanted:
        return found
    value_set = pa.array(sorted(wanted), pa.int64())
    for blob in parts:
        if len(found) == len(wanted):
            break  # every id resolved; the remaining parts hold none of them
        table = pq.read_table(io.BytesIO(blob), columns=["id", "record_id", "chunk_idx"])
        hits = table.filter(pc.is_in(table.column("id"), value_set=value_set))
        for vid, record_id, chunk_idx in zip(
            hits.column("id").to_pylist(),
            hits.column("record_id").to_pylist(),
            hits.column("chunk_idx").to_pylist(),
        ):
            found[int(vid)] = (record_id, int(chunk_idx))
    return found
