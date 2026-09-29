"""The parquet reader: dialects, footers, slices, rejected rows."""

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from helpers import write_canonical, write_owi
from vectordb.utils.parquet import (
    CANONICAL,
    OWI_V2,
    EmptyParquetFile,
    ParquetSourceError,
    detect_dialect,
    inspect,
    iter_ranges,
    read_rows,
)


class TestInspect:
    def test_owi_file_footer(self, owi_file):
        uri, _ = owi_file
        info = inspect(uri)
        assert info.dialect == OWI_V2
        assert info.dimension == 4
        assert info.row_groups == (4, 4, 2)
        assert info.num_rows == 10

    def test_canonical_file_footer(self, tmp_path):
        path = tmp_path / "vectors.parquet"
        write_canonical(path, rows=6, dimension=3, row_group_size=6)
        info = inspect(str(path))
        assert info.dialect == CANONICAL
        assert info.dimension == 3
        assert info.row_groups == (6,)

    def test_unknown_columns_are_named(self, tmp_path):
        path = tmp_path / "other.parquet"
        pq.write_table(pa.table({"a": [1], "b": [2]}), path)
        # among the files of a directory source, the message must say which one
        with pytest.raises(ParquetSourceError, match=r"other\.parquet: no known vector dialect.*'a', 'b'"):
            inspect(str(path))

    def test_vector_column_must_be_a_list(self, tmp_path):
        path = tmp_path / "scalar.parquet"
        pq.write_table(pa.table({"id": [1], "vector": [1.0]}), path)
        with pytest.raises(ParquetSourceError, match="not a list"):
            inspect(str(path))

    @pytest.mark.parametrize("values, kind", [
        ([["1.0", "2.0"]], "string"),
        ([[True, False]], "bool"),
    ], ids=["digits as text", "booleans"])
    def test_the_values_of_the_list_must_be_numbers(self, tmp_path, values, kind):
        # the decoder casts to float32: without this check text with digits
        # would be indexed as coordinates, and text without them would fail
        # with an error that names no file
        path = tmp_path / "odd.parquet"
        pq.write_table(pa.table({"id": [1], "vector": values}), path)
        with pytest.raises(ParquetSourceError, match=f"holds {kind} values, not numbers"):
            inspect(str(path))

    def test_integer_vectors_are_still_read(self, tmp_path):
        path = tmp_path / "ints.parquet"
        pq.write_table(pa.table({"id": [1], "vector": [[1, 2, 3]]}), path)
        assert inspect(str(path)).dimension == 3

    def test_missing_file_is_a_source_error(self, tmp_path):
        with pytest.raises(ParquetSourceError, match="cannot read"):
            inspect(str(tmp_path / "absent.parquet"))

    def test_detect_dialect_prefers_a_full_match(self):
        assert detect_dialect(["record_id", "chunk_idx", "embedding"]) == OWI_V2
        assert detect_dialect(["id", "vector", "tags"]) == CANONICAL
        with pytest.raises(ParquetSourceError):
            detect_dialect(["id", "embedding"])


class TestReadRows:
    def test_owi_rows_decode_to_float32_with_provenance(self, owi_file):
        uri, vectors = owi_file
        rows = read_rows(uri, row_group=1, dimension=4)
        assert rows.vectors.dtype == np.float32 and rows.vectors.shape == (4, 4)
        np.testing.assert_allclose(rows.vectors, vectors[4:8], rtol=1e-3)
        assert rows.record_ids == ["doc-2", "doc-2", "doc-3", "doc-3"]
        assert rows.chunk_idx.tolist() == [0, 1, 0, 1]
        assert rows.positions.tolist() == [0, 1, 2, 3]
        assert rows.rejected == 0 and rows.ids is None

    def test_slice_of_a_row_group(self, owi_file):
        uri, vectors = owi_file
        rows = read_rows(uri, row_group=0, dimension=4, start=1, end=3)
        np.testing.assert_allclose(rows.vectors, vectors[1:3], rtol=1e-3)
        assert rows.positions.tolist() == [0, 1]

    def test_slice_outside_the_row_group_is_refused(self, owi_file):
        uri, _ = owi_file
        with pytest.raises(ParquetSourceError, match=r"rows \[3, 9\) outside"):
            read_rows(uri, row_group=0, dimension=4, start=3, end=9)

    def test_wrong_length_vectors_are_rejected_and_positions_kept(self, tmp_path):
        path = tmp_path / "bad.parquet"
        vectors = write_owi(path, rows=5, bad_rows={1, 3})
        rows = read_rows(str(path), row_group=0, dimension=4)
        assert rows.rejected == 2
        assert rows.positions.tolist() == [0, 2, 4]
        np.testing.assert_allclose(rows.vectors, vectors[[0, 2, 4]], rtol=1e-3)
        assert rows.record_ids == ["doc-0", "doc-1", "doc-2"]

    def test_owi_vectors_come_back_at_unit_length(self, tmp_path):
        # float16 storage moves the published unit vectors off length 1,
        # and then L2 no longer ranks like the cosine
        path = tmp_path / "scaled.parquet"
        pq.write_table(
            pa.table({
                "record_id": ["a", "b"],
                "chunk_idx": pa.array([0, 0], pa.int32()),
                "embedding": pa.array([[3.0, 4.0], [0.0, 0.5]], pa.list_(pa.float16())),
            }),
            path,
        )
        rows = read_rows(str(path), 0, 2)
        np.testing.assert_allclose(rows.vectors, [[0.6, 0.8], [0.0, 1.0]], rtol=1e-6)

    def test_owi_rows_that_cannot_be_scaled_are_rejected_and_positions_kept(self, tmp_path):
        path = tmp_path / "unscalable.parquet"
        pq.write_table(
            pa.table({
                "record_id": ["a", "b", "c", "d", "e"],
                "chunk_idx": pa.array([0, 1, 2, 3, 4], pa.int32()),
                "embedding": pa.array(
                    [[1.0, 0.0], [0.0, 0.0], [float("nan"), 1.0], [70000.0, 1.0], [0.0, 2.0]],
                    pa.list_(pa.float16()),  # 70000 overflows float16 to infinity
                ),
            }),
            path,
        )
        rows = read_rows(str(path), 0, 2)
        assert rows.rejected == 3
        assert rows.positions.tolist() == [0, 4]
        assert rows.record_ids == ["a", "e"] and rows.chunk_idx.tolist() == [0, 4]
        np.testing.assert_allclose(rows.vectors, [[1.0, 0.0], [0.0, 1.0]])

    def test_canonical_rows_keep_the_file_ids(self, tmp_path):
        path = tmp_path / "vectors.parquet"
        vectors = write_canonical(path, rows=6, dimension=3, row_group_size=3)
        rows = read_rows(str(path), row_group=1, dimension=3)
        assert rows.ids.tolist() == [103, 104, 105]
        np.testing.assert_allclose(rows.vectors, vectors[3:6])
        assert rows.record_ids is None


class TestDialectAndTyping:
    def test_a_file_matching_both_dialects_is_refused(self, tmp_path):
        path = tmp_path / "both.parquet"
        pq.write_table(
            pa.table({
                "id": pa.array([1], pa.int64()),
                "vector": pa.array([[1.0, 2.0]], pa.list_(pa.float32())),
                "record_id": ["a"],
                "chunk_idx": pa.array([0], pa.int32()),
                "embedding": pa.array([[1.0, 2.0]], pa.list_(pa.float16())),
            }),
            path,
        )
        with pytest.raises(ParquetSourceError, match="ambiguous dialect"):
            inspect(str(path))

    def test_a_null_id_next_to_a_good_vector_is_refused_by_name(self, tmp_path):
        path = tmp_path / "nullid.parquet"
        pq.write_table(
            pa.table({
                "record_id": ["a", None],
                "chunk_idx": pa.array([0, 1], pa.int32()),
                "embedding": pa.array([[1.0, 2.0], [3.0, 4.0]], pa.list_(pa.float16())),
            }),
            path,
        )
        with pytest.raises(ParquetSourceError, match="column 'record_id' has 1 null"):
            read_rows(str(path), 0, 2)

    def test_a_null_chunk_index_is_refused_rather_than_turned_into_a_sentinel(self, tmp_path):
        path = tmp_path / "nullidx.parquet"
        pq.write_table(
            pa.table({
                "record_id": ["a", "b"],
                "chunk_idx": pa.array([0, None], pa.int32()),
                "embedding": pa.array([[1.0, 2.0], [3.0, 4.0]], pa.list_(pa.float16())),
            }),
            path,
        )
        with pytest.raises(ParquetSourceError, match="column 'chunk_idx' has 1 null"):
            read_rows(str(path), 0, 2)

    def test_a_null_vector_is_a_rejected_row_not_a_failure(self, tmp_path):
        path = tmp_path / "nullvec.parquet"
        pq.write_table(
            pa.table({
                "record_id": ["a", "b"],
                "chunk_idx": pa.array([0, 1], pa.int32()),
                "embedding": pa.array([[1.0, 2.0], None], pa.list_(pa.float16())),
            }),
            path,
        )
        rows = read_rows(str(path), 0, 2)
        assert rows.kept == 1 and rows.rejected == 1 and rows.positions.tolist() == [0]

    def test_an_empty_file_is_named_as_empty(self, tmp_path):
        path = tmp_path / "empty.parquet"
        pq.write_table(
            pa.table({
                "record_id": pa.array([], pa.string()),
                "chunk_idx": pa.array([], pa.int32()),
                "embedding": pa.array([], pa.list_(pa.float16())),
            }),
            path,
        )
        with pytest.raises(EmptyParquetFile, match="no rows"):
            inspect(str(path))


class TestBatchedReads:
    def test_whole_row_groups_read_together_decode_exactly_as_one_by_one(self, tmp_path):
        path = tmp_path / "many.parquet"
        vectors = write_owi(path, rows=12, row_group_size=3)  # four row groups
        ranges = [(0, 0, None), (1, 0, None), (2, 0, None), (3, 0, None)]
        batched = list(iter_ranges(str(path), 4, ranges))
        one_by_one = [read_rows(str(path), rg, 4) for rg in range(4)]
        assert len(batched) == 4
        for got, want, start in zip(batched, one_by_one, range(0, 12, 3)):
            np.testing.assert_array_equal(got.vectors, want.vectors)
            assert got.record_ids == want.record_ids
            assert got.chunk_idx.tolist() == want.chunk_idx.tolist()
            np.testing.assert_allclose(got.vectors, vectors[start:start + 3], rtol=1e-3)

    def test_a_sliced_range_between_whole_ones_keeps_its_place(self, tmp_path):
        path = tmp_path / "mixed.parquet"
        vectors = write_owi(path, rows=12, row_group_size=4)
        ranges = [(0, 0, None), (1, 1, 3), (2, 0, None)]
        rows = list(iter_ranges(str(path), 4, ranges))
        assert [r.kept for r in rows] == [4, 2, 4]
        np.testing.assert_allclose(rows[1].vectors, vectors[5:7], rtol=1e-3)
