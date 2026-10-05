"""Sealing a parquet build, and reading provenance back from idmap files."""

import io

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from helpers import write_canonical, write_owi
from vectordb.indexing.planner import PlanError
from vectordb.indexing.prepare import check_ephemeral_storage, prepare_build
from vectordb.utils.idmap import idmap_prefix, select


class TestPrepareBuild:
    def test_seals_what_was_read(self, tmp_path):
        a, b = tmp_path / "metadata_1_embeddings.parquet", tmp_path / "metadata_0_embeddings.parquet"
        write_owi(a, rows=6, row_group_size=3)
        write_owi(b, rows=4, row_group_size=2, seed=3)
        result, sealed = prepare_build([str(a), str(b)], {"num_index": 2, "k": 1, "features": 4, "implementation": "blocks"})
        assert result.num_index == 2 and result.total_vectors == 10
        assert sealed["source_format"] == "parquet"
        assert sealed["source_keys"] == [str(b), str(a)]  # sorted: files are read in this order
        assert sealed["source_rows"] == sealed["total_vectors"] == sealed["num_vectors"] == 10
        assert sealed["block_ranges"] == [[0, 0, 3], [1, 4, 9]] or sealed["block_ranges"] == [[0, 0, 5], [1, 6, 9]]
        assert sealed["features"] == 4 and sealed["num_index"] == 2 and sealed["k"] == 1
        assert sealed["implementation"] == "blocks"  # untouched keys travel
        assert sealed["unit_norm"] is True  # owi-v2 vectors are scaled on read

    def test_canonical_vectors_are_sealed_as_not_scaled(self, tmp_path):
        path = tmp_path / "vectors.parquet"
        write_canonical(path, rows=6, dimension=4)
        _, sealed = prepare_build([str(path)], {"num_index": 1, "k": 1, "features": 4, "implementation": "blocks"})
        assert sealed["unit_norm"] is False

    def test_k_must_be_declared_and_the_message_suggests_one(self, tmp_path):
        source = tmp_path / "x.parquet"
        write_owi(source, rows=80, row_group_size=40)
        with pytest.raises(PlanError, match=r"k .* must be declared; the smallest block holds 40 .* suggested k: 1 \(at most 1\)"):
            prepare_build([str(source)], {"num_index": 2, "features": 4})

    def test_num_index_must_be_declared(self, tmp_path):
        source = tmp_path / "x.parquet"
        write_owi(source, rows=8)
        with pytest.raises(PlanError, match="num_index"):
            prepare_build([str(source)], {"k": 1, "features": 4})
        with pytest.raises(PlanError, match="no sources"):
            prepare_build([], {"num_index": 1, "k": 1, "features": 4})

    def test_features_is_checked_against_the_files(self, tmp_path):
        source = tmp_path / "x.parquet"
        write_owi(source, rows=8)
        with pytest.raises(PlanError, match="features is 1024 but the source vectors have 4"):
            prepare_build([str(source)], {"num_index": 1, "k": 1, "features": 1024})

    def test_features_must_be_declared(self, tmp_path):
        source = tmp_path / "x.parquet"
        write_owi(source, rows=8)
        with pytest.raises(PlanError, match="features .* must be declared"):
            prepare_build([str(source)], {"num_index": 1, "k": 1})

    def test_a_source_declared_twice_is_refused(self, tmp_path):
        source = tmp_path / "x.parquet"
        write_owi(source, rows=8)
        # the same rows would be indexed twice under two id ranges
        with pytest.raises(PlanError, match="declared more than once"):
            prepare_build([str(source), str(source)], {"num_index": 2, "k": 1, "features": 4})

    def test_only_the_blocks_implementation_reads_a_plan(self, tmp_path):
        source = tmp_path / "x.parquet"
        write_owi(source, rows=8)
        with pytest.raises(PlanError, match="implementation 'centroids'"):
            prepare_build([str(source)], {"num_index": 1, "k": 1, "features": 4, "implementation": "centroids"})

    def test_an_unknown_key_fails_here_not_after_the_build(self, tmp_path):
        source = tmp_path / "x.parquet"
        write_owi(source, rows=8)
        with pytest.raises(TypeError, match="unexpected keyword argument"):
            prepare_build([str(source)], {"num_index": 1, "k": 1, "features": 4, "n_probes": 8})

    def test_a_block_that_would_not_fit_the_function_disk_is_refused(self, tmp_path):
        source = tmp_path / "x.parquet"
        write_owi(source, rows=40, dimension=8)
        result, _ = prepare_build([str(source)], {"num_index": 1, "k": 1, "features": 8})
        # 40 rows x 8 values is tiny, so make the limit tiny as well
        with pytest.raises(PlanError, match="ephemeral storage of 0 MB"):
            check_ephemeral_storage(result, 0)
        check_ephemeral_storage(result, 512)

    def test_the_disk_size_is_not_an_index_setting(self, tmp_path):
        # it comes from the Lithops configuration of the executor
        source = tmp_path / "x.parquet"
        write_owi(source, rows=8)
        with pytest.raises(TypeError, match="ephemeral_storage"):
            prepare_build([str(source)], {"num_index": 1, "k": 1, "features": 4, "ephemeral_storage": 512})


def idmap_bytes(ids, record_ids, chunk_idx) -> bytes:
    sink = io.BytesIO()
    pq.write_table(
        pa.table({"id": pa.array(ids, pa.int64()), "record_id": record_ids, "chunk_idx": pa.array(chunk_idx, pa.int32())}),
        sink,
    )
    return sink.getvalue()


class TestIdmapSelect:
    def test_finds_ids_across_parts_and_ignores_the_rest(self):
        parts = [idmap_bytes([0, 1, 2], ["a", "a", "b"], [0, 1, 0]), idmap_bytes([3, 4], ["c", "d"], [0, 0])]
        assert select(parts, [4, 1, 99]) == {1: ("a", 1), 4: ("d", 0)}
        assert select(parts, []) == {}

    def test_prefix_layout(self):
        assert idmap_prefix("ds", "blocks") == "indexes/ds/blocks/idmap/"
