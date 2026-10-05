"""The build planner: exactly num_index non-empty blocks, dense ids."""

import pytest

from vectordb.indexing.planner import PlanError, largest_nlist, plan, suggested_nlist
from vectordb.utils.parquet import CANONICAL, OWI_V2, FileInfo


def info(uri, *row_groups, dimension=4, dialect=OWI_V2):
    return FileInfo(uri=uri, dialect=dialect, dimension=dimension, row_groups=tuple(row_groups))


def ids_of(result):
    """Every id the plan assigns, in block order."""
    out = []
    for block in result.blocks:
        for part in block.ranges:
            out.extend(range(part.id_offset, part.id_offset + part.rows))
    return out


class TestWholeRowGroups:
    def test_one_file_one_group_per_block(self):
        result = plan([info("a", 5, 5, 5, 5)], num_index=4)
        assert [b.rows for b in result.blocks] == [5, 5, 5, 5]
        assert all(len(b.ranges) == 1 and b.ranges[0].start == 0 for b in result.blocks)
        assert ids_of(result) == list(range(20))
        assert result.total_vectors == 20 and result.min_block_rows == 5

    def test_uneven_groups_are_balanced_on_edges(self):
        # 100 rows in groups 10,40,10,40 -> two blocks of 50 without slicing
        result = plan([info("a", 10, 40, 10, 40)], num_index=2)
        assert [b.rows for b in result.blocks] == [50, 50]
        assert [len(b.ranges) for b in result.blocks] == [2, 2]

    def test_files_are_ordered_by_uri_and_ids_run_across_files(self):
        result = plan([info("b", 3), info("a", 2, 2)], num_index=3)
        assert result.sources == ("a", "b")
        assert [b.rows for b in result.blocks] == [2, 2, 3]
        assert ids_of(result) == list(range(7))
        assert result.blocks[2].ranges[0].uri == "b"

    def test_every_block_gets_at_least_one_group(self):
        # 8 groups, 8 blocks: no snapping may swallow a block
        result = plan([info("a", *([1] * 8))], num_index=8)
        assert [b.rows for b in result.blocks] == [1] * 8

    def test_empty_row_groups_are_skipped(self):
        result = plan([info("a", 3, 0, 3)], num_index=2)
        assert [b.rows for b in result.blocks] == [3, 3]


class TestSlicedRowGroups:
    def test_single_row_group_is_sliced_at_even_rows(self):
        result = plan([info("a", 10)], num_index=4)
        assert sum(b.rows for b in result.blocks) == 10
        assert all(2 <= b.rows <= 3 for b in result.blocks)
        assert ids_of(result) == list(range(10))
        first = result.blocks[0].ranges[0]
        assert (first.row_group, first.start, first.id_offset) == (0, 0, 0)

    def test_a_block_may_span_two_groups_when_sliced(self):
        result = plan([info("a", 4, 4)], num_index=3)
        assert sum(b.rows for b in result.blocks) == 8
        assert ids_of(result) == list(range(8))
        spanning = [b for b in result.blocks if len(b.ranges) == 2]
        assert len(spanning) == 1


class TestRefusals:
    def test_more_blocks_than_rows(self):
        with pytest.raises(PlanError, match="num_index 5 exceeds the 3 rows"):
            plan([info("a", 3)], num_index=5)

    def test_k_larger_than_the_smallest_block_names_a_suggestion(self):
        with pytest.raises(PlanError, match=r"k .* is 4096 but block \d holds only 100 rows.* Suggested k: 2"):
            plan([info("a", 100, 100)], num_index=2, k=4096)
        assert suggested_nlist(100) == 2 and suggested_nlist(10) == 1

    def test_the_suggestion_is_four_root_rows_capped_at_what_faiss_trains(self):
        # a large block: far below the cap
        assert suggested_nlist(170_131) == 1650 and largest_nlist(170_131) == 4362
        # a small block: the cap decides
        assert suggested_nlist(1_560) == largest_nlist(1_560) == 40
        with pytest.raises(PlanError, match=r"Suggested k: 1650 \(at most 4362\)"):
            plan([info("a", 170_131)], num_index=1, k=200_000)

    def test_dimension_must_match_features_and_be_consistent(self):
        with pytest.raises(PlanError, match="features is 8 but the source vectors have 4"):
            plan([info("a", 3)], num_index=1, features=8)
        with pytest.raises(PlanError, match="mixed vector dimensions"):
            plan([info("a", 3), info("b", 3, dimension=5)], num_index=1)
        with pytest.raises(PlanError, match="mixed dialects"):
            plan([info("a", 3), info("b", 3, dialect=CANONICAL)], num_index=1)

    def test_no_files_and_bad_num_index(self):
        with pytest.raises(PlanError, match="no source files"):
            plan([], num_index=1)
        with pytest.raises(PlanError, match="num_index must be at least 1"):
            plan([info("a", 3)], num_index=0)
