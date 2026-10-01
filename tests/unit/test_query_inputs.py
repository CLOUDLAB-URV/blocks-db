"""What a query is allowed to ask, and what happens when there is nothing to
search: both are decided before any function runs."""

from types import SimpleNamespace

import numpy as np
import pytest

from vectordb import client as client_module
from vectordb.client import NoIndex, QueryMismatch, VectorDBClient, check_queries


class TestCheckQueries:
    def test_a_query_of_another_dimension_is_refused_by_name(self):
        # inside the map function this is a bare assertion from faiss
        with pytest.raises(QueryMismatch, match="index holds vectors of 1024 dimensions, the query has 384"):
            check_queries(np.zeros((1, 384)), 1024)

    def test_an_empty_batch_is_refused_before_any_function_runs(self):
        with pytest.raises(QueryMismatch, match="at least one vector"):
            check_queries(np.zeros((0, 1024)), 1024)

    def test_a_matching_query_passes(self):
        assert check_queries(np.zeros((3, 1024)), 1024) is None


def client_with(monkeypatch, indexes, pending=(), search=None):
    """A client whose dataset ``ds`` has those indexes and pending vectors."""
    client = object.__new__(VectorDBClient)
    client.bucket = "bucket"
    client.wait_timeout = None
    client.tracker = SimpleNamespace(
        table_name="table",
        dynamodb=SimpleNamespace(meta=SimpleNamespace(client=SimpleNamespace(meta=SimpleNamespace(region_name="region")))),
    )
    monkeypatch.setattr(client, "list_indexes", lambda name: indexes)
    monkeypatch.setattr(client, "_get_k_result", lambda name: 10)
    monkeypatch.setattr(client, "has_pending_vectors", lambda name: bool(pending))
    monkeypatch.setattr(client, "get_pending_vectors", lambda name: list(pending))
    monkeypatch.setattr(client_module, "load_index_config", lambda *args: {"num_index": 4, "features": 2})
    monkeypatch.setattr(
        client_module, "ServerlessVectorDB",
        lambda **config: SimpleNamespace(params=SimpleNamespace(features=config["features"]), search=search),
    )
    return client


class TestAnEmptyQuery:
    """Every way of asking refuses an empty batch the same way."""

    def test_an_empty_list_of_vectors_is_refused(self, monkeypatch):
        client = client_with(monkeypatch, indexes=[("blocks", 4)])
        with pytest.raises(QueryMismatch, match="No query vectors provided"):
            client.query_batch("ds", [])

    def test_an_empty_indexed_only_query_is_refused_before_the_index_is_read(self, monkeypatch):
        client = client_with(monkeypatch, indexes=[])
        with pytest.raises(QueryMismatch, match="No query vectors provided"):
            client.query_indexed_only("ds", vectors=[])

    def test_a_query_file_without_vectors_is_refused(self, tmp_path, monkeypatch):
        client = client_with(monkeypatch, indexes=[("blocks", 4)])
        empty = tmp_path / "queries.csv"
        empty.write_text("\n")
        with pytest.raises(QueryMismatch, match="No valid vectors found in CSV"):
            client.query_from_file("ds", str(empty))


@pytest.fixture
def queries(tmp_path):
    path = tmp_path / "queries.csv"
    path.write_text("0.0 1.0\n")
    return str(path)


# The public ways to ask for the vector (0, 1), each giving back (results, times)
def query(client, path):
    hit, times = client.query("ds", [0.0, 1.0], k=1)
    return [hit], times


def query_batch(client, path):
    return client.query_batch("ds", [[0.0, 1.0]], k=1)


def query_hybrid(client, path):
    return client.query_hybrid("ds", [[0.0, 1.0]], k=1)


def query_from_file(client, path):
    return client.query_from_file("ds", path, k=1)


def query_indexed_only(client, path):
    return client.query_indexed_only("ds", vector=[0.0, 1.0], k=1)


def query_not_hybrid(client, path):
    hit, times = client.query("ds", [0.0, 1.0], k=1, hybrid=False)
    return [hit], times


def query_batch_not_hybrid(client, path):
    return client.query_batch("ds", [[0.0, 1.0]], k=1, hybrid=False)


def query_from_file_not_hybrid(client, path):
    return client.query_from_file("ds", path, k=1, hybrid=False)


HYBRID = [query, query_batch, query_hybrid, query_from_file]
INDEXED_ONLY = [query_indexed_only, query_not_hybrid, query_batch_not_hybrid, query_from_file_not_hybrid]


def names(ask):
    return ask.__name__


class TestADatasetWithNothingToSearch:
    @pytest.mark.parametrize("ask", HYBRID, ids=names)
    def test_a_query_without_an_index_says_so_instead_of_answering_nothing(self, monkeypatch, queries, ask):
        # empty results and exit code 0 read like "no neighbors found"
        client = client_with(monkeypatch, indexes=[])
        with pytest.raises(NoIndex, match="No index found for dataset 'ds'"):
            ask(client, queries)

    @pytest.mark.parametrize("ask", HYBRID, ids=names)
    def test_pending_vectors_are_still_searched_when_there_is_no_index(self, monkeypatch, queries, ask):
        client = client_with(monkeypatch, indexes=[], pending=[(7, [0.0, 1.0])])
        monkeypatch.setattr(
            "vectordb.utils.hybrid_search.brute_force_search",
            lambda vectors, unindexed, k: [[(7, 0.0)]],
        )
        results, times = ask(client, queries)
        assert results == [[(7, 0.0, "pending")]]
        assert times["fallback"].startswith("no index")

    @pytest.mark.parametrize("pending", [[], [(7, [0.0, 1.0])]], ids=["nothing pending", "pending vectors"])
    @pytest.mark.parametrize("ask", INDEXED_ONLY, ids=names)
    def test_an_indexed_only_query_without_an_index_says_so_whatever_is_pending(self, monkeypatch, queries, ask, pending):
        # an empty answer with times["error"] would read as a search that
        # found nothing
        client = client_with(monkeypatch, indexes=[], pending=pending)
        with pytest.raises(NoIndex, match="No index found for dataset 'ds'"):
            ask(client, queries)


class TestADatasetWithAnIndexAndNothingPending:
    @pytest.mark.parametrize("ask", HYBRID, ids=names)
    def test_a_hybrid_query_answers_with_the_neighbors_of_the_search(self, monkeypatch, queries, ask):
        client = client_with(monkeypatch, indexes=[("blocks", 4)], search=lambda *a, **k: ([[(3, 0.5, "centroid_0")]], {}))
        results, times = ask(client, queries)
        assert results == [[(3, 0.5, "centroid_0")]]
        assert times["hybrid_search"] is True and times["has_pending"] is False

    @pytest.mark.parametrize("ask", INDEXED_ONLY, ids=names)
    def test_an_indexed_only_query_answers_with_the_neighbors_of_the_search(self, monkeypatch, queries, ask):
        client = client_with(monkeypatch, indexes=[("blocks", 4)], search=lambda *a, **k: ([[(3, 0.5, "centroid_0")]], {}))
        results, times = ask(client, queries)
        assert results == [[(3, 0.5, "centroid_0")]]
        assert "hybrid_search" not in times

    @pytest.mark.parametrize("ask", HYBRID + INDEXED_ONLY, ids=names)
    def test_no_neighbors_is_an_empty_answer_not_a_missing_index(self, monkeypatch, queries, ask):
        client = client_with(monkeypatch, indexes=[("blocks", 4)], search=lambda *a, **k: ([], {}))
        results, times = ask(client, queries)
        assert results == [[]]


class TestWhatAQueryHandsToTheFunctions:
    def test_the_source_list_and_block_ranges_of_a_parquet_index_stay_in_config_json(self, monkeypatch):
        # every map and reduce task carries the parameters; with a few
        # dozen source files the two lists push a task over what Lithops
        # sends inline, and provenance() reads them from config.json anyway
        client = client_with(monkeypatch, indexes=[("blocks", 4)])
        sealed = {
            "num_index": 4, "features": 2, "k": 1, "source_format": "parquet",
            "source_keys": [f"s3://bucket/datasets/ds/source/metadata_{i}_embeddings.parquet" for i in range(63)],
            "block_ranges": [[0, 0, 9], [1, 10, 19], [2, 20, 29], [3, 30, 39]],
        }
        handed = {}

        def open_index(**config):
            handed.update(config)
            return SimpleNamespace(params=SimpleNamespace(features=2), search=lambda *a, **k: ([[(3, 0.0, "indexed")]], {}))

        monkeypatch.setattr(client_module, "load_index_config", lambda *args: dict(sealed))
        monkeypatch.setattr(client_module, "ServerlessVectorDB", open_index)

        results, _ = client.query_batch("ds", [[0.0, 1.0]], k=1)

        assert results == [[(3, 0.0, "indexed")]]
        assert "source_keys" not in handed and "block_ranges" not in handed
        assert handed["source_format"] == "parquet" and handed["k"] == 1 and handed["dataset"] == "ds"


class TestASearchThatFailsIsNotAnEmptyAnswer:
    def test_the_error_of_the_search_reaches_the_caller(self, monkeypatch):
        # a failure inside the functions is not "no index available":
        # only the lookup of the index is caught
        def explode(*args, **kwargs):
            raise ValueError("the block could not be read")

        client = client_with(monkeypatch, indexes=[("blocks", 4)], search=explode)
        with pytest.raises(ValueError, match="the block could not be read"):
            client.query_batch("ds", [[0.0, 1.0]], k=1)

    def test_a_query_of_the_wrong_dimension_stops_before_the_functions(self, monkeypatch):
        client = client_with(monkeypatch, indexes=[("blocks", 4)], search=lambda *a, **k: pytest.fail("searched"))
        with pytest.raises(QueryMismatch, match="the query has 3"):
            client.query_batch("ds", [[0.0, 1.0, 2.0]], k=1)

    def test_a_batch_size_below_one_is_refused_with_what_it_means(self, monkeypatch):
        client = client_with(monkeypatch, indexes=[("blocks", 4)], search=lambda *a, **k: pytest.fail("searched"))
        with pytest.raises(QueryMismatch, match="must be at least 1, got 0"):
            client.query_batch("ds", [[0.0, 1.0]], k=1, batch_size=0)


class TestTheCommandLine:
    def test_an_empty_query_file_ends_with_the_reason(self, tmp_path, monkeypatch):
        from vectordb import cli

        class EmptyFileClient:
            def __init__(self, **kwargs):
                pass

            def query_from_file(self, *args, **kwargs):
                return client_with(monkeypatch, indexes=[("blocks", 4)]).query_from_file(*args, **kwargs)

        empty = tmp_path / "queries.csv"
        empty.write_text("\n")
        monkeypatch.setattr(cli, "CONFIG_DIR", tmp_path)
        monkeypatch.setattr(cli, "CONFIG_FILE", tmp_path / "backend_config.json")
        monkeypatch.setattr(cli, "VectorDBClient", EmptyFileClient)
        monkeypatch.setattr("sys.argv", ["blocks-db", "--bucket", "b", "query", "ds", "--file", str(empty)])
        with pytest.raises(SystemExit) as stopped:
            cli.main()
        assert str(stopped.value.code) == "Error: No valid vectors found in CSV."

    def test_a_query_with_nothing_to_search_ends_with_the_reason(self, tmp_path, monkeypatch):
        from vectordb import cli

        class RefusingClient:
            def __init__(self, **kwargs):
                pass

            def query(self, *args, **kwargs):
                raise NoIndex("No index found for dataset 'ds'. Run indexing first.")

        monkeypatch.setattr(cli, "CONFIG_DIR", tmp_path)
        monkeypatch.setattr(cli, "CONFIG_FILE", tmp_path / "backend_config.json")
        monkeypatch.setattr(cli, "VectorDBClient", RefusingClient)
        monkeypatch.setattr("sys.argv", ["blocks-db", "--bucket", "b", "query", "ds", "--vector", "0.1 0.2"])
        with pytest.raises(SystemExit) as stopped:
            cli.main()
        assert str(stopped.value.code) == "Error: No index found for dataset 'ds'. Run indexing first."
