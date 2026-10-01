"""What a query is allowed to ask, and what happens when there is nothing to
search: both used to be found out inside the functions, or not at all."""

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


class TestADatasetWithNothingToSearch:
    def test_a_query_without_an_index_says_so_instead_of_answering_nothing(self, monkeypatch):
        # empty results and exit code 0 read like "no neighbours found"
        client = client_with(monkeypatch, indexes=[])
        with pytest.raises(NoIndex, match="No index found for dataset 'ds'"):
            client.query_batch("ds", [[0.0, 1.0]], k=1)

    def test_pending_vectors_are_still_searched_when_there_is_no_index(self, monkeypatch):
        client = client_with(monkeypatch, indexes=[], pending=[(7, [0.0, 1.0])])
        monkeypatch.setattr(
            "vectordb.utils.hybrid_search.brute_force_search",
            lambda vectors, unindexed, k: [[(7, 0.0)]],
        )
        results, times = client.query_batch("ds", [[0.0, 1.0]], k=1)
        assert results == [[(7, 0.0, "pending")]]
        assert times["fallback"].startswith("no index")


class TestASearchThatFailsIsNotAnEmptyAnswer:
    def test_the_error_of_the_search_reaches_the_caller(self, monkeypatch):
        # the catch used to cover the search as well, so a failure inside the
        # functions came back as "no index available" and no results
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
        # before, the command printed nothing and exited with 0
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
