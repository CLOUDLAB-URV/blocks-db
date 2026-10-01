"""The CSV build function imports on an image without pyarrow."""

import importlib
import sys


def test_the_csv_build_function_needs_no_pyarrow(monkeypatch):
    # a None entry makes any import of the module fail, as on an image
    # built before pyarrow became a dependency
    for name in ("pyarrow", "pyarrow.parquet"):
        monkeypatch.setitem(sys.modules, name, None)
    for name in ("vectordb.implementations.blocks.initialize", "vectordb.utils.parquet"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    module = importlib.import_module("vectordb.implementations.blocks.initialize")
    assert callable(module.generate_index_blocks)
