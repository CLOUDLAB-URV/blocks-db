"""Fixtures shared by the unit tests."""

import pytest

from helpers import write_owi


@pytest.fixture
def owi_file(tmp_path):
    path = tmp_path / "metadata_0_embeddings.parquet"
    vectors = write_owi(path, rows=10, row_group_size=4)  # row groups 4, 4, 2
    return str(path), vectors
