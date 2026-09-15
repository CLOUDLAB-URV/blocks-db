"""n_probe is a query-time parameter, not a property baked into a block."""

import faiss
import numpy as np

from vectordb.implementations.blocks.indexing import (
    FaissIVFIndex,
    apply_search_parameters,
)


class BuildParams:
    features = 8
    k = 4
    n_probe = 1


def test_n_probe_from_the_config_overrides_the_value_stored_in_the_block(tmp_path):
    vectors = np.random.default_rng(0).random((64, 8), dtype=np.float32)
    index = FaissIVFIndex(BuildParams()).build(list(range(64)), vectors)
    path = tmp_path / "block.ann"
    faiss.write_index(index, str(path))

    loaded = faiss.read_index(str(path))
    assert loaded.nprobe == 1  # the build-time value travels with the file

    apply_search_parameters(loaded, 3)
    assert loaded.nprobe == 3
