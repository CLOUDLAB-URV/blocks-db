from vectordb.core.indexing import IndexBuilder
import faiss
import numpy as np


class FaissIVFIndex(IndexBuilder):

    def __init__(self, params):

        self.features = params.features
        self.k = params.k
        self.nprobe = params.n_probe


    def build(self, ids, vectors):

        index = faiss.index_factory(self.features, f"IVF{self.k},Flat")

        x = np.asarray(vectors, dtype=np.float32)

        index.train(x)

        index.nprobe = self.nprobe

        index.add_with_ids(x, np.asarray(ids, dtype=np.int64))

        return index


def apply_search_parameters(index, n_probe: int):
    """Apply the query-time IVF search width to a block read from storage.

    ``write_index`` stores the ``nprobe`` the block was built with, so a
    block read back searches with the build-time value whatever the
    ``n_probe`` in the dataset's config.json says. Setting it after
    ``read_index`` makes the config.json value the one that is searched
    with, so it can be changed without rebuilding the blocks.
    """
    index.nprobe = n_probe
    return index


IMPLEMENTATION_INDEX_BUILDER = FaissIVFIndex