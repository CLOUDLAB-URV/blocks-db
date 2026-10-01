"""Which names list_datasets finds, from what a build leaves in the bucket."""

from vectordb.client import VectorDBClient


class FakeBucket:
    """Keys listed in S3 order, two to a page, so a name past the first
    page is only found by a client that turns the pages."""

    def __init__(self, keys):
        self.keys = sorted(keys)

    def get_paginator(self, _name):
        keys = self.keys

        class Paginator:
            def paginate(self, Bucket, Prefix):
                found = [key for key in keys if key.startswith(Prefix)]
                for start in range(0, len(found), 2):
                    yield {"Contents": [{"Key": key} for key in found[start:start + 2]]}
                if not found:
                    yield {}

        return Paginator()


def client_over(keys):
    client = object.__new__(VectorDBClient)
    client.bucket = "bucket"
    client.s3 = FakeBucket(keys)
    return client


def test_a_csv_dataset_is_listed_once():
    client = client_over([
        "datasets/glove/source.csv",
        "indexes/glove/blocks/centroid_0.ann",
        "indexes/glove/blocks/config.json",
        "pending/glove/1.csv",
        "tracking/csv_blocks_glove.json",
    ])
    assert client.list_datasets() == ["glove"]


def test_a_parquet_dataset_built_from_uploaded_files_is_listed_once():
    client = client_over([
        "datasets/owi/source/language=eng/metadata_0_embeddings.parquet",
        "datasets/owi/source/language=spa/metadata_0_embeddings.parquet",
        "indexes/owi/blocks/centroid_0.ann",
        "indexes/owi/blocks/config.json",
        "indexes/owi/blocks/idmap/block_0.parquet",
    ])
    assert client.list_datasets() == ["owi"]


def test_a_parquet_dataset_read_in_place_is_listed_from_its_configuration():
    # the sources stayed where they were, so nothing is under datasets/
    client = client_over([
        "indexes/owi/blocks/centroid_0.ann",
        "indexes/owi/blocks/config.json",
        "indexes/owi/blocks/idmap/block_0.parquet",
        "shared/day/metadata_0_embeddings.parquet",
    ])
    assert client.list_datasets() == ["owi"]


def test_names_keep_the_order_of_the_listing_across_pages_and_prefixes():
    client = client_over([
        "datasets/a/source.csv",
        "datasets/b/source/x.parquet",
        "datasets/c/source.csv",
        "indexes/a/blocks/config.json",
        "indexes/b/blocks/config.json",
        "indexes/d/blocks/config.json",
    ])
    assert client.list_datasets() == ["a", "b", "c", "d"]


def test_a_bucket_without_datasets_lists_nothing():
    assert client_over([]).list_datasets() == []
