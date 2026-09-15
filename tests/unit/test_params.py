"""The index parameters must be what their annotations say they are."""

import dataclasses

from vectordb.config import SvlessVectorDBParams


def test_defaults_match_their_annotations():
    # a trailing comma turns a default into a one-element tuple, which
    # then reaches arithmetic such as ceil(num_index / query_batch_size)
    params = SvlessVectorDBParams()
    for field in dataclasses.fields(SvlessVectorDBParams):
        value = getattr(params, field.name)
        if field.type is int:
            assert isinstance(value, int), f"{field.name}: {value!r}"


def test_query_batch_size_is_an_integer():
    assert SvlessVectorDBParams().query_batch_size == 16
