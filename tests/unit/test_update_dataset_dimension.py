"""update_dataset takes the dimension from the last row of source.csv, which
may carry tags."""

import pytest

from vectordb.utils import dataset_ops


class Body:
    def __init__(self, data):
        self.data = data

    def read(self):
        return self.data


class FakeS3:
    """One text object; get_object honors the "bytes=-N" range the reader asks for."""

    def __init__(self, text):
        self.data = text.encode()

    def get_object(self, Bucket, Key, Range=None):
        data = self.data
        if Range:
            data = data[-int(Range[len("bytes=-"):]):]
        return {"Body": Body(data)}


def test_the_tags_of_the_last_row_are_not_counted_as_dimensions(monkeypatch):
    rows = '1,0.1 0.2\n2,0.3 0.4,"{""source"": ""web"", ""lang"": ""es""}"\n'
    monkeypatch.setattr(dataset_ops, "s3", FakeS3(rows))
    assert dataset_ops.get_last_id_and_dim("bucket", "datasets/ds/source.csv") == (2, 2)


def test_a_last_row_without_a_comma_is_still_refused(monkeypatch):
    monkeypatch.setattr(dataset_ops, "s3", FakeS3("1,0.1 0.2\n2 0.3 0.4\n"))
    with pytest.raises(ValueError, match="missing comma"):
        dataset_ops.get_last_id_and_dim("bucket", "datasets/ds/source.csv")
