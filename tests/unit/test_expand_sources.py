"""What a declared source stands for."""

import pytest

from vectordb.indexing.planner import PlanError
from vectordb.indexing.prepare import expand_sources


def test_local_file_directory_and_missing_path(tmp_path):
    (tmp_path / "b.parquet").write_bytes(b"")
    (tmp_path / "a.parquet").write_bytes(b"")
    (tmp_path / "notes.txt").write_bytes(b"")
    assert expand_sources(str(tmp_path / "a.parquet")) == [str(tmp_path / "a.parquet")]
    assert expand_sources(str(tmp_path)) == [str(tmp_path / "a.parquet"), str(tmp_path / "b.parquet")]
    with pytest.raises(PlanError, match="not found"):
        expand_sources(str(tmp_path / "absent.parquet"))
    (tmp_path / "empty").mkdir()
    with pytest.raises(PlanError, match=r"no files matching '\*.parquet' in the directory"):
        expand_sources(str(tmp_path / "empty"))


def test_s3_file_and_prefix():
    assert expand_sources("s3://b/k/file.parquet") == ["s3://b/k/file.parquet"]
    listed = expand_sources("s3://b/day/", lambda bucket, prefix: [f"{prefix}z.parquet", f"{prefix}a.parquet", f"{prefix}a.txt"])
    assert listed == ["s3://b/day/a.parquet", "s3://b/day/z.parquet"]
    with pytest.raises(PlanError, match="needs a lister"):
        expand_sources("s3://b/day/")
    with pytest.raises(PlanError, match=r"no files matching '\*.parquet' under the prefix"):
        expand_sources("s3://b/day/", lambda bucket, prefix: [])


OWI_DAY = ("language=spa/metadata_0_embeddings.parquet", "language=spa/metadata_0_records.parquet",
           "language=deu/metadata_0_embeddings.parquet", "language=deu/metadata_0_records.parquet")


def test_a_files_pattern_narrows_a_directory_and_a_prefix_alike(tmp_path):
    # a day of the Open Web Index keeps a records file beside every
    # embeddings file, and only the embeddings are vector sources
    for name in OWI_DAY:
        (tmp_path / name).parent.mkdir(exist_ok=True)
        (tmp_path / name).write_bytes(b"")
    local = expand_sources(str(tmp_path), files="*_embeddings.parquet")
    assert local == [str(tmp_path / "language=deu/metadata_0_embeddings.parquet"),
                     str(tmp_path / "language=spa/metadata_0_embeddings.parquet")]
    listed = expand_sources("s3://b/day/", lambda bucket, prefix: [prefix + name for name in OWI_DAY],
                            files="*_embeddings.parquet")
    assert listed == ["s3://b/day/language=deu/metadata_0_embeddings.parquet",
                      "s3://b/day/language=spa/metadata_0_embeddings.parquet"]
    # without a pattern a directory still stands for every parquet file
    assert len(expand_sources(str(tmp_path))) == 4


def test_a_pattern_that_matches_nothing_is_named(tmp_path):
    (tmp_path / "a.parquet").write_bytes(b"")
    with pytest.raises(PlanError, match=r"no files matching '\*_embeddings.parquet' in the directory"):
        expand_sources(str(tmp_path), files="*_embeddings.parquet")
    with pytest.raises(PlanError, match=r"no files matching '\*_embeddings.parquet' under the prefix"):
        expand_sources("s3://b/day/", lambda bucket, prefix: [prefix + "a.parquet"], files="*_embeddings.parquet")


def test_a_single_declared_file_is_taken_as_declared(tmp_path):
    (tmp_path / "a.parquet").write_bytes(b"")
    assert expand_sources(str(tmp_path / "a.parquet"), files="*_embeddings.parquet") == [str(tmp_path / "a.parquet")]
    assert expand_sources("s3://b/k/a.parquet", files="*_embeddings.parquet") == ["s3://b/k/a.parquet"]
