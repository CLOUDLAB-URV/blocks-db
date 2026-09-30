"""What the command line refuses before any work starts: an option of the
format a build is not using."""

import json

import pytest

from vectordb import cli


@pytest.fixture
def run(tmp_path, monkeypatch):
    """Runs a command with a client that records what it is handed and never
    reaches the cloud. tmp_path holds vectors.csv, a.parquet and index.json."""
    handed = {}

    class RecordingClient:
        def __init__(self, **kwargs):
            pass

        def create_dataset(self, name, source):
            handed["source"] = source

        def index_dataset(self, **kwargs):
            handed.update(kwargs)
            return {}

        def index_parquet_dataset(self, name, sources, config, replace=False):
            handed.update(sources=sources, replace=replace)
            return {}

    monkeypatch.setattr(cli, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(cli, "CONFIG_FILE", tmp_path / "backend_config.json")
    monkeypatch.setattr(cli, "VectorDBClient", RecordingClient)
    (tmp_path / "vectors.csv").write_text("1,0.0 1.0\n")
    (tmp_path / "a.parquet").write_bytes(b"")
    (tmp_path / "index.json").write_text(json.dumps({"num_index": 1}))

    def run(*argv):
        monkeypatch.setattr("sys.argv", ["blocks-db", "--bucket", "b", *argv])
        cli.main()
        return handed

    run.handed = handed
    return run


@pytest.fixture
def build(run, tmp_path):
    """initialize-database of a file in tmp_path with the index.json there."""
    def build(source, *options):
        return run("initialize-database", "ds", str(tmp_path / source), "--config", str(tmp_path / "index.json"), *options)

    return build


class TestAnOptionOfTheOtherFormat:
    @pytest.mark.parametrize("option", [
        ["--workers", "16"], ["--workers", "0"], ["--no-update-threshold"], ["--skip-auto-indexer"], ["--build-local"], ["--csv-block-size", "1000"],
    ], ids=["workers", "workers-zero", "no-update-threshold", "skip-auto-indexer", "build-local", "csv-block-size"])
    def test_a_csv_build_option_is_refused_with_format_parquet(self, build, option, capsys):
        with pytest.raises(SystemExit) as stopped:
            build("a.parquet", "--format", "parquet", *option)
        assert stopped.value.code == 2
        assert f"{option[0]} does not apply to --format parquet" in capsys.readouterr().err

    @pytest.mark.parametrize("option", [["--replace"], ["--files", "*.parquet"]], ids=["replace", "files"])
    def test_a_parquet_build_option_is_refused_with_format_csv(self, build, option, capsys):
        with pytest.raises(SystemExit) as stopped:
            build("vectors.csv", *option)
        assert stopped.value.code == 2
        assert f"{option[0]} does not apply to --format csv" in capsys.readouterr().err

    def test_an_option_not_given_is_not_reported(self, build):
        # --workers defaults to 16 and --files to *.parquet, but only an
        # option the user gives is refused
        assert build("a.parquet", "--format", "parquet")["replace"] is False
        assert build("vectors.csv", "--no-update-threshold")["num_workers"] == 16

    def test_a_csv_build_takes_the_workers_it_is_given(self, build):
        assert build("vectors.csv", "--no-update-threshold", "--workers", "4")["num_workers"] == 4
