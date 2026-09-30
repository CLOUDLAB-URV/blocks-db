"""What the command line refuses before any work starts: an option of the
format a build is not using, and a path that does not exist."""

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


class TestAPathThatDoesNotExist:
    @pytest.mark.parametrize("argv", [
        ["initialize-database", "ds", "{missing}", "--config", "{config}"],
        ["initialize-database", "ds", "{csv}", "--config", "{missing}"],
        ["initialize-database", "ds", "{parquet}", "--format", "parquet", "--config", "{missing}"],
        ["put", "ds", "{missing}"],
        ["query", "ds", "--file", "{missing}"],
    ], ids=["csv-source", "config", "config-of-a-parquet-build", "put", "query-file"])
    def test_the_command_ends_naming_it_before_anything_is_handed_to_the_client(self, run, tmp_path, argv):
        missing = tmp_path / "missing"
        argv = [word.format(missing=missing, config=tmp_path / "index.json", csv=tmp_path / "vectors.csv", parquet=tmp_path / "a.parquet")
                for word in argv]
        with pytest.raises(SystemExit) as stopped:
            run(*argv)
        assert str(stopped.value.code) == f"Error: {missing}: not found"
        assert run.handed == {}

    def test_a_file_not_found_error_inside_the_client_is_a_defect_and_keeps_its_traceback(self, run, monkeypatch):
        class BrokenClient:
            def __init__(self, **kwargs):
                pass

            def list_indexes(self, name):
                raise FileNotFoundError("/tmp/blocks/part-3.npy")

        monkeypatch.setattr(cli, "VectorDBClient", BrokenClient)
        with pytest.raises(FileNotFoundError):
            run("status", "ds")
