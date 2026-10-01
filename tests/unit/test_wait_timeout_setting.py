"""Where the wait timeout comes from: the client argument, the command line,
the environment, or the saved configuration."""

import json
from types import SimpleNamespace

import pytest

from vectordb import cli
from vectordb import client as client_module
from vectordb.client import VectorDBClient


def test_the_client_hands_its_wait_timeout_to_every_index_it_opens(monkeypatch):
    client = object.__new__(VectorDBClient)
    client.bucket = "bucket"
    client.wait_timeout = 300
    client.tracker = SimpleNamespace(
        table_name="table",
        dynamodb=SimpleNamespace(meta=SimpleNamespace(client=SimpleNamespace(meta=SimpleNamespace(region_name="us-east-1")))),
    )
    monkeypatch.setattr(client, "list_indexes", lambda name: [("blocks", 4)])
    monkeypatch.setattr(client_module, "load_index_config", lambda *args: {"num_index": 4, "features": 2})
    built = {}
    monkeypatch.setattr(client_module, "ServerlessVectorDB", lambda **config: built.update(config))
    client._load_default_index("ds")
    assert built["wait_timeout"] == 300


class Captured(Exception):
    """Raised by the fake client once the CLI has built it."""


class TestCli:
    @pytest.fixture
    def cli_env(self, tmp_path, monkeypatch):
        config_file = tmp_path / "backend_config.json"
        monkeypatch.setattr(cli, "CONFIG_DIR", tmp_path)
        monkeypatch.setattr(cli, "CONFIG_FILE", config_file)
        monkeypatch.delenv("SVDB_WAIT_TIMEOUT", raising=False)
        built = {}

        class FakeClient:
            def __init__(self, **kwargs):
                built.update(kwargs)

            def __getattr__(self, name):
                raise Captured(name)

        monkeypatch.setattr(cli, "VectorDBClient", FakeClient)
        return config_file, built

    def run(self, monkeypatch, *argv):
        monkeypatch.setattr("sys.argv", ["blocks-db", *argv])
        cli.main()

    def test_the_flag_reaches_the_client(self, cli_env, monkeypatch):
        _, built = cli_env
        with pytest.raises(Captured):
            self.run(monkeypatch, "--bucket", "b", "--wait-timeout", "0", "status", "ds")
        assert built["wait_timeout"] == 0

    def test_the_environment_reaches_the_client(self, cli_env, monkeypatch):
        _, built = cli_env
        monkeypatch.setenv("SVDB_WAIT_TIMEOUT", "900")
        with pytest.raises(Captured):
            self.run(monkeypatch, "--bucket", "b", "status", "ds")
        assert built["wait_timeout"] == 900

    def test_an_environment_value_that_is_not_a_number_is_refused(self, cli_env, monkeypatch, capsys):
        monkeypatch.setenv("SVDB_WAIT_TIMEOUT", "soon")
        with pytest.raises(SystemExit):
            self.run(monkeypatch, "--bucket", "b", "status", "ds")
        assert "SVDB_WAIT_TIMEOUT must be a number of seconds, got 'soon'" in capsys.readouterr().err

    def test_configure_saves_it_for_later_commands(self, cli_env, monkeypatch):
        config_file, built = cli_env
        self.run(monkeypatch, "configure", "--bucket", "b", "--wait-timeout", "1200")
        assert json.loads(config_file.read_text())["wait_timeout"] == 1200
        with pytest.raises(Captured):
            self.run(monkeypatch, "status", "ds")
        assert built["wait_timeout"] == 1200

    def test_setup_keeps_the_saved_value(self, cli_env, monkeypatch):
        # setup rewrites the saved file, and the README runs configure first
        config_file, built = cli_env
        monkeypatch.setattr(cli, "run_setup", lambda **kwargs: None)
        self.run(monkeypatch, "configure", "--bucket", "b", "--wait-timeout", "1200")
        self.run(monkeypatch, "--region", "us-east-1", "setup", "--bucket", "b")
        assert json.loads(config_file.read_text())["wait_timeout"] == 1200
        # the global flag lasts for one command, setup included
        self.run(monkeypatch, "--region", "us-east-1", "--wait-timeout", "30", "setup", "--bucket", "b")
        assert json.loads(config_file.read_text())["wait_timeout"] == 1200

    def test_without_any_setting_the_client_derives_it(self, cli_env, monkeypatch):
        _, built = cli_env
        with pytest.raises(Captured):
            self.run(monkeypatch, "--bucket", "b", "status", "ds")
        assert built["wait_timeout"] is None

    def test_a_wait_that_gave_up_ends_the_command_with_its_message(self, cli_env, monkeypatch):
        from vectordb.utils.waiting import FunctionsTimedOut

        class GivingUpClient:
            def __init__(self, **kwargs):
                pass

            def query(self, *args, **kwargs):
                raise FunctionsTimedOut("8 of 8 functions never started")

        monkeypatch.setattr(cli, "VectorDBClient", GivingUpClient)
        monkeypatch.setattr("sys.argv", ["blocks-db", "--bucket", "b", "query", "ds", "--vector", "0.1 0.2"])
        with pytest.raises(SystemExit) as stopped:
            cli.main()
        assert str(stopped.value.code) == "Error: 8 of 8 functions never started"
