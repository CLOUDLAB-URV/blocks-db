"""The DynamoDB table is chosen by configuration, not fixed in the code.

Several deployments can share one AWS account; a client that always
writes to the default table writes into whichever deployment owns it.
None of these tests call AWS: boto3 resources and tables are lazy.
"""

import json

import pytest

from vectordb import cli
from vectordb import client as client_module
from vectordb.client import VectorDBClient
from vectordb.utils.vector_tracking import VectorIndexTracker


class FakeTable:
    def __init__(self):
        self.updates = []
        self.items = []

    def update_item(self, **kwargs):
        self.updates.append(kwargs)

    def put_item(self, Item):
        self.items.append(Item)


class TestTracker:
    def test_without_a_name_the_default_table_is_kept(self):
        tracker = VectorIndexTracker("bucket", "us-east-1")
        assert tracker.table_name == "BlocksDB-default"
        assert tracker.table.name == "BlocksDB-default"

    def test_a_declared_table_is_the_one_used(self):
        tracker = VectorIndexTracker("bucket", "us-east-1", table_name="team-table")
        assert tracker.table.name == "team-table"
        assert tracker.dynamodb.meta.client.meta.region_name == "us-east-1"


class TestClient:
    def test_the_client_hands_its_table_to_the_tracker(self):
        client = VectorDBClient("bucket", "us-east-1", dynamodb_table_name="team-table")
        assert client.tracker.table_name == "team-table"

    def test_the_auto_indexer_state_goes_to_the_configured_table(self):
        client = VectorDBClient("bucket", "us-east-1", dynamodb_table_name="team-table")
        client.tracker.table = FakeTable()

        client._setup_auto_indexer_state("ds", {"num_index": 4})

        (update,) = client.tracker.table.updates
        assert update["Key"] == {"centroid_id": "ds_CONFIG", "sk": "META"}
        assert update["ExpressionAttributeValues"][":next"] == 4

    def test_the_centroid_tags_go_to_the_configured_table(self, monkeypatch):
        class TagsS3:
            def get_object(self, Bucket, Key):
                if Key != "indexes/ds/blocks/centroid_0_tags.json":
                    raise KeyError(Key)

                class Body:
                    def read(self):
                        return json.dumps({"7": {"source": "web"}}).encode()

                return {"Body": Body()}

        monkeypatch.setattr(client_module.boto3, "client", lambda *args, **kwargs: TagsS3())
        client = VectorDBClient.__new__(VectorDBClient)
        client.bucket = "bucket"
        client.tracker = VectorIndexTracker("bucket", "us-east-1", table_name="team-table")
        client.tracker.table = FakeTable()

        client._aggregate_centroid_tags_to_ddb("ds", 2, {"implementation": "blocks"})

        assert client.tracker.table.items == [
            {"centroid_id": "DATASET#ds", "sk": "CENTROID#0#META", "tags": {"source": ["web"]}}
        ]


class Captured(Exception):
    """Raised by the fake client once the CLI has built it."""


class TestCli:
    @pytest.fixture
    def cli_env(self, tmp_path, monkeypatch):
        config_file = tmp_path / "backend_config.json"
        monkeypatch.setattr(cli, "CONFIG_DIR", tmp_path)
        monkeypatch.setattr(cli, "CONFIG_FILE", config_file)
        monkeypatch.delenv("SVDB_DYNAMODB_TABLE", raising=False)
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
            self.run(monkeypatch, "--bucket", "b", "--table-name", "team-table", "status", "ds")
        assert built["dynamodb_table_name"] == "team-table"

    def test_the_environment_reaches_the_client(self, cli_env, monkeypatch):
        _, built = cli_env
        monkeypatch.setenv("SVDB_DYNAMODB_TABLE", "env-table")
        with pytest.raises(Captured):
            self.run(monkeypatch, "--bucket", "b", "status", "ds")
        assert built["dynamodb_table_name"] == "env-table"

    def test_configure_saves_the_table_for_later_commands(self, cli_env, monkeypatch):
        config_file, built = cli_env
        self.run(monkeypatch, "configure", "--bucket", "b", "--region", "us-east-1", "--table-name", "saved-table")
        assert json.loads(config_file.read_text())["dynamodb_table_name"] == "saved-table"

        with pytest.raises(Captured):
            self.run(monkeypatch, "status", "ds")
        assert built == {
            "bucket": "b",
            "region": "us-east-1",
            "sqs_queue_url": None,
            "dynamodb_table_name": "saved-table",
        }

    def test_the_global_flag_also_reaches_configure_and_setup(self, cli_env, monkeypatch):
        # a subparser with its own --table-name and a default of None used to
        # overwrite the value given before the command name
        config_file, _ = cli_env
        self.run(monkeypatch, "--table-name", "team-table", "configure", "--bucket", "b")
        assert json.loads(config_file.read_text())["dynamodb_table_name"] == "team-table"

    def test_without_any_setting_the_client_keeps_its_default(self, cli_env, monkeypatch):
        _, built = cli_env
        with pytest.raises(Captured):
            self.run(monkeypatch, "--bucket", "b", "status", "ds")
        assert built["dynamodb_table_name"] is None
