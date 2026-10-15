"""SNAP ingester and graph_stats experiment, with Supabase mocked."""

import gzip
from unittest.mock import MagicMock

import networkx as nx

from pipelines.experiments.network_science import graph_stats
from pipelines.ingest import snap

EDGES = b"# comment\n0 1\n1 0\n1 2\n2 0\n3 3\n"
GZ = gzip.compress(EDGES)
SOURCE = snap.SOURCES["email-Eu-core"]


def test_count_edges_skips_comments():
    assert snap.count_edges(GZ) == 5


def mock_client(existing_sha=None, exists=True):
    client = MagicMock()
    datasets = MagicMock()
    datasets.select.return_value.eq.return_value.execute.return_value.data = (
        [{"sha256": existing_sha}] if existing_sha else []
    )
    client.table.return_value = datasets
    client.storage.from_.return_value.exists.return_value = exists
    return client, datasets


def test_ingest_skips_unchanged_file():
    client, datasets = mock_client(existing_sha=snap.sha256_bytes(GZ))
    assert snap.ingest(client, SOURCE, fetch=lambda url: GZ) == "unchanged"
    client.storage.from_.return_value.upload.assert_not_called()
    datasets.upsert.assert_not_called()


def test_ingest_uploads_and_records_new_file():
    client, datasets = mock_client()
    assert snap.ingest(client, SOURCE, fetch=lambda url: GZ) == "updated"
    path = client.storage.from_.return_value.upload.call_args.args[0]
    assert path == "snap/email-Eu-core.txt.gz"
    row = datasets.upsert.call_args.args[0]
    assert row["sha256"] == snap.sha256_bytes(GZ) and row["row_count"] == 5
    assert row["license"] is None and "Leskovec" in row["citation"]


def test_ingest_reuploads_when_object_missing():
    client, _ = mock_client(existing_sha=snap.sha256_bytes(GZ), exists=False)
    assert snap.ingest(client, SOURCE, fetch=lambda url: GZ) == "updated"


def test_compute_on_small_graph():
    g = graph_stats.load_graph(GZ)
    rows = graph_stats.compute(g)
    scalars = {r["metric"]: r["value"] for r in rows if "dims" not in r}
    assert scalars["nodes"] == 4 and scalars["edges"] == 5 and scalars["self_loops"] == 1
    assert scalars["strong_components"] == 2  # {0, 1, 2} and {3}
    assert scalars["largest_scc_fraction"] == 0.75
    assert scalars["reciprocity"] == nx.reciprocity(g)
    total = {
        r["dims"]["degree"]: r["value"]
        for r in rows
        if r["metric"] == "degree_count" and r["dims"]["kind"] == "total"
    }
    assert sum(total.values()) == 4  # every node counted once
