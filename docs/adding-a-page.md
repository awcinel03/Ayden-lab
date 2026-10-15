# Adding a page

Every entry is **one page file + one registry entry**. Entries backed by data also get an offline pipeline that writes to Supabase. The worked example is *email-Eu-core: basic stats and degree distribution*. Copy its files and change them.

| Piece | Example |
|---|---|
| Registry entry | `lab/registry.py`, the `Entry(slug="email-eu-core-stats", …)` in `ENTRIES` |
| Page | `app_pages/network_science/graph_stats.py` |
| Ingester (if the data is new) | `pipelines/ingest/snap.py` |
| Experiment | `pipelines/experiments/network_science/graph_stats.py` |
| Tests | `tests/test_pipelines_snap.py`, `tests/test_app_smoke.py` |

## 1. Register the entry

Add an `Entry` to `ENTRIES` in `lab/registry.py`:

- `slug`: lowercase and hyphenated, unique across the lab. It is used for the page URL (`/<slug>`), as `experiments.slug` in Supabase, and as the first folder of the entry's artifacts in Storage.
- `category`: one of the five sections. The page file must live in `app_pages/<category.key>/`.
- `type`, `status`, `date`, `tags`, `icon`: shown in the page header and in the home page filters.
- `published=False` keeps the page out of navigation and keeps its data invisible to the public key until you're ready.

Navigation, the category index, and the home page pick the entry up automatically.

## 2. Write the page

Pages import only from `lab/` and third-party packages, never from `pipelines/` or from other pages. Follow the page contract:

1. `page_header(entry("<slug>"))`
2. **Purpose**: why the entry exists.
3. **Method**: data, definitions, choices.
4. **Results**: read with `db.latest_run(slug)` and `db.results(run.id)` (long format: `metric`, `value`, `unit`, `dims`). If there is no run yet, show `empty_state(...)` and call `st.stop()`, never raise.
5. **Takeaways**: compute the numbers in the text from the results, so the text stays true when the data changes.
6. `provenance_footer(run, db.run_datasets(run.id), CODE_PATHS)`, where `CODE_PATHS` lists the pipeline files. The links point at the run's commit.

Charts go through `lab/charts.py` (Altair). Add a helper there when a second page needs the same chart.

## 3. Get the data (offline)

**New public dataset:** add a source to an ingester, or write a new one modelled on `pipelines/ingest/snap.py`:

- Check the source's terms and rate limits first, and send a descriptive User-Agent.
- Record `license` (use `None` if the source states none) and `citation`.
- Upload files of 50 MB or less to the private `datasets` bucket, then upsert the `datasets` row on `slug`.
- Skip all writes when the sha256 hasn't changed.
- Larger files stay at their `source_url`, with `storage_path = null` and the sha256 recorded.

**Experiment:** model it on `pipelines/experiments/network_science/graph_stats.py`:

- `sync_experiments(client)` upserts the registry entry and returns its id.
- `start_run(...)` returns `None` when an identical run (same config, commit and dataset hashes) has already succeeded.
- Do the work inside `with tracked(client, run_id):`, so a crash marks the run `failed`.
- Verify the dataset's sha256 after downloading it.
- Write long-format results with `write_results`. Put larger tables in the `artifacts` bucket at `<slug>/<run_id>/<file>` and record them with `record_artifact`.

Bump `CONFIG["version"]` whenever the computation changes.

## 4. Run it

Commit first. Runs refuse a dirty tree, so that every run points at code that exists:

```
.venv/bin/python -m pipelines.ingest.snap email-Eu-core
.venv/bin/python -m pipelines.experiments.network_science.graph_stats
```

Run each twice. The second run must say `unchanged` / `skipped`.

## 5. Test it

- **Pipelines:** unit-test the pure functions (parsing, computing) and the idempotency branches with Supabase mocked. See `tests/test_pipelines_snap.py`.
- **Page:** `tests/test_app_smoke.py` already renders every registered page with no data. Add a test that monkeypatches `lab.db.latest_run`, `results` and `run_datasets` with small fixtures and renders the page.
- `tests/test_registry.py` fails if the page file and the registry disagree.

Before committing, run:

```
.venv/bin/ruff format . && .venv/bin/ruff check . && .venv/bin/pytest -q
```

Tests never touch the live project. `tests/conftest.py` hides local Streamlit secrets.
