# Ayden's Lab — Project Brief

## 1. What this is

A public research site built with Streamlit that collects my projects, tools, findings, explorations and experiments. Each item is one page, grouped into five categories in the sidebar using Streamlit's `st.navigation` sections:

| Section | Covers |
|---|---|
| Network Science | Graph analysis, network models, complexity |
| Machine Learning | Models, training experiments, classifiers |
| Claude API | Tools and experiments built on the Claude API |
| Model Testing | Cross-provider evals and comparisons (Anthropic, OpenAI, Google, others) |
| AI General | Broader explorations, essays-with-data, one-offs |

Hosting: Streamlit Community Cloud, deployed from a GitHub repo.
Data: Supabase (Postgres + Storage) holds experiment results **and** the public datasets experiments use (e.g. SEC Form D filings, Stanford SNAP graphs).
Owner: Ayden Cinel.

## 2. Fixed architecture decisions

1. **Read-only app, offline compute.** The Streamlit app only reads from Supabase and renders. Experiments, model API calls, and data ingestion run as scripts in `pipelines/` on my machine. No page calls a paid model API on load. Live demos are out of scope; if added later they go behind an auth gate as a separate decision.
2. **Supabase is the system of record** for (a) experiment runs and results, and (b) raw/processed public datasets in Storage, each described by a provenance row in a `datasets` table.
3. **Two keys, two contexts.** The app uses the anon/publishable key with RLS-enforced read-only access. Pipelines use the service-role/secret key from a local `.env`. The service key never appears in Streamlit secrets, app code, or git.
4. **Registry-driven navigation.** Every page is declared once in `lab/registry.py` with metadata. `streamlit_app.py` builds `st.navigation` from the registry. Adding a page = one page file + one registry entry.
5. **Explicit routing.** Use `st.navigation` / `st.Page`. Page files live in `app_pages/`, not `pages/`, to avoid Streamlit's legacy multipage auto-discovery.

## 3. Proposed file structure

The agent may refine this during planning but must justify every change.

```
aydens-lab/
├── streamlit_app.py              # entrypoint: page config, builds st.navigation from registry
├── .streamlit/
│   ├── config.toml               # theme only
│   └── secrets.toml.example      # key names, no values (real secrets.toml is gitignored)
├── app_pages/
│   ├── home.py                   # intro, recent entries, filter by category/type/tag
│   ├── network_science/
│   │   ├── index.py              # category index, auto-lists entries from registry
│   │   └── graph_stats.py        # example entry (M3)
│   ├── machine_learning/index.py
│   ├── claude_api/index.py
│   ├── model_testing/index.py
│   └── ai_general/index.py
├── lab/                          # shared, importable package used by the app
│   ├── __init__.py
│   ├── registry.py               # Entry dataclass + list of all entries
│   ├── db.py                     # cached Supabase client + typed read helpers
│   ├── storage.py                # dataset/artifact fetch from Storage, local cache
│   ├── ui.py                     # page_header, provenance footer, empty-state helpers
│   └── charts.py                 # shared Plotly/Altair helpers
├── pipelines/                    # offline only — never imported by the app
│   ├── common/                   # service-role client, run logging, checksums, git SHA
│   ├── ingest/
│   │   └── snap.py               # example ingester (M3)
│   └── experiments/
│       └── network_science/
│           └── graph_stats.py    # example experiment (M3)
├── supabase/
│   ├── migrations/               # numbered SQL; never edit an applied migration
│   └── seed.sql
├── tests/
├── docs/
│   ├── PLAN.md                   # written at end of Phase 1
│   ├── DECISIONS.md              # dated decision log
│   ├── adding-a-page.md          # how to add an entry using the example as template
│   └── deploy.md                 # Community Cloud checklist
├── .github/workflows/ci.yml      # ruff + pytest
├── BRIEF.md
├── CLAUDE.md                     # build rules (§9), created at start of Phase 2
├── requirements.txt              # pinned; deploy manifest for Community Cloud
├── requirements-pipelines.txt    # extra deps for offline scripts only
├── .env.example
└── .gitignore
```

## 4. Data model (starting point)

**Tables**
- `datasets` — slug, name, source, source_url, license, storage_path, format, bytes, sha256, row_count, retrieved_at, notes
- `experiments` — slug, title, category, description, published (bool), created_at
- `runs` — experiment_id, status, started_at, finished_at, config (jsonb), git_sha, dataset_ids
- `results` — long format: run_id, metric, value, unit, dims (jsonb)
- `artifacts` — run_id, kind (plot, model, raw_outputs, …), storage_path, bytes

**Model Testing runs must also capture:** provider, exact model ID, all sampling parameters, system prompt and prompt-template version (hash), input set ID, input/output token counts, latency, estimated cost, raw outputs (Storage), timestamp.

**Storage buckets:** `datasets` and `artifacts`. Decide public vs signed-URL access in the plan.

**RLS:** anon may `select` published rows only; no anon insert/update/delete anywhere.

## 5. Page contract

Every entry page:
- Starts with `lab.ui.page_header(entry)`: title, category, type badge (Tool / Experiment / Finding / Graph / Model / Exploration), status, date, tags.
- Follows: Purpose → Method → Results (interactive) → Takeaways → Provenance (datasets, run IDs, GitHub link to code at the run's commit SHA).
- Reads data only through `lab/` helpers, cached.
- Degrades gracefully when data is missing (`st.info`, never a traceback).

Each category has an index page that lists its entries from the registry, so all five sections exist before they have content. The home page shows an intro, recent entries, and filters by category, type, and tag.

## 6. First build scope

**In scope**
- Full skeleton from §3: entrypoint, registry, home, five category index pages.
- Supabase: migrations, RLS policies, buckets, `lab/db.py`, `lab/storage.py`, `pipelines/common/`.
- One end-to-end example entry that becomes the template: **Network Science → basic stats and degree distribution of a small SNAP graph** (e.g. email-Eu-core).
  - `pipelines/ingest/snap.py`: download → checksum → upload to Storage → upsert `datasets` row.
  - `pipelines/experiments/network_science/graph_stats.py`: load from Storage → compute stats → write `runs`/`results`/`artifacts`.
  - `app_pages/network_science/graph_stats.py`: render per the page contract.
- `docs/adding-a-page.md`, `docs/deploy.md`.
- Tests and CI (M4).

**Out of scope**
- Live model API demos and auth.
- SEC Form D ingestion (schema must support it; build later).
- Custom CSS/JS beyond the theme in `config.toml`.

## 7. Phase 1 — Planning (no code)

Deliverable: a plan, written to `docs/PLAN.md` after approval, containing:
1. Final file tree with a one-line purpose per file.
2. Registry schema (fields, types) and exactly how navigation is built from it.
3. SQL schema, RLS policies, and bucket configuration.
4. Dependencies with pinned versions and a reason for each; which go in `requirements.txt` vs `requirements-pipelines.txt`.
5. Data-size strategy: check **current** Supabase free-tier limits (DB size, Storage total, max upload size) and Community Cloud resource limits, then define what is stored raw, what as compressed Parquet, and what stays at its source URL with a recorded checksum.
6. Milestones (§8) with concrete acceptance criteria.
7. Anything in this brief you think is wrong, with reasons.
8. Open questions for me.

Use plan mode. Verify current APIs and limits from documentation rather than memory. Stop after presenting the plan and wait for approval.

## 8. Phase 2 — Milestones

| # | Milestone | Done when |
|---|---|---|
| M1 | Repo & skeleton | App runs locally; sidebar shows all five sections with index pages and home; registry drives nav; no Supabase yet |
| M2 | Supabase layer | Migrations applied; RLS verified (anon can read published rows, cannot write); `lab/db.py` and `lab/storage.py` working against real project |
| M3 | Example vertical slice | Ingest → experiment → page works end to end; re-running pipelines is idempotent; `docs/adding-a-page.md` written |
| M4 | Quality | ruff clean; pytest passes; AppTest smoke test renders every registered page; CI green on GitHub |
| M5 | Deploy prep | `docs/deploy.md` checklist complete; I deploy via the Community Cloud dashboard |

Stop at the end of each milestone, summarize what changed, and wait for approval before merging and starting the next.

## 9. Build rules

Copy this section verbatim into `CLAUDE.md` at the start of Phase 2.

### Process
- The approved plan is the contract. New dependencies, schema changes, file moves, or new top-level directories require stopping to ask, with a rationale.
- One milestone at a time. Meet its acceptance criteria, report, wait.
- Read before writing. Small diffs. No speculative abstractions or features not in the plan.
- Log non-obvious decisions in `docs/DECISIONS.md` (date, decision, reason).

### Git
- First action: `git init` and commit `.gitignore` before anything else is staged.
- One branch per milestone (`m1-skeleton`, `m2-supabase`, …). Merge to `main` with `--no-ff` only after I approve.
- Small, single-purpose commits using Conventional Commits: `feat:`, `fix:`, `refactor:`, `test:`, `docs:`, `chore:`, `data:`.
- Before every commit: `ruff format`, `ruff check`, `pytest` pass; review `git status` and `git diff --staged` for secrets and large files.
- Stage files explicitly; never `git add -A` / `git add .` without reviewing the result.
- Never force-push, rewrite `main` history, or commit `.env`, `secrets.toml`, keys, or data files over 1 MB (data belongs in Supabase).
- Create the GitHub remote with `gh` only if authenticated and after I confirm repo visibility; otherwise ask me.

### Secrets & safety
- `.env` (pipelines) and `.streamlit/secrets.toml` (app) are gitignored; commit `.env.example` and `secrets.toml.example` with key names only.
- The service-role/secret key is used only in `pipelines/`. App code must never import from `pipelines/` or reference that key.
- Public data only, with source and license recorded in `datasets`. No confidential, proprietary, or personal data.
- Anything that spends money (model API calls beyond a minimal smoke test, paid tiers) → ask first with a cost estimate.
- Destructive operations (dropping tables, deleting buckets/objects, `rm -rf`, `git reset --hard`) → ask first.

### Code
- Python 3.12, type hints throughout, docstrings on public functions in `lab/` and `pipelines/common/`.
- Pages import only from `lab/` and third-party packages; no page-to-page imports.
- All DB reads go through `lab/db.py` using `st.cache_data(ttl=…)`; clients via `st.cache_resource`.
- Pipelines are idempotent: upsert on natural keys, checksum datasets, skip unchanged work.
- Every run records its git SHA and full config.
- Respect external sources' terms and rate limits (e.g. SEC EDGAR requires a descriptive User-Agent and caps request rate); verify current policy before writing an ingester.
- Prefer native Streamlit components plus Plotly or Altair. No custom JS/CSS hacks.

### Testing
- `pytest` for `lab/` and `pipelines/` with Supabase mocked.
- `streamlit.testing.v1.AppTest` smoke test: every registered page renders without exception, including with empty data.
- A test that the registry and `app_pages/` agree: no orphan page files, no registry entries pointing to missing files.
