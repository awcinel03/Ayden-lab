# Ayden's Lab — Plan

Status: **approved 2026-10-15**. This is the Phase 2 contract (see CLAUDE.md).

## Context
The brief lives at `~/Downloads/BRIEF.md`. `~/aydens-lab` is empty and is not a git repo yet. This plan covers BRIEF §7 items 1–8. Once you approve it, it becomes `docs/PLAN.md` and serves as the contract for Phase 2.

### What I verified (2026-10-15)
| Item | Finding | Source |
|---|---|---|
| `st.navigation` | `st.navigation(pages, *, position="sidebar", expanded=False)`. `pages` can be a dict `{section: [pages]}`. It returns the selected page, which you call `.run()` on. Once any session calls it, `pages/` is ignored. | docs.streamlit.io st.navigation (v1.66.0) |
| `st.Page` | `st.Page(page, *, title, icon, url_path, default=False, visibility="visible")`. **`url_path` cannot contain `/`** (no nested URLs). By default it is inferred from the filename. The default page's path is `""`. | docs st.Page |
| AppTest | Supports `st.navigation` apps. Use `from_file(entrypoint)` → `.run()` → `switch_page(path)` → `.run()`. `at.secrets` can be set. | docs AppTest |
| Streamlit | 1.66.0 (released 2026-10-14, yesterday); 1.65.0 (2026-10-02). Requires Python ≥3.11. | PyPI |
| supabase-py | 2.32.0 (2026-10-02). Requires Python ≥3.9. Docs don't say whether it accepts `sb_publishable_`/`sb_secret_` keys, so I'll test that in M2. | PyPI |
| Supabase Free | 500 MB DB per project, 1 GB Storage, 5 GB egress/month, **50 MB max file upload**, 2 free projects. **Projects pause after 7 days of low activity** and can be restored for up to 90 days. | supabase.com billing-on-free-plan, file-limits, free-project-pausing |
| Supabase keys | Legacy `anon`/`service_role` JWT keys are **deprecated by end of 2026**. The replacements are `sb_publishable_…` (role `anon`, RLS applies) and `sb_secret_…` (role `service_role`, bypasses RLS). | supabase.com api-keys |
| Supabase grants | **Breaking change: from 2026-10-30, new `public` tables on all projects are no longer auto-granted to `anon`/`authenticated`/`service_role`.** Migrations need explicit `GRANT`s. A missing grant returns PostgREST 42501. | supabase changelog #45329 |
| Community Cloud | 0.078–2 CPU, 690 MB–2.7 GB RAM, 50 GB disk (ephemeral). Apps sleep after 12 h without traffic. Python is chosen in Advanced settings (default 3.12). `runtime.txt` is ignored. Forum reports of builds landing on 3.14 regardless. | docs.streamlit.io manage-your-app, deploy |
| SNAP email-Eu-core | 1,005 nodes, 25,571 edges, `.txt.gz` is 79,754 bytes. **No license stated** on the page; citations are Yin et al. KDD'17 and Leskovec et al. TKDD'07. | snap.stanford.edu |
| Local machine | No Python 3.12 (system has 3.9.6, uv has 3.14.7). uv 0.12.3 is present. No `gh`, `brew`, `node`, `psql` or Supabase CLI. **No git user.name/email set.** | local checks |

## 7.1 Final file tree
Changes from BRIEF §3 are marked **Δ**, with the reason.

```
aydens-lab/
├── streamlit_app.py            # set_page_config; build_navigation(REGISTRY) → st.navigation(...).run()
├── .python-version             # Δ "3.12" so uv/local match Community Cloud (CC ignores it; deploy.md says to set it in UI)
├── .streamlit/
│   ├── config.toml             # theme only
│   └── secrets.toml.example    # SUPABASE_URL, SUPABASE_PUBLISHABLE_KEY (names only)
├── app_pages/
│   ├── home.py                 # intro, recent entries, filters (category/type/tag) over the registry
│   ├── network_science/
│   │   ├── index.py            # lab.ui.category_index("network_science")
│   │   └── graph_stats.py      # M3 example entry
│   ├── machine_learning/index.py
│   ├── claude_api/index.py
│   ├── model_testing/index.py
│   └── ai_general/index.py
├── lab/
│   ├── __init__.py
│   ├── registry.py             # Category/EntryType enums, Entry dataclass, ENTRIES; pure Python, no streamlit import
│   ├── nav.py                  # Δ build_navigation(): the only st.Page construction; split out so registry stays importable by pipelines/tests without streamlit
│   ├── db.py                   # M2: cached client (cache_resource) + typed reads (cache_data ttl)
│   ├── storage.py              # M2: artifact download via anon key, cached in memory (cache_data); Δ no disk cache, since CC disk is ephemeral
│   ├── ui.py                   # page_header, category_index, provenance_footer, empty_state
│   └── charts.py               # M3: Altair helpers, created when the first chart needs one (not in M1)
├── pipelines/
│   ├── __init__.py             # Δ so `python -m pipelines.…` works from the repo root
│   ├── common/
│   │   ├── client.py           # secret-key client from .env
│   │   ├── runs.py             # start/finish run, fingerprint, idempotency check
│   │   ├── provenance.py       # sha256, git SHA (refuse to run on a dirty tree unless --allow-dirty)
│   │   └── sync.py             # Δ upsert `experiments` rows from lab.registry (one source of truth, see §7.7)
│   ├── ingest/snap.py
│   └── experiments/network_science/graph_stats.py
├── supabase/
│   ├── migrations/             # Δ named YYYYMMDDHHMMSS_name.sql (Supabase CLI format, not 001_)
│   └── Δ seed.sql dropped: nothing to seed, real data comes from the pipelines
├── tests/
│   ├── test_registry.py        # registry ↔ app_pages agreement, unique slugs/url_paths
│   ├── test_app_smoke.py       # AppTest: every registered page renders, with empty data
│   └── (M2+) test_db.py, test_pipelines_*.py with Supabase mocked
├── docs/  PLAN.md, DECISIONS.md, adding-a-page.md, deploy.md
├── .github/workflows/ci.yml    # ruff format --check, ruff check, pytest
├── BRIEF.md                    # copied in from ~/Downloads
├── CLAUDE.md                   # BRIEF §9 verbatim
├── requirements.txt            # app deploy manifest, pinned
├── requirements-pipelines.txt  # -r requirements.txt + pipeline extras
├── requirements-dev.txt        # Δ -r requirements-pipelines.txt + ruff, pytest (keeps the deploy manifest lean; CI installs this)
├── pyproject.toml              # Δ ruff + pytest config only (no packaging)
├── .env.example                # SUPABASE_URL, SUPABASE_SECRET_KEY
└── .gitignore                  # .env, .streamlit/secrets.toml, .venv, __pycache__, data/, *.parquet, *.gz, .DS_Store
```
Local scratch data for pipelines goes in `data/` (gitignored).

## 7.2 Registry and navigation
```python
class Category(StrEnum):   # value = section label; order = sidebar order
    NETWORK_SCIENCE="Network Science"; MACHINE_LEARNING="Machine Learning"
    CLAUDE_API="Claude API"; MODEL_TESTING="Model Testing"; AI_GENERAL="AI General"
class EntryType(StrEnum): TOOL, EXPERIMENT, FINDING, GRAPH, MODEL, EXPLORATION
class Status(StrEnum):    DRAFT, ACTIVE, COMPLETE, ARCHIVED

@dataclass(frozen=True)
class Entry:
    slug: str                 # globally unique; used as url_path and as experiments.slug
    title: str
    category: Category
    type: EntryType
    path: str                 # relative to repo root, e.g. "app_pages/network_science/graph_stats.py"
    summary: str
    date: datetime.date       # shown on home "recent entries" and in the header
    status: Status = Status.ACTIVE
    tags: tuple[str, ...] = ()
    icon: str | None = None   # material icon, e.g. ":material/hub:"
    published: bool = True    # False = hidden from nav/index/home; synced to experiments.published

CATEGORY_INDEX: dict[Category, str]   # category → index page path
ENTRIES: tuple[Entry, ...]
```
`lab/nav.py::build_navigation()` returns `{"": [home(default=True)], "Network Science": [index, *entries], …}`.
- Every `st.Page` gets an **explicit `url_path`**. The five index files are all called `index.py`, and since `url_path` can't contain `/`, inferred paths would collide. Index pages use `url_path=<category-kebab>` (e.g. `network-science`). Entries use `url_path=slug`.
- A category with no entries still shows its index, so all five sections are always present.
- Only published entries are added (`published=True`).
- `position="sidebar"`, `expanded=True`.
- `streamlit_app.py` is three lines: page config, `pg = st.navigation(build_navigation())`, `pg.run()`.
- `lab.ui.category_index(cat)` reads `ENTRIES` and links to entries with `st.page_link(entry.path)`.

## 7.3 SQL schema, RLS and buckets (M2)
Migration `…_init.sql`:

```sql
create table datasets (
  id bigint generated always as identity primary key,
  slug text unique not null, name text not null, source text not null, source_url text not null,
  license text,                       -- null = "not stated"; shown verbatim in provenance
  citation text,                      -- Δ SNAP asks for citations, not a licence
  storage_path text,                  -- null = not mirrored, lives at source_url only
  format text not null, bytes bigint, sha256 text not null, row_count bigint,
  retrieved_at timestamptz not null, notes text);
create table experiments (
  id bigint generated always as identity primary key,
  slug text unique not null, title text not null, category text not null
    check (category in ('network_science','machine_learning','claude_api','model_testing','ai_general')),
  description text, published boolean not null default false, created_at timestamptz default now());
create table runs (
  id bigint generated always as identity primary key,
  experiment_id bigint not null references experiments on delete restrict,
  status text not null check (status in ('running','succeeded','failed')),
  started_at timestamptz not null default now(), finished_at timestamptz,
  config jsonb not null, git_sha text not null,
  fingerprint text unique not null);  -- Δ sha256(config, git_sha, dataset sha256s): the idempotency key
create table run_datasets (           -- Δ join table instead of a dataset_ids array (keeps FKs, easy RLS)
  run_id bigint references runs on delete cascade, dataset_id bigint references datasets,
  primary key (run_id, dataset_id));
create table results (
  id bigint generated always as identity primary key,
  run_id bigint not null references runs on delete cascade,
  metric text not null, value double precision, unit text, dims jsonb not null default '{}',
  unique (run_id, metric, dims));
create table artifacts (
  id bigint generated always as identity primary key,
  run_id bigint not null references runs on delete cascade,
  kind text not null check (kind in ('plot','table','model','raw_outputs','other')),
  storage_path text unique not null, bytes bigint, sha256 text, content_type text);
create table model_test_runs (        -- Δ 1:1 with runs, for the Model Testing capture list in §4
  run_id bigint primary key references runs on delete cascade,
  provider text not null, model_id text not null, sampling_params jsonb not null,
  system_prompt_sha256 text, prompt_template_version text not null, input_set_id text not null,
  input_tokens bigint, output_tokens bigint, latency_ms_p50 double precision, latency_ms_p95 double precision,
  est_cost_usd numeric(10,4), called_at timestamptz not null);
  -- per-call raw outputs/latency → artifacts(kind='raw_outputs') Parquet
create view published_runs with (security_invoker = true) as …  -- latest succeeded run per published experiment
```
Indexes go on every FK and on `results(run_id, metric)`. SEC Form D needs no schema change: it is a `datasets` row per quarterly file plus Parquet in Storage.

**Grants (required by the 2026-10-30 change):**
`revoke all on all tables in schema public from anon, authenticated;`
`grant select on <each table/view> to anon;` and
`grant select, insert, update, delete on <each table> to service_role;` plus `grant usage` on the identity sequences.

**RLS:** enabled on every table. Anon gets `select` policies only:
- `experiments`: `published`.
- `runs`, `model_test_runs`, `results`, `artifacts`, `run_datasets`: `exists (… join experiments e … where e.published and runs.status='succeeded')`.
- `datasets`: `true`. Provenance metadata is public by design.
- No insert/update/delete policies for anon or authenticated. Secret-key pipelines bypass RLS.

**Buckets, both private** (no public bucket):
- `artifacts`: a `storage.objects` select policy for anon where `bucket_id='artifacts'` and the first path segment (`<experiment_slug>/<run_id>/<file>`) matches a published experiment. The app downloads with the publishable key through `lab/storage.py` (cached). Unpublished work stays private, and nobody can list or guess-fetch drafts, which a public bucket would allow.
- `datasets`: no anon access at all. The app never reads raw datasets, and keeping it private avoids publicly re-hosting data whose licence isn't stated.
- File-size limit is 50 MB per bucket (the Free cap). Allowed MIME types are restricted per bucket.

**RLS check (M2 acceptance):** `tests/rls_check.py`, run manually against the real project with the publishable key. It asserts that published rows are readable, unpublished rows are invisible, and insert/update/delete plus a storage upload are all rejected.

## 7.4 Dependencies
Each pin is the newest stable release at least ~2 weeks old, re-checked on install. Day-old releases are avoided (streamlit 1.66.0, ruff 0.17.x, pyarrow 26).

| Package | Pin | File | Why |
|---|---|---|---|
| streamlit | 1.65.0 | requirements.txt | App framework with `st.navigation` |
| supabase | 2.32.0 | requirements.txt | DB and Storage client (app with publishable key, pipelines with secret key) |
| pandas | 3.0.6 | requirements.txt | Shapes results for display (also a streamlit dependency, pinned explicitly) |
| pyarrow | 25.0.1 | requirements.txt | Reads Parquet artifacts (streamlit dependency, pinned) |
| altair | 6.3.0 | requirements.txt | Charts. Already a streamlit dependency, so no extra install. **Δ Plotly dropped** until a chart needs it. |
| networkx | 3.7 | pipelines | Graph statistics |
| python-dotenv | 1.2.4 | pipelines | Loads `.env` |
| *(httpx)* | from supabase | — | Downloads from SNAP using the HTTP client supabase already pulls in. No `requests`. |
| ruff | 0.16.10 | dev | Lint and format |
| pytest | 9.1.1 | dev | Tests |

Python 3.12 is installed with `uv python install 3.12`, and the venv is created with `uv venv --python 3.12 .venv`. `uv pip install -r requirements-dev.txt` installs everything. uv is used only as an installer; there is no uv.lock, and `requirements*.txt` remain the manifests.

## 7.5 Data-size strategy
The binding Free-tier limits are **5 GB/month egress**, **1 GB Storage** and **50 MB per file**. The 500 MB DB is not the bottleneck.

| Data | Where | Format |
|---|---|---|
| Small public raw file (≤ 50 MB compressed, redistribution not prohibited), e.g. email-Eu-core (80 KB) | Private `datasets` bucket, mirrored as received | Original `.gz` plus sha256 |
| Medium tabular raw (e.g. SEC Form D quarterly TSVs) | Private `datasets` bucket | Converted to zstd-compressed Parquet. The sha256 of both the source and the Parquet are recorded. |
| Anything > 50 MB compressed, or with a licence that forbids redistribution (large SNAP graphs, full EDGAR) | Not stored. `datasets.storage_path=null` | `source_url` + sha256 + `retrieved_at`. The pipeline re-downloads and verifies the checksum. |
| Scalar metrics and small series (< ~5k rows/run, e.g. degree distribution) | `results` table | Long format |
| Larger per-run tables, raw model outputs | `artifacts` bucket | Parquet (zstd) |
| Plots | Not stored. The page renders them from `results`. | — |

App rules:
- The app reads only `results`, metadata and artifacts under 5 MB. It never reads raw datasets.
- Reads are cached with `st.cache_data(ttl=3600)`.
- A budget check in `pipelines/common` warns when the bucket total exceeds 750 MB.
- Git holds no data, and the existing 1 MB rule is enforced by a pre-commit check in CLAUDE.md practice.

## 7.6 Milestones and acceptance criteria
**M1 Skeleton** (`m1-skeleton`, no Supabase):
- `streamlit run streamlit_app.py` starts with no errors or warnings in the console.
- The sidebar shows Home plus five sections, each with its index page, in registry order.
- Every page URL is unique, and each category index shows an empty state.
- Home shows the intro and its filters with a "no entries yet" state.
- `lab/ui.page_header` exists.
- `tests/test_registry.py` and `tests/test_app_smoke.py` pass.
- `ruff format --check` and `ruff check` are clean.
- `docs/DECISIONS.md` is started.

**M2 Supabase** (`m2-supabase`):
- Migrations are applied to the real project.
- `tests/rls_check.py` passes against the real project.
- The `lab/db.py` and `lab/storage.py` read helpers return typed results and show `st.info` (no traceback) when data is empty or the connection fails.
- `pipelines/common` is in place with mocked unit tests.
- The format of the new publishable and secret keys works with supabase-py 2.32.0. Otherwise the fallback is logged in DECISIONS.md.

**M3 Vertical slice:**
- `python -m pipelines.ingest.snap email-Eu-core` uploads the file and the `datasets` row. A second run does nothing ("unchanged, sha256 match").
- `python -m pipelines.experiments.network_science.graph_stats` writes a run with nodes, edges, density, avg/max degree, clustering, components and the degree distribution. A second run on the same commit is skipped by fingerprint.
- The page follows the full §5 contract, with an Altair log-log degree distribution and provenance linking to `github.com/<repo>/blob/<git_sha>/…`.
- `docs/adding-a-page.md` is written.

**M4 Quality:**
- ruff is clean.
- pytest passes, including the AppTest smoke test over every registered page with the db layer mocked to empty.
- The registry/app_pages agreement test passes.
- `ci.yml` is green on GitHub. This needs the remote (open question 3).

**M5 Deploy prep:**
- `docs/deploy.md` covers: Python 3.12 in Advanced settings (and checking the build log really shows 3.12, given the 3.14 reports), the secrets to paste, the custom subdomain, the post-deploy smoke checklist, and the keep-alive.

## 7.7 Where I think the brief is wrong or incomplete
1. **Missing Data API grants would break the app on 2026-10-30.** RLS alone is no longer enough. Every migration has to `GRANT` explicitly (§7.3), and the M2 RLS check covers the grants as well.
2. **"anon key" / "service-role key" are being retired by end of 2026.** Use `sb_publishable_` and `sb_secret_` from day one, and name the variables `SUPABASE_PUBLISHABLE_KEY` and `SUPABASE_SECRET_KEY`. The §9 rules still read correctly with the new key names.
3. **A Free project will pause after 7 days of low activity.** A personal research site can easily go a week without visitors. On top of that, the Community Cloud app sleeps after 12 h. The site would then show errors until someone restores the project. Fix: a scheduled GitHub Action that makes one read with the publishable key every 3 days, added in M4/M5. This is a new file, so it needs your OK (question 2).
4. **Duplicate metadata in the registry and the `experiments` table.** Slug, title, category, description and published would drift apart. Fix: the registry is the single source of truth, and `pipelines/common/sync.py` upserts `experiments` from it. Pipelines may import `lab.registry` (pure Python). The ban is only on the app importing `pipelines`.
5. **Five files named `index.py` collide on `url_path`.** This is solved by explicit url_paths (§7.2) and needs no file renames.
6. **"Decide public vs signed URL" has a third, better option.** Private buckets plus an anon storage policy scoped to published experiments (§7.3). Signed URLs would need the app to hold a key that can sign for any object.
7. **`dataset_ids` as an array** loses referential integrity and makes RLS awkward. I replaced it with a `run_datasets` join table.
8. **Runs need an idempotency key.** "Skip unchanged work" requires a natural key. I added `runs.fingerprint`.
9. **`lab/storage.py` "local cache" on disk** doesn't survive on Community Cloud, where disk is ephemeral and wiped on reboot. It uses in-memory `st.cache_data` instead.
10. **Plotly *or* Altair:** I picked Altair alone. It is already installed with streamlit, which means one less dependency and one charting idiom across pages.
11. **`requirements-dev.txt`.** Otherwise ruff and pytest have no home: they shouldn't be in the deploy manifest, and they aren't pipeline dependencies.
12. **Licence for the example dataset.** SNAP states no licence, only citations. email-Eu-core is also *anonymised email between real people*, which brushes against "no personal data". The IDs are integers and it is widely used, so I think it's fine, but it's your call. An alternative is `ca-GrQc` (arXiv co-authorship).
13. **Mirroring raw datasets "is the system of record"** only works within the 1 GB / 50 MB limits. The tiered strategy in §7.5 makes source-URL-plus-checksum a first-class option.
14. `seed.sql` dropped (nothing to seed). Migration names follow the Supabase CLI timestamp format, not "numbered".

## Execution after approval
1. Set up the repo:
   - `git init -b main`.
   - Write `.gitignore` and commit it as `chore: add .gitignore`. Nothing else is staged.
   - Copy `~/Downloads/BRIEF.md` into the repo.
   - Write `docs/PLAN.md` (this plan) and `CLAUDE.md` (BRIEF §9 verbatim, from "### Process" through the end of Testing).
   - Commit as `docs: project plan and build rules`.
2. On branch `m1-skeleton`:
   - `uv python install 3.12`, create the venv and install the dependencies.
   - Build the M1 files.
   - Commit in small steps, each run through ruff and pytest.
   - Stop and report.

## Verification (M1)
- `.venv/bin/streamlit run streamlit_app.py --server.headless true`, then curl `localhost:8501` and each url_path to check for health. You can also check the sidebar visually.
- `.venv/bin/pytest -q` (registry agreement and AppTest over Home, the five indexes, and empty states).
- `.venv/bin/ruff format --check . && .venv/bin/ruff check .`

## Open questions
Resolved 2026-10-15:
- Q1 → repo-local identity is `Ayden Cinel <313170034+awcinel03@users.noreply.github.com>` (GitHub noreply).
- Q3 → the remote is `https://github.com/awcinel03/Ayden-lab.git` (**private**). Provenance links to GitHub will only resolve for the owner while the repo stays private.
- Q7 → BRIEF.md is committed.

Still open:
1. **Git identity:** no `user.name`/`user.email` is set. May I set them repo-locally to your personal email, or do you want to use a different email (e.g. a GitHub noreply address)?
2. **Keep-alive workflow** (§7.7 item 3): OK to add `.github/workflows/keepalive.yml` in M4? It needs `SUPABASE_URL` and the publishable key as GitHub secrets.
3. **GitHub:** `gh` isn't installed. Should I install it with uv/pipx or by downloading the binary, or will you create the remote yourself? And should the repo be public or private? (A public repo is needed for the provenance links to be useful.)
4. **Applying migrations in M2:** no Supabase CLI or psql here. The options are (a) installing the Supabase CLI binary, (b) you pasting the migrations into the SQL editor, or (c) adding `psycopg` to the pipeline dependencies and a small `apply_migrations` script. I recommend (a).
5. **Example dataset:** keep email-Eu-core, or switch to ca-GrQc (§7.7 item 12)?
6. **Supabase project:** do you already have one, and is it on the Free plan? Is it new (created after 2026-05-30, so the grant change already applies)?
7. Should BRIEF.md be committed in the docs commit, alongside PLAN.md and CLAUDE.md? My default is yes.

## Addendum (2026-10-15): pre-approved for the network-science entries

The owner approved these in advance for the Enron, C. elegans and wine flavor entries (`sources/network-project-prompts.md`). Sessions don't need to stop and ask about them, but must still log each one in `docs/DECISIONS.md`.

1. **Charts:** use whichever library is better for the interactive network visualizations. Plotly is approved for app pages. Pin it in `requirements.txt` at the newest stable release at least ~2 weeks old. Altair stays fine for simple charts.
2. **Dependencies:** add whatever a faithful port needs (e.g. `python-louvain`, `scipy`, `numpy`). Pin each one. App-only packages go in `requirements.txt`, pipeline-only packages in `requirements-pipelines.txt`. The app must never need a pipeline package.
3. **Branches:** `feat/*` branch names from the prompts are allowed. The session adds `feat/*` to the push triggers in `.github/workflows/ci.yml`.
4. **Discrepancies:** follow each prompt's instructions as written. Deterministic values must match exactly and stochastic values must be close. On any mismatch, stop and report with a diagnosis, and never tune to force a match. Recompute the disputed log-log fit (slope 0.06, R² 0.010) and report the true value.

Everything else in CLAUDE.md still applies: stop at milestones, run the pre-commit checks, follow the secrets rules, and merge only after approval.
