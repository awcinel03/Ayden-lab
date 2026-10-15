# CLAUDE.md — Build rules

Copied verbatim from BRIEF.md §9. The approved plan is docs/PLAN.md.

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
