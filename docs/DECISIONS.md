# Decision log

Plan-level decisions are in [PLAN.md](PLAN.md) §7.7. This log records decisions made during the build.

| Date | Decision | Reason |
|---|---|---|
| 2026-10-15 | GitHub remote is `awcinel03/Ayden-lab`, **private** for now. | Owner's choice. Provenance links to code at a commit SHA will only resolve for the owner until the repo is public. |
| 2026-10-15 | Repo visibility changed to **public** (supersedes the row above and the private note in PLAN.md Q3). | Provenance links to code at a run's commit SHA now resolve for every reader. CI gets unlimited Actions minutes. |
| 2026-10-15 | Commit identity is the GitHub noreply address. The 8 unpushed commits were rewritten to use it, with the owner's approval and with trees verified identical apart from scrubbing the email from PLAN.md. | The repo is public, so the personal email should not appear in history. |
| 2026-10-15 | Dependencies are added in the milestone that first uses them. M1 pins only streamlit, pandas, pyarrow, altair, ruff and pytest. supabase arrives in M2; networkx and python-dotenv arrive in M2/M3 along with `requirements-pipelines.txt`. | No unused dependencies on a branch. The pinned versions match PLAN §7.4. |
| 2026-10-15 | `CATEGORY_INDEX` dict replaced by `Category.index_path` / `.url_path` / `.key` properties. | Derived from the enum, so the mapping can't fall out of step with the categories. |
| 2026-10-15 | `lab/nav.py` passes absolute paths (`REPO_ROOT / path`) to `st.Page`. | Page resolution no longer depends on which script is the main one, so AppTest probes and the real app behave the same. |
| 2026-10-15 | Navigation is tested inside `AppTest.from_function`, not in plain pytest. | `st.Page` fails outside a script run context (`AttributeError: _url_path` in 1.65.0). |
| 2026-10-15 | Category index pages are titled "Overview" inside their section. | The section header already names the category. |
| 2026-10-15 | `.streamlit/secrets.toml.example`, `.env.example` and `lab/ui.provenance_footer` deferred to M2/M3. | They belong to the milestones that read secrets and data. |
