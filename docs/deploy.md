# Deploying to Streamlit Community Cloud

Checklist for the first deploy and later checks. The app is read-only: it needs the **publishable** key only. The secret key never leaves your machine's `.env`.

## Before you deploy

- [ ] `main` is pushed and the latest **CI** run on `main` is green: `gh run list --branch main --limit 1`
- [ ] The Supabase project is active (not paused): Dashboard → project → status shows *Healthy*.
- [ ] The example entry has a published run. The page should render locally with `.venv/bin/streamlit run streamlit_app.py`.
- [ ] Have your Supabase project URL and **publishable** key ready (Dashboard → Project Settings → API Keys). Copy them from the dashboard, or from your local `.streamlit/secrets.toml`. **Not** from `.env`, which also holds the secret key.

## Deploy

1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub (`awcinel03`).
2. Click **Create app**, then **Yup, I have an app**.
3. Fill in:
   - **Repository:** `awcinel03/Ayden-lab`
   - **Branch:** `main`
   - **Main file path:** `streamlit_app.py`
   - **App URL:** pick a subdomain, e.g. `aydens-lab`. You can change it later in App settings.
4. Click **Advanced settings**:
   - **Python version:** `3.12`. This must match CI and `.python-version`, and you can't change it without deleting and redeploying the app.
   - **Secrets:** paste exactly these two lines with your values:
     ```toml
     SUPABASE_URL = "https://mxirjgkdtqfezgmycjwd.supabase.co"
     SUPABASE_PUBLISHABLE_KEY = "sb_publishable_…"
     ```
     Never paste `SUPABASE_SECRET_KEY` here.
5. Click **Save** and watch the build log on the right.

Community Cloud installs from `requirements.txt`, the first dependency file it finds; `pyproject.toml` here only configures tools. `requirements-pipelines.txt` and `requirements-dev.txt` aren't read. A clean install of `requirements.txt` alone was verified to render every page.

## Check the build log

- [ ] It shows **Python 3.12.x**. There are reports of builds landing on 3.14 regardless of the setting. If you see that, delete the app and redeploy with 3.12 selected.
- [ ] It installs `streamlit==1.65.0` and `supabase==2.32.0` without errors.
- [ ] No traceback after "You can now view your Streamlit app".

## Smoke test the live app

- [ ] The home page loads and lists *email-Eu-core: basic stats and degree distribution*.
- [ ] The sidebar shows Home and all five sections, each with an **Overview** page.
- [ ] The example page (`/email-eu-core-stats`) shows 1,005 nodes and 25,571 edges, the degree chart, the takeaways, and the provenance section.
- [ ] Switching the degree chart between total, in and out works.
- [ ] A provenance code link opens the file on GitHub at the run's commit, not a 404.
- [ ] Open the app in a private window: it must be viewable without signing in. Check App settings → Sharing; public is the default for public repos.

## After deploying

- **Code changes:** pushing to `main` redeploys automatically. Dependency changes take a few minutes to install.
- **New data:** run the pipelines locally (see `docs/adding-a-page.md`). The app picks up new runs within an hour (cache TTL). To refresh now, choose **Reboot app** from the app menu.
- **Rotating the publishable key:** create a new one in Supabase, then update it in App settings → Secrets, in your local `.streamlit/secrets.toml`, and in the GitHub secret: `gh secret set SUPABASE_PUBLISHABLE_KEY`.

## If something breaks

| Symptom | Likely cause | Fix |
|---|---|---|
| Pages show "No published results yet" | Supabase project paused, or wrong secrets | Restore the project in the Supabase dashboard (possible for 90 days after a pause), then check the two secrets. |
| Sleeping-app page for visitors | No traffic for 12 h | Expected on Community Cloud. Any visitor can wake it. |
| Keep-alive stopped running | GitHub disables scheduled workflows after 60 days with no repo activity | Actions → *Supabase keep-alive* → **Enable workflow**, then **Run workflow**. |
| `ModuleNotFoundError` in build log | A package the app imports isn't in `requirements.txt` | Add it pinned to `requirements.txt` (not the pipelines file) and push. |
| "Oh no." error page | Resource limits (about 2.7 GB RAM) or a crash | Check the logs. Apps should read only small results, never raw datasets. |
