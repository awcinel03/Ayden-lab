-- Initial schema for Ayden's Lab. See docs/PLAN.md §7.3.
-- The app reads with the publishable key (role anon); pipelines write with the
-- secret key (role service_role, bypasses RLS). Never edit once applied.

-- ---------------------------------------------------------------- tables

create table public.datasets (
  id bigint generated always as identity primary key,
  slug text not null unique,
  name text not null,
  source text not null,
  source_url text not null,
  license text,          -- null = not stated by the source
  citation text,
  storage_path text,     -- null = not mirrored; lives at source_url only
  format text not null,
  bytes bigint,
  sha256 text not null,
  row_count bigint,
  retrieved_at timestamptz not null,
  notes text
);

create table public.experiments (
  id bigint generated always as identity primary key,
  slug text not null unique,
  title text not null,
  category text not null check (
    category in ('network_science', 'machine_learning', 'claude_api', 'model_testing', 'ai_general')
  ),
  description text,
  published boolean not null default false,
  created_at timestamptz not null default now()
);

create table public.runs (
  id bigint generated always as identity primary key,
  experiment_id bigint not null references public.experiments on delete restrict,
  status text not null check (status in ('running', 'succeeded', 'failed')),
  started_at timestamptz not null default now(),
  finished_at timestamptz,
  config jsonb not null,
  git_sha text not null,
  fingerprint text not null unique  -- sha256 of config + git_sha + dataset sha256s
);
create index runs_experiment_id_idx on public.runs (experiment_id);

create table public.run_datasets (
  run_id bigint not null references public.runs on delete cascade,
  dataset_id bigint not null references public.datasets on delete restrict,
  primary key (run_id, dataset_id)
);
create index run_datasets_dataset_id_idx on public.run_datasets (dataset_id);

create table public.results (
  id bigint generated always as identity primary key,
  run_id bigint not null references public.runs on delete cascade,
  metric text not null,
  value double precision,
  unit text,
  dims jsonb not null default '{}',
  unique (run_id, metric, dims)
);
create index results_run_id_metric_idx on public.results (run_id, metric);

create table public.artifacts (
  id bigint generated always as identity primary key,
  run_id bigint not null references public.runs on delete cascade,
  kind text not null check (kind in ('plot', 'table', 'model', 'raw_outputs', 'other')),
  storage_path text not null unique,  -- <experiment_slug>/<run_id>/<file> in bucket "artifacts"
  bytes bigint,
  sha256 text,
  content_type text
);
create index artifacts_run_id_idx on public.artifacts (run_id);

create table public.model_test_runs (
  run_id bigint primary key references public.runs on delete cascade,
  provider text not null,
  model_id text not null,
  sampling_params jsonb not null,
  system_prompt_sha256 text,
  prompt_template_version text not null,
  input_set_id text not null,
  input_tokens bigint,
  output_tokens bigint,
  latency_ms_p50 double precision,
  latency_ms_p95 double precision,
  est_cost_usd numeric(10, 4),
  called_at timestamptz not null
);

-- Latest succeeded run per experiment. security_invoker so RLS applies to the caller.
create view public.latest_runs with (security_invoker = true) as
select distinct on (r.experiment_id)
  e.slug as experiment_slug, r.*
from public.runs r
join public.experiments e on e.id = r.experiment_id
where r.status = 'succeeded'
order by r.experiment_id, r.finished_at desc nulls last, r.id desc;

-- ---------------------------------------------------------------- grants
-- New public tables are no longer auto-exposed to the Data API (Supabase,
-- 2026-10-30), so every grant is explicit. anon: read only. authenticated: none.

revoke all on all tables in schema public from anon, authenticated;

grant select on
  public.datasets, public.experiments, public.runs, public.run_datasets,
  public.results, public.artifacts, public.model_test_runs, public.latest_runs
to anon;

grant select, insert, update, delete on
  public.datasets, public.experiments, public.runs, public.run_datasets,
  public.results, public.artifacts, public.model_test_runs
to service_role;
grant select on public.latest_runs to service_role;
grant usage on all sequences in schema public to service_role;

-- ---------------------------------------------------------------- RLS
-- anon may select published rows only. No write policies exist for anon or
-- authenticated, so all writes from those roles are denied.

alter table public.datasets enable row level security;
alter table public.experiments enable row level security;
alter table public.runs enable row level security;
alter table public.run_datasets enable row level security;
alter table public.results enable row level security;
alter table public.artifacts enable row level security;
alter table public.model_test_runs enable row level security;

-- Provenance metadata is public by design.
create policy "anon reads datasets" on public.datasets
  for select to anon using (true);

create policy "anon reads published experiments" on public.experiments
  for select to anon using (published);

create policy "anon reads succeeded runs of published experiments" on public.runs
  for select to anon using (
    status = 'succeeded'
    and exists (
      select 1 from public.experiments e where e.id = experiment_id and e.published
    )
  );

-- Child tables are visible exactly when their run is (the runs policy applies inside).
create policy "anon reads visible run_datasets" on public.run_datasets
  for select to anon using (exists (select 1 from public.runs r where r.id = run_id));

create policy "anon reads visible results" on public.results
  for select to anon using (exists (select 1 from public.runs r where r.id = run_id));

create policy "anon reads visible artifacts" on public.artifacts
  for select to anon using (exists (select 1 from public.runs r where r.id = run_id));

create policy "anon reads visible model_test_runs" on public.model_test_runs
  for select to anon using (exists (select 1 from public.runs r where r.id = run_id));

-- ---------------------------------------------------------------- storage
-- Both buckets private. 50 MB is the Free-plan per-file cap.

insert into storage.buckets (id, name, public, file_size_limit, allowed_mime_types)
values
  ('datasets', 'datasets', false, 52428800,
   array['application/gzip', 'application/vnd.apache.parquet', 'application/octet-stream',
         'text/plain', 'text/csv', 'text/tab-separated-values', 'application/zip']),
  ('artifacts', 'artifacts', false, 52428800,
   array['application/vnd.apache.parquet', 'application/octet-stream', 'application/json',
         'image/png', 'image/svg+xml', 'text/plain', 'text/csv']);

-- anon may read artifacts of published experiments; path is <experiment_slug>/<run_id>/<file>.
-- No policy on "datasets": raw data is never served by the app.
create policy "anon reads artifacts of published experiments" on storage.objects
  for select to anon using (
    bucket_id = 'artifacts'
    and exists (
      select 1 from public.experiments e
      where e.slug = (storage.foldername(name))[1] and e.published
    )
  );
