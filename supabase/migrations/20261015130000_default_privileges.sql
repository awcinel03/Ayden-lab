-- Stop new objects in public from being auto-granted to the Data API roles.
-- This project still had the legacy defaults (ALL on new tables to anon and
-- authenticated). From now on every migration must grant explicitly, as
-- 20261015120000_init.sql does. service_role keeps its defaults so pipelines
-- can write to new tables once RLS and grants are in place.

alter default privileges for role postgres in schema public
  revoke all on tables from anon, authenticated;
alter default privileges for role postgres in schema public
  revoke all on sequences from anon, authenticated;
alter default privileges for role postgres in schema public
  revoke all on functions from anon, authenticated;
