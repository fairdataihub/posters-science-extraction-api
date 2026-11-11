# Environment features and extraction batches

One API deployment serves both environments, with one background worker and
one GPU extraction at a time. The website creates jobs; the worker schedules
them.

## Configuration

| API setting | Initial deployment value | Purpose |
| --- | --- | --- |
| `STAGING_FEATURES` | `bulk_extraction,batch_completion_email` | Registered features for jobs from the staging database |
| `PRODUCTION_FEATURES` | empty | Registered features for jobs from the production database |
| `SINGLE_JOBS_PER_BULK` | `3` | Single jobs per reserved bulk turn when both classes have work |
| `RESEND_API_KEY` | Resend key | Sends batch completion emails; without it they are skipped |
| `STAGING_SITE_URL` | empty | Dashboard link host for staging emails, default `https://sandbox.posters.science` |
| `PRODUCTION_SITE_URL` | empty | Dashboard link host for production emails, default `https://posters.science` |

Feature values are comma-separated names from `feature_flags.KNOWN_FEATURES`.
Use `none` to explicitly disable every feature, including when a Compose or
workflow default would otherwise enable staging features. Unknown names and
invalid scheduling weights fail configuration validation.

The deployment workflow reads these from GitHub repository **secrets**.
Docker Compose reads the resulting server `.env` file. Changing a secret takes
effect on redeployment; flags are loaded when the worker starts.

The corresponding posters-science setting is the server-only `NUXT_FEATURES`.
Its staging deployment enables `bulk_extraction`; production defaults to empty.
The website enforces admission. The API's flag context is for processing
features and does not cancel already accepted jobs when admission is disabled.

## Scheduling

`scheduler.QueueScheduler` chooses the class first, then the environment:

1. Prefer singles until the configured weight is reached.
2. Reserve one bulk poster, then start counting singles again.
3. Rotate staging/production independently within each class.
4. Skip empty queues immediately. Idle polls do not reset fairness.
5. Rotate bulk batch IDs within each environment, preserving FIFO within a batch.

Every dispatch processes one poster, not an entire batch. A running extraction
finishes before priority is reconsidered. Weight is measured in jobs, not time;
an unusually slow poster still delays subsequent work. Counters reset on process
restart. PostgreSQL row locks prevent two claimers from claiming the same job,
but do not coordinate model memory across processes.

Jobs are claimed with `FOR UPDATE SKIP LOCKED`. `job_queue.py`
detects batch columns on each connection. Unmigrated databases continue to work
as single-only queues. The worker never treats an unfinished upload as eligible
extraction work. Uploads abandoned for 15 minutes become failed jobs with an
upload retry prompt. Queued jobs are not subject to this upload timeout.

## Batch completion emails

With `batch_completion_email` enabled for an environment, the worker emails a
batch owner once every job in the batch has finished. `batch_notifications.py`
sweeps for finished batches at the end of every cycle, idle cycles included, so
batches finished by the upload timeout or stuck-job cleanup are covered too.

- A batch is finished when it has at least one job and none is incomplete.
- Each batch row is locked with `FOR UPDATE SKIP LOCKED` while its email is
  sent, and `"completionEmailSentAt"` is set in the same transaction, so each
  batch is emailed at most once. Retrying a failed file later sends nothing.
- Network errors, 429s and 5xx responses roll back and retry on a later sweep.
  Other Resend rejections, such as an invalid address, mark the batch and are
  not retried.
- Without `RESEND_API_KEY`, or on a database without the column, the sweep does
  nothing and leaves batches unmarked.
- Logs record the batch id and recipient domain, never the full address.

The column comes from the posters-science migration
`20261111120000_add_extraction_batch_completion_email`. Apply it before enabling
the flag.

## Adding a staging-first feature

1. Add its name to `KNOWN_FEATURES` here and `FEATURE_NAMES` in posters-science
   if both services use it.
2. Reuse shared functions. If extraction behavior changes, branch in
   `job_worker.extract_for_environment(local_path, context)` using
   `context.enabled("feature_name")`.
3. Protect the website server entry point with
   `requireFeature("feature_name", event)`. Use `/api/features` for UI visibility.
4. Enable the name only in staging's feature lists. Deploy API changes from main
   and website changes from staging.
5. Verify the staging behavior and unchanged production behavior. Enable the
   production flags only when ready, then retire temporary branches and flags
   after the feature is stable everywhere.

The context comes from the configured database target, never a job field,
storage prefix, browser flag, or `staging=true` parameter. Do not mutate
`os.environ` per job. Model configuration and the installed poster2json version
are shared across both environments; feature flags do not isolate dependencies.

## Safe deployment order

1. Deploy this API to main first. It supports the old production schema.
2. Deploy posters-science staging, whose migration creates `ExtractionBatch`
   and adds job metadata with backward-compatible defaults.
3. Test staging single uploads, bulk progress, failed-file retries and a normal
   production single upload. Confirm that only one extraction runs at a time.
4. Leave production bulk disabled. A production website/schema rollout is a
   separate step and is not required for the shared API to process old jobs.

Never enable bulk submission against an old worker: it cannot honor priorities.
If rolling back the API, first disable new bulk submissions and let outstanding
bulk jobs drain. Older workers ignore classification and could run bulk jobs FIFO.
Schema rollback by dropping columns is unnecessary and would lose batch history.
