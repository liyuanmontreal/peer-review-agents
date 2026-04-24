# Spec: SLURM cluster support for reva

## Goal
Add cluster execution to `reva` so agents can run as SLURM batch jobs on a shared cluster (Mila), complete with automatic chain-on-timeout, without disturbing the existing local tmux flow. Most agents will run locally today; cluster is additive for operators running a long sprint (e.g. the competition window).

## Context

### Current design (local tmux)
- `reva launch --name foo` does three things: assembles the prompt, builds a restart-looping bash script via `cli/reva/tmux.py::build_launch_script`, and hands that script to a detached tmux session (`reva_<name>`).
- The bash script owns **all** of the restart semantics: per-invocation timeout, resume vs. fresh-start decision, session-id extraction, `.api_key` loading, `.env` sourcing. tmux is just a PTY harness so backends like gemini-cli that hate being headless don't get SIGTTIN'd.
- Agent state on disk (`agent_configs/<name>/`): `.api_key`, `.env` (optional), `agent.log`, `prompt.md`, `initial_prompt.txt`, `CLAUDE.md`/`AGENTS.md`/etc., `last_session_id` or `.reva_has_run`, plus markdown reasoning files the agent writes. Backend session stores live under `$HOME` (e.g. `~/.claude/projects/`).
- Recent PR #5 and PR #6 taught us resume is fragile: `--mcp-config` must be re-passed each resume; resume templates need `str.format()` applied; opencode/codex resume must re-send `initial_prompt.txt` or they have no task. **Any cluster path must reuse the exact output of `build_launch_script` — no reimplementation.**

### Target cluster (Mila)
- SLURM. Launches happen from an interactive allocation (user is already inside an `salloc`); submitted `sbatch` jobs run independently of the parent alloc.
- Partition defaults: `main-cpu` (CPU-only, `MaxTime=5-00:00:00`, `PreemptMode=REQUEUE`). Reviewers are API-driven; GPU is wasteful.
- Filesystem: `$HOME=/home/mila/<u>/<user>` is Lustre, persistent, shared across nodes. `$SCRATCH=/network/scratch/<u>/<user>` exists but not needed — agent state is small. Backend session stores in `$HOME/.claude/…` travel with the user across nodes.
- `/usr/bin/tmux` is present on compute nodes, but there's no reason to run tmux inside an sbatch job — nobody attaches.
- `MaxSubmit=1000` jobs/user; `MaxBatchRequeue=5`. Our explicit chain cap (default 3) stays well inside these.
- SLURM auto-requeues preempted jobs on `main-cpu` (`PreemptMode=REQUEUE`). We do **not** need to handle preempt in the chain logic — only wall-time exhaustion.

## Requirements

1. **New `--cluster` flag on `reva launch`.** When set, skip tmux and submit an sbatch job instead. New companion options:
   - `--partition` (default `main-cpu`)
   - `--time` (default `5-00:00:00`, SLURM duration format)
   - `--cpus` (default `4`, → `--cpus-per-task`)
   - `--mem` (default `16G`, → `--mem`)
   - `--max-chain` (default `3`, max number of sbatch jobs in the chain including the first)

   The existing `--duration` flag is **ignored** in cluster mode (chain length × wall-time already bounds total runtime). `--session-timeout` is still honored inside the SLURM job as the per-invocation backend timeout.

2. **Cluster submission lives in a new `cli/reva/cluster.py`** with three entry points:
   - `submit_agent(agent_dir, launch_script, *, partition, time, cpus, mem, max_chain, chain_n=1) -> int` — renders `.reva_cluster.sbatch` into the agent dir, invokes `sbatch`, returns the job ID. Does **not** import or touch `tmux.py`.
   - `list_cluster_jobs() -> list[ClusterJob]` — calls `squeue -u $USER --name=reva_* -h -o '%A|%j|%T|%L'` and parses. Returns name-stripped `ClusterJob(agent_name, job_id, state, time_left)`.
   - `cancel_chain(agent_name) -> int` — writes a `.reva_stop` sentinel into the agent dir, then `scancel`s every job whose `--job-name == reva_<agent_name>`. Returns count cancelled.

3. **Generated sbatch script** (written to `agent_configs/<name>/.reva_cluster.sbatch`):
   ```bash
   #!/bin/bash
   #SBATCH --job-name=reva_<name>
   #SBATCH --partition=<partition>
   #SBATCH --time=<time>
   #SBATCH --cpus-per-task=<cpus>
   #SBATCH --mem=<mem>
   #SBATCH --output=<agent_dir>/cluster.%j.out
   #SBATCH --error=<agent_dir>/cluster.%j.err

   cd <agent_dir>
   rm -f .reva_stop   # clean slate at job start

   # Trap wall-time exhaustion (SIGTERM from SLURM) → chain next job.
   # Skip chain if .reva_stop sentinel is present (user requested stop).
   _chain_next() {
       if [ -f .reva_stop ]; then
           echo "[reva cluster] stop sentinel present, not chaining."
           return
       fi
       if [ "$REVA_CHAIN_N" -ge "$REVA_MAX_CHAIN" ]; then
           echo "[reva cluster] chain limit reached (${REVA_CHAIN_N}/${REVA_MAX_CHAIN}), not chaining."
           return
       fi
       NEXT=$((REVA_CHAIN_N + 1))
       sbatch --dependency=afterany:$SLURM_JOB_ID \
           --export=ALL,REVA_CHAIN_N=$NEXT,REVA_MAX_CHAIN=$REVA_MAX_CHAIN \
           <agent_dir>/.reva_cluster.sbatch
   }
   trap _chain_next EXIT

   REVA_CHAIN_N=${REVA_CHAIN_N:-1}
   REVA_MAX_CHAIN=${REVA_MAX_CHAIN:-<max_chain>}
   export REVA_CHAIN_N REVA_MAX_CHAIN

   bash <agent_dir>/.reva_launch.sh
   ```
   The `.reva_launch.sh` file is the exact output of the existing `build_launch_script` — identical to the tmux path. We reuse it via `create_session`'s current file-writing logic, extracted into a small helper so both paths share it.

4. **`reva stop --name foo --cluster`** calls `cancel_chain(foo)` instead of `kill_session(foo)`. `--all --cluster` cancels every reva cluster chain for the user. Without `--cluster`, behavior is unchanged.

5. **`reva status`** shows cluster jobs alongside tmux sessions. One merged table: columns `NAME`, `BACKEND`, `MODE` (`tmux`/`slurm`), `JOB/SESSION`, `STATE`. An agent with both tmux and cluster runs gets two rows.

6. **Environment propagation**: the sbatch script gets `--export=ALL` plus the explicit `REVA_CHAIN_N`/`REVA_MAX_CHAIN` vars. `.env` + `.api_key` loading happens inside `.reva_launch.sh` exactly as today (unchanged); the backend-specific env-key forwarding in `tmux.py::create_session` is **not** needed inside sbatch since the script runs under the submitter's environment.

7. **Factor out script-file writing** from `tmux.py::create_session`. Move the `.reva_launch.sh` + `.reva_env.sh` writing into a new `cli/reva/launch_script.py::write_launch_files(agent_dir, launch_script)`. `create_session` calls it then does the tmux dance; `cluster.submit_agent` calls it then does sbatch. This is the only refactor to existing code.

8. **Pre-submit validation**. `reva launch --cluster` must fail fast if:
   - `sbatch` is not on `$PATH` (exit 2 with "SLURM not available on this host")
   - `.api_key` missing/empty (same check as local path)
   - `--time` string doesn't match `^(\d+-)?\d{1,2}:\d{2}(:\d{2})?$` (SLURM format)

9. **README update**. New "Running on SLURM (Mila)" section. One-paragraph walkthrough: `reva launch --name foo --cluster` from inside an salloc, default resource envelope, how to stop, how to see logs (same `reva log`, reads `agent.log` on shared FS).

10. **No change to `GLOBAL_RULES.md`, `DEFAULT_INITIAL_PROMPT`, backend templates, resume paths, or MCP config.** Cluster support is purely an outer harness swap.

## Constraints

- Local tmux flow is unchanged. `reva launch --name foo` (no `--cluster`) produces byte-identical output to today.
- `build_launch_script` is **reused verbatim** from cluster path. No duplicate restart loop.
- No new Python dependencies (use `subprocess` for `sbatch`/`squeue`/`scancel` — same pattern as `tmux.py`).
- `KOALA_BASE_URL` behavior unchanged: `.env` is loaded by `.reva_launch.sh` inside the sbatch job exactly as in tmux.
- No login-node daemon. The CLI is fire-and-forget — submission happens once, SLURM owns everything after.
- No k8s, ray, MCP server, or cluster-account provisioning. Only `sbatch`/`squeue`/`scancel`.
- Chain limit is **hard-capped** at whatever `--max-chain` was at first-submit time; the trap reads it from env, so mid-chain changes are ignored by design (prevents runaway reconfiguration).

## Test Plan

New: `tests/test_cluster.py`. All tests mock `subprocess.run` / `shutil.which` — no real SLURM calls.

- `test_submit_agent_renders_sbatch_script`: writes `.reva_cluster.sbatch` with `#SBATCH` headers for partition/time/cpus/mem/job-name; body invokes `.reva_launch.sh`; trap clause present.
- `test_submit_agent_calls_sbatch_with_script_path`: mocked `subprocess.run` receives `["sbatch", ".reva_cluster.sbatch"]` in the agent dir.
- `test_submit_agent_returns_parsed_job_id`: `sbatch` stdout `"Submitted batch job 12345\n"` → function returns `12345`.
- `test_submit_agent_refuses_if_sbatch_missing`: `shutil.which("sbatch")` returns None → raises `RuntimeError`.
- `test_submit_agent_validates_time_format`: bad `--time` value raises before any `subprocess.run`.
- `test_chain_trap_uses_max_chain_env_var`: inspect generated script, assert `$REVA_MAX_CHAIN` appears and `.reva_stop` sentinel check is present.
- `test_cancel_chain_writes_stop_sentinel_then_scancels`: call order verified via mock.
- `test_list_cluster_jobs_parses_squeue_output`: feed a canned `squeue` stdout, assert parsed `ClusterJob` records.
- `test_list_cluster_jobs_returns_empty_when_no_jobs`: `squeue` exits with no matching rows.

Extend `tests/test_cli.py`:
- `test_launch_cluster_invokes_submit_agent_not_create_session`: monkeypatch both; assert `submit_agent` called, `create_session` not called.
- `test_launch_cluster_fails_without_api_key`: existing gate still runs.
- `test_launch_cluster_fails_with_bad_time_format`: surfaces validation error to CLI.
- `test_launch_without_cluster_still_uses_tmux`: regression — tmux path untouched.
- `test_stop_cluster_cancels_chain`: `reva stop --name foo --cluster` calls `cancel_chain`, not `kill_session`.
- `test_status_merges_tmux_and_cluster`: both sources mocked, both rows appear.

Extend `tests/test_tmux.py`: assert that extracting `write_launch_files` leaves `create_session` behavior identical (same on-disk output).

## Acceptance Criteria

- [ ] `reva launch --name foo --cluster` on a machine with `sbatch` in `$PATH` submits a job, prints `Submitted job <id> for agent foo (chain 1/<N>)`, and returns exit 0.
- [ ] The generated `agent_configs/foo/.reva_cluster.sbatch` contains exactly these `#SBATCH` headers at the top (and no others): `--job-name=reva_foo`, `--partition=<partition>`, `--time=<time>`, `--cpus-per-task=<cpus>`, `--mem=<mem>`, `--output=…cluster.%j.out`, `--error=…cluster.%j.err`.
- [ ] `reva launch --name foo --cluster` with no `.api_key` or empty `.api_key` exits non-zero with the same `.api_key missing` message as the local path.
- [ ] `reva launch --name foo --cluster` on a host without `sbatch` exits non-zero with `SLURM not available` in the message.
- [ ] `reva launch --name foo --cluster --time 99-notvalid` exits non-zero with a format error before any `sbatch` call.
- [ ] After `reva stop --name foo --cluster`, `.reva_stop` exists in `agent_configs/foo/` and every job named `reva_foo` has been `scancel`'d. A successor job (if one was already queued) does not run because the trap reads the sentinel.
- [ ] `reva status` prints one row per running tmux session AND one row per running cluster job. Column `MODE` distinguishes `tmux` from `slurm`.
- [ ] The generated `.reva_launch.sh` is byte-identical between the tmux path and the cluster path for the same agent + options (proves we reuse `build_launch_script`).
- [ ] `reva launch --name foo` (no `--cluster`) produces the same tmux session as on `main` today: `tmux has-session -t reva_foo` succeeds, no sbatch files are created.
- [ ] `uv run pytest` passes, with `KOALA_BASE_URL` unset and with `KOALA_BASE_URL=https://staging.koala.science` set.
- [ ] `grep -rn "https://koala.science" cli/reva/cluster.py` returns zero hits (cluster module goes through the existing accessor or `.reva_launch.sh`).
- [ ] README contains a "Running on SLURM (Mila)" section with the default resource envelope (`main-cpu`, 5d, 4 cpus, 16G) documented.
- [ ] End-to-end smoke test on Mila: `reva launch --name smoke --cluster --time 00:10:00 --max-chain 1` from inside an salloc, agent starts, `agent.log` receives backend output, job completes naturally when backend exits, no successor submitted.
