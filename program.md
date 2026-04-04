# comma-video-compression autoresearch

This project is set up for autonomous experiment loops on the compression strategy.

## Setup

To set up a new run, work with the user to:

1. **Agree on a run tag**: propose a date-based tag (for example `apr4`).
2. **Create a fresh branch**: `git checkout -b autoresearch/<tag>` from current `main`.
3. **Read in-scope files**:
   - `README.md` (challenge context)
   - `compress.py` (the only file to edit)
   - `eval.py` (fixed evaluator, do not edit)
   - `program.md` (this workflow)
4. **Verify tooling**:
   - Python environment has project deps (`torch`, `av`, etc.)
   - `ffmpeg` is installed and in `PATH`
5. **Initialize results log**:
   - Ensure `results.tsv` exists with the header row.
6. **Confirm setup** and then start experiments.

## Experiment Scope

**Edit only:** `compress.py`

**Do not edit:**
- `eval.py`
- `evaluate.py`, `frame_utils.py`, `modules.py`
- model weights or evaluation data
- dependency definitions (`pyproject.toml`, `uv.lock`)

The goal is to minimize challenge score:

`score = 100 * segnet_dist + sqrt(10 * posenet_dist) + 25 * rate`

Lower is better.

## Running One Experiment

Run exactly this (redirected) command:

```bash
./.venv/bin/python eval.py > run.log 2>&1
```

Then extract metrics:

```bash
grep -E '^score:|^segnet_dist:|^posenet_dist:|^rate:|^archive_bytes:' run.log
```

If grep is empty, treat as a crash and inspect:

```bash
tail -n 80 run.log
```

## Results Logging

Log every attempt in `results.tsv` (tab-separated):

```
timestamp	commit	score	segnet_dist	posenet_dist	rate	archive_bytes	status	description
```

`status` must be one of:
- `improved` (score strictly lower than best prior score)
- `not_improved` (equal or higher score)
- `crash` (run failed to produce score)

## Commit Policy (Important)

Every experiment must produce exactly one commit, and every commit is kept.

- No `git reset --hard`
- No reverting failed experiments
- No dropping experiments from history

Each experiment commit includes:
- `compress.py`
- `results.tsv`

Recommended commit message:

```text
exp: <status> score=<score> <short description>
```

## Loop

Loop indefinitely until manually interrupted:

1. Inspect current best score from `results.tsv`.
2. Implement one concrete idea in `compress.py`.
3. Run `./.venv/bin/python eval.py > run.log 2>&1`.
4. Parse score/metrics (or classify as crash).
5. Append one row to `results.tsv`.
6. Commit once for that experiment.
7. Repeat.

The first experiment should be an unchanged baseline run of current `compress.py`.
