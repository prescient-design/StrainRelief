# dag

## Getting started

### Installing dependencies

This project depends on the parent [`strain-relief`](../) package via a local
editable source (`[tool.uv.sources]` in `pyproject.toml`), so changes to the
parent's `src/strain_relief` are picked up immediately.

Ensure [`uv`](https://docs.astral.sh/uv/) is installed (see their
[installation docs](https://docs.astral.sh/uv/getting-started/installation/)).

```bash
uv venv --python 3.11                                    # strain-relief requires 3.11
uv pip install "torch==2.8.0"                            # torch-cluster needs torch at build time
uv pip install hatchling editables                       # build tools (--no-build-isolation)
uv pip install -e ".[dev]" --no-build-isolation          # this project + strain-relief + dg CLI
uv pip install "antlr4-python3-runtime>=4.13,<4.14"      # Dagster needs antlr 4.13, not omegaconf's 4.9
```

Then activate the virtual environment:

| OS | Command |
| --- | --- |
| MacOS | ```source .venv/bin/activate``` |
| Windows | ```.venv\Scripts\activate``` |


#### Verify the install

```bash
python -c "import dag, strain_relief, dagster, torch_cluster; print('OK')"
dagster definitions validate -m dag.definitions   # or: dg check defs
```

### Running the pipeline

One job, `strain_relief`: `input_df → docked_mols → conformers → local_minima → global_minima →
aggregate_results → plot_results`. The compute-heavy assets (`conformers`, `local_minima`,
`global_minima`) are partitioned into chunks of `CHUNK_SIZE` ligands (env var, default 50) so each
chunk runs as its own process/run and the calculator model loads once per chunk.

Run from `dag/` with the venv active and a persistent `DAGSTER_HOME` (so chunk partitions persist):

```bash
export DAGSTER_HOME=$PWD/.dagster_home
```

**Single chunk** (input fits in one `CHUNK_SIZE`) — one command runs the whole job:

```bash
dg launch --job strain_relief --partition 0 --config run.yaml
```

**Many chunks** (parallel) — use a backfill in the UI (`aggregate_results` fans in every chunk, so
it must run after all chunks; multi-run partition ranges need the daemon, not `dg launch`):

```bash
dg dev   # http://localhost:3000
```

1. Materialize `input_df` + `docked_mols` once — this registers the chunk partitions.
2. **Backfill** `strain_relief` over the chunk range — Dagster runs the compute assets one run per
   chunk (in parallel) and `aggregate_results` / `plot_results` once at the end.

Point `input_df` / `docked_mols` at your own parquet and switch calculator/optimiser in `run.yaml`.
Change the chunk size with the `CHUNK_SIZE` env var (default 50).

### Where outputs are stored

Layout: `data/<asset>/result[_<chunk>].<ext>` (the `_<chunk>` suffix is on the partitioned compute
assets only).

| Asset | Location |
| --- | --- |
| `input_df` | `data/input_df/result.parquet` |
| `conformers` / `local_minima` / `global_minima` | `data/<asset>/result_<chunk>.pt` |
| `aggregate_results` | `data/aggregate_results/result.parquet` |
| `output` (final, human-readable) | `data/output.parquet` |

`docked_mols` and `plot_results` use Dagster's default IO manager (instance storage).

## Learn more

To learn more about this template and Dagster in general:

- [Dagster Documentation](https://docs.dagster.io/)
- [Dagster University](https://courses.dagster.io/)
- [Dagster Slack Community](https://dagster.io/slack)
