# Repository Guidelines

## Project Structure & Module Organization

Core code lives under `src/`: models and streaming state are in `src/models/`, dataset loaders in `src/dataset/`, and losses, metrics, configuration helpers, and distributed utilities in `src/utils/`. Training entry point `main_slarm.py` and the shared `engine_tools.py` stay at the repository root (imported as top-level modules); the user-facing launcher scripts (`inference_stream.py`, `eval_stream25_base.py`, `render_stream25_base.py`, `train_stream25_base.py`) live in `scripts/`, and their one-line `bash` wrappers live in `run_sh/`. Experiment YAML files belong in `configs/`. Helper libraries and regression tools (`compare_dump.py` / `compare_report.py`) live in `tools/`. Treat `third_party/` as vendored code. `raw_data/`, `data/`, `work_dirs/`, and `ckpts/` contain local or generated artifacts and should not be committed.

## Build, Test, and Development Commands

Create the supported Python environment and install CUDA dependencies as documented in `README.md`:

```bash
conda create -n SLARM python=3.10 -y
pip install -r requirements.txt
```

Train upstream SLARM with `torchrun --nproc_per_node=1 main_slarm.py --config configs/<experiment>.yaml`. Use `bash run_sh/train_stream25_base.sh stereo` or `triview` for the retained full Stream25 runs. Keep generated datasets, token caches, and checkpoints out of Git.

## Coding Style & Naming Conventions

Use Python 3.10, four-space indentation, `snake_case` for functions/modules, `PascalCase` for classes, and uppercase names for constants. YAML keys and test fixtures use lowercase `snake_case`. No repository-wide formatter is configured; match adjacent code, keep imports explicit, add type hints to new public helpers, and run `git diff --check` before committing.

## Commit & Pull Request Guidelines

History uses concise prefixes such as `feat:`, `fix:`, `docs:`, and scoped forms like `tool(catch-reader):`. Keep commits single-purpose. Pull requests should state the objective, affected configuration, smoke commands and results. Include metric tables or visual samples for reconstruction changes and link the governing issue/spec.

## Agent-Specific Safety

When `.codegraph/` exists, run `codegraph explore "<question>"` before text search. Preserve unrelated dirty-worktree changes, never overwrite generated runs, and do not commit credentials, model weights, datasets, or experiment outputs.
