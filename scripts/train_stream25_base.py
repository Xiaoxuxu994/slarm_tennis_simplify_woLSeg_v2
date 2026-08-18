"""Full-only launcher for stereo and tri-view Stream25 training."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import sys
from typing import Any, Sequence

import yaml


WORKTREE = Path(__file__).resolve().parent.parent
CONFIGS = {
    "stereo": WORKTREE / "configs/slarm_stream25_24cm_nopitch_window6.yaml",
    "triview": WORKTREE / "configs/slarm_stream25_24cm_triview_window6.yaml",
}
MODE_EXPECTATIONS = {
    "stereo": {
        "dataset": ["ball_catch_24cm_stereo40_stream25_nopitch"],
        "num_max_cameras": 2,
        "num_iterations": 20_000,
    },
    "triview": {
        "dataset": ["ball_catch_24cm_triview"],
        "num_max_cameras": 3,
        "num_iterations": 40_000,
    },
}


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_config(mode: str) -> str:
    return str(CONFIGS[mode])


def _load_config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def validate_full_config(path: str | Path, mode: str) -> dict[str, Any]:
    return _load_config(Path(path))


def _consume_config(arguments: Sequence[str]) -> tuple[Path | None, list[str]]:
    requested = None
    forwarded: list[str] = []
    iterator = iter(arguments)
    for argument in iterator:
        if argument == "--config":
            requested = Path(next(iterator))
        elif argument.startswith("--config="):
            requested = Path(argument.split("=", 1)[1])
        else:
            forwarded.append(argument)
    return requested, forwarded


def build_launch_command(mode: str, extra_args: Sequence[str]) -> list[str]:
    requested, forwarded = _consume_config(extra_args)
    config_path = (requested or Path(resolve_config(mode))).resolve()
    validate_full_config(config_path, mode)
    return [
        sys.executable,
        str(WORKTREE / "main_slarm.py"),
        f"--config={config_path}",
        *forwarded,
    ]


def main(mode: str, extra_args: Sequence[str]) -> list[str] | None:
    command = build_launch_command(mode, extra_args)
    config_path = Path(command[2].split("=", 1)[1])
    config = _load_config(config_path)
    checkpoint = Path(str(config["load_from"]))
    if not checkpoint.is_absolute():
        checkpoint = WORKTREE / checkpoint
    print(f"[Stream25] mode={mode} config={config_path}")
    print(f"[Stream25] config_sha256={_sha256(config_path)}")
    if checkpoint.is_file():
        print(f"[Stream25] initial_checkpoint_sha256={_sha256(checkpoint)}")
    print(f"[Stream25] command={' '.join(command)}")
    if os.environ.get("STREAM25_LAUNCH_DRY_RUN") == "1":
        return command
    os.environ.setdefault("SLARM_OFFLOAD_TARGET_FEAT", "1")
    os.environ.setdefault("SLARM_SINGLE_PROCESS", "1")
    os.execvp(sys.executable, command)
    return None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("stereo", "triview"))
    parsed, unknown = parser.parse_known_args()
    main(parsed.mode, unknown)
