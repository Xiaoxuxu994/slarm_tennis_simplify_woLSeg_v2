"""Check launcher arguments without starting Python training or CUDA workers."""
import os
from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[2]
JOINT_CONFIG = "configs/exp0910_004_balltoken_temporal_joint.yml"
PREFIX_CONFIG = "configs/exp0910_005_balltoken_prefix_only.yml"


@pytest.fixture
def launch(tmp_path):
    stub = (
        '#!/bin/bash\n'
        'printf "EXEC=%s\\nCWD=%s\\nGPUS=%s\\n" '
        '"${0##*/}" "$PWD" "$CUDA_VISIBLE_DEVICES"\n'
        'printf "ARG=%s\\n" "$@"\n'
    )
    for name in ("python", "torchrun"):
        executable = tmp_path / name
        executable.write_text(stub)
        executable.chmod(0o755)

    def run(overrides=None, args=()):
        env = os.environ.copy()
        for name in ("GPUS", "CONFIG", "RESUME"):
            env.pop(name, None)
        env.update(overrides or {})
        env["PATH"] = str(tmp_path) + os.pathsep + env["PATH"]
        result = subprocess.run(
            ["bash", str(ROOT / "run_sh/train.sh"), *args],
            cwd=tmp_path, env=env, capture_output=True, text=True, check=True,
        )
        return result.stdout.splitlines()

    return run


def test_default_is_four_gpu_joint_finetune_without_resume(launch):
    lines = launch()
    assert "EXEC=torchrun" in lines
    assert f"CWD={ROOT}" in lines
    assert "GPUS=0,1,2,3" in lines
    assert "ARG=--nproc_per_node=4" in lines
    assert f"ARG=--config={JOINT_CONFIG}" in lines
    assert "ARG=--enable_tensorboard" in lines
    assert "ARG=--auto_resume" not in lines
    assert "ARG=--resume_from" not in lines


def test_environment_can_select_other_four_gpus_and_prefix_config(launch):
    lines = launch({"GPUS": "4,5,6,7", "CONFIG": PREFIX_CONFIG})
    assert "GPUS=4,5,6,7" in lines
    assert "ARG=--nproc_per_node=4" in lines
    assert f"ARG=--config={PREFIX_CONFIG}" in lines


def test_single_gpu_and_checkpoint_argument_remain_supported(launch):
    lines = launch({"GPUS": "2"}, ("--load_from", "/checkpoint dir/ckpt_007999.pth"))
    assert "EXEC=python" in lines
    assert "GPUS=2" in lines
    assert "ARG=--load_from" in lines
    assert "ARG=/checkpoint dir/ckpt_007999.pth" in lines
    assert not any(line.startswith("ARG=--nproc_per_node") for line in lines)


def test_resume_remains_explicit_opt_in(launch):
    assert "ARG=--auto_resume" in launch({"RESUME": "1"})
