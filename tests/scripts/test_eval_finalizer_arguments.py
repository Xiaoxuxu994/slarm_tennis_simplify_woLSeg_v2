"""Exercise report argument wiring without importing CUDA model dependencies."""

import ast
import inspect
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest


SOURCE = Path(__file__).resolve().parents[2] / "scripts/eval_stream25_base.py"


@pytest.mark.parametrize("fit_frames", [None, [6, 9, 12, 15]])
def test_finalizer_accepts_fit_frames_and_writes_json(tmp_path, monkeypatch, fit_frames):
    tree = ast.parse(SOURCE.read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
              and n.name == "_finalize_and_write")
    runtime = ModuleType("tools.stream25_runtime")
    runtime.sha256_file = lambda path: "test-hash"
    monkeypatch.setitem(sys.modules, "tools.stream25_runtime", runtime)
    gates = {"scope_reports": {"aggregate": {"metrics": {}, "valid_counts": {}}},
             "gates": {}, "missing_gates": [], "all_gates_pass": True, "worst_ratio": 0}
    env = {"summarize_stream25_scene_results": lambda scenes: gates,
           "TIME_BUCKETS": {}, "_json_safe": lambda value: value, "json": json}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(SOURCE), "exec"), env)
    finalizer = env[fn.name]
    # Check every production call's keyword names, including future additions.
    signature = inspect.signature(finalizer)
    for call in ast.walk(tree):
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == fn.name:
            signature.bind(*([None] * len(call.args)), **{kw.arg: None for kw in call.keywords})
    output = tmp_path / "report.json"
    result = finalizer(
        [], split="validation", checkpoint_path="test.pth", config_path="test.yml",
        manifest="test.txt", evaluation_seed=0, reference=False,
        output_json=str(output), output_markdown=None, balltoken_fit_frames=fit_frames,
    )
    assert result["balltoken_fit_frames"] == fit_frames
    assert json.loads(output.read_text())["balltoken_fit_frames"] == fit_frames
