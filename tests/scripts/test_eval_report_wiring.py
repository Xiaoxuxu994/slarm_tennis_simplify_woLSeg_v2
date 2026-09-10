"""报告写出路径的接线。

存在的理由：`_finalize_and_write` 是**独立的顶层函数**（为了支持多 shard 合并），
不是 `run_evaluation` 的内嵌闭包。commit a325081 往它体内加了对
`ball_surface_offset` 的引用却没加参数，结果是**每一次 eval 都 NameError** ——
连不开补偿的也一样，因为写进 result 的那两行无条件执行。当时的 AST 检查只覆盖了
计算链（run_evaluation -> evaluate_scene -> ... -> compute_rendered_frame24_position_errors），
漏了报告链，而已有测试没有一条真正调用过这个函数。

    pytest tests/scripts/test_eval_report_wiring.py -q
"""
from __future__ import annotations

import ast
import builtins
import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils.stream25_metrics import CAMERA_ORDER  # noqa: E402


def _load_finalizer():
    """导入 eval 脚本。它会一路拉到 slarm.py 的 gsplat（CUDA-only），
    所以在没有 CUDA 的机器上跳过 —— 静态那条测试不需要 import，永远会跑。"""
    pytest.importorskip("gsplat", reason="eval_stream25_base imports the CUDA rasterizer")
    from scripts.eval_stream25_base import _finalize_and_write
    return _finalize_and_write

EVAL_SRC = Path(__file__).resolve().parents[2] / "scripts" / "eval_stream25_base.py"


def _scene_results():
    scopes = {}
    for index, scope in enumerate(("aggregate",) + CAMERA_ORDER):
        scopes[scope] = {
            "metrics": {"rgb_psnr": {"anchor": 28.0 - index}},
            "valid_counts": {"rgb_psnr": {"anchor": 6}},
        }
    return [{"scopes": scopes, "scene_index": 0}]


def _write(tmp_path, **kwargs):
    ckpt = tmp_path / "ckpt.pth"
    ckpt.write_bytes(b"not a real checkpoint, only hashed")
    out = tmp_path / f"result_{kwargs.get('ball_radius_compensation', 0)}.json"
    return _load_finalizer()(
        _scene_results(),
        split="validation",
        checkpoint_path=str(ckpt),
        config_path="configs/does_not_need_to_exist.yml",
        manifest=None,
        evaluation_seed=0,
        reference=False,
        output_json=str(out),
        output_markdown=str(out.with_suffix(".md")),
        **kwargs,
    ), out


def test_writing_the_report_does_not_raise(tmp_path):
    """最基本的一条：这个函数被调用过。NameError 会在这里炸。"""
    result, out = _write(tmp_path)
    assert out.exists()
    assert json.loads(out.read_text())["overall"] in ("PASS", "FAIL")


def test_compensation_off_is_recorded_as_off(tmp_path):
    result, _ = _write(tmp_path)
    assert result["ball_surface_offset_m"] == 0.0
    assert result["frame24_position_method"].endswith("frame15")


def test_compensation_on_is_stamped_into_the_method_name(tmp_path):
    """开了补偿之后，口径必须跟着数字一起写进结果，否则两次运行无法比对。"""
    result, out = _write(
        tmp_path,
        ball_surface_offset=0.021,
        ball_radius=0.0325,
        ball_radius_compensation=0.646,
    )
    assert result["ball_surface_offset_m"] == pytest.approx(0.021)
    assert result["ball_radius_m"] == pytest.approx(0.0325)
    assert result["frame24_position_method"].endswith("_ball_center_compensated")
    assert "compensation" in out.with_suffix(".md").read_text()


def test_no_top_level_function_references_an_unresolvable_name():
    """静态兜底：捕捉"编辑落进了错误的函数作用域"这一整类 bug。

    a325081 就是这个形状 —— 改动看起来在 run_evaluation 里，实际落在
    _finalize_and_write 里，闭包不成立。逐个函数检查自由变量是否可解析，
    比逐个补端到端测试便宜得多。
    """
    tree = ast.parse(EVAL_SRC.read_text(encoding="utf-8"))
    module_names = set(dir(builtins)) | {"__name__", "__file__"}
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            module_names |= {a.asname or a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.Assign):
            module_names |= {t.id for t in node.targets if isinstance(t, ast.Name)}
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            module_names.add(node.name)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            module_names.add(node.target.id)

    problems = []
    for fn in [n for n in tree.body if isinstance(n, ast.FunctionDef)]:
        bound = {a.arg for a in fn.args.args} | {a.arg for a in fn.args.kwonlyargs}
        bound |= {a.arg for a in fn.args.posonlyargs}
        for extra in (fn.args.vararg, fn.args.kwarg):
            if extra is not None:
                bound.add(extra.arg)
        for sub in ast.walk(fn):
            if isinstance(sub, ast.Name) and isinstance(sub.ctx, (ast.Store, ast.Del)):
                bound.add(sub.id)
            elif isinstance(sub, (ast.Import, ast.ImportFrom)):
                bound |= {a.asname or a.name.split(".")[0] for a in sub.names}
            elif isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                bound.add(sub.name)
            elif isinstance(sub, ast.ExceptHandler) and sub.name:
                bound.add(sub.name)
            elif isinstance(sub, (ast.comprehension,)):
                for t in ast.walk(sub.target):
                    if isinstance(t, ast.Name):
                        bound.add(t.id)
            elif isinstance(sub, ast.Lambda):
                bound |= {a.arg for a in sub.args.args}
        for sub in ast.walk(fn):
            if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Load):
                if sub.id not in bound and sub.id not in module_names:
                    problems.append(f"{fn.name}() line {sub.lineno}: {sub.id}")
    assert not problems, "无法解析的自由变量：\n  " + "\n  ".join(sorted(set(problems)))
