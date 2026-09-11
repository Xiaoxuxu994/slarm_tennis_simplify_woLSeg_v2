"""滑动观测窗口（context offset）。

存在的理由：这是一个**纯数据侧**开关。`datasets.py` 的 `get_frame` 算的是

    dt = time_in_seconds[frame_idx] - time_in_seconds[source_frame_idx]

而 `source_frame_idx = context_frames[0]`，所以模型收到的时间是相对**窗口自己的
第一帧**的。窗口整体后移、source 跟着移，模型看到的时间值逐字节不变（0, 0.1,
..., 0.5 秒），连续的 time_embedder 分辨不出两个窗口。变的只有图像里球更近。

这份测试盯住三件会悄悄毁掉实验的事：

1. offset 0 必须**逐字节**等于冻结契约。任何漂移都会让所有历史数字失效。
2. 不变量 `targets[15] == context[-1]`。eval 里每一处终端读出都写死了目标下标
   15（`pred_depth[15]`、`target_ray_origins[0, 15]`），这条不成立就读错帧，
   而且不会报错，只会给出一个看起来合理的错数。
3. 训练路径必须拒绝非零 offset。训练的 target scheduler 仍然说冻结帧号，
   滑窗后会把移位的 context 和没移位的 target 配在一起。

    pytest tests/dataset/test_context_offset.py -q
"""
from __future__ import annotations

import ast
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

ROOT = Path(__file__).resolve().parents[2]

# stream25.py 顶上 import cv2/torch，没装依赖的机器跑不了 import 那几条；
# 纯算术的部分从 AST 里取出来单独执行，任何机器都能跑。
_BASE_CONTEXT = (0, 3, 6, 9, 12, 15)
_BASE_TARGETS = tuple(range(25))


def _standalone_shifted_contract():
    source = (ROOT / "src" / "dataset" / "stream25.py").read_text()
    tree = ast.parse(source)
    fn = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "shifted_contract"
    )
    namespace = {
        "Tuple": tuple,
        "STREAM25_CONTEXT_FRAMES": _BASE_CONTEXT,
        "STREAM25_ALL_TARGET_FRAMES": _BASE_TARGETS,
    }
    exec(compile(ast.Module([fn], []), "<shifted_contract>", "exec"), namespace)
    return namespace["shifted_contract"]


shifted_contract = _standalone_shifted_contract()


def test_zero_offset_is_the_frozen_contract_byte_for_byte():
    context, targets = shifted_contract(0)
    assert context == _BASE_CONTEXT
    assert targets == _BASE_TARGETS


@pytest.mark.parametrize("offset", [0, 1, 3, 6, 9])
def test_target_index_fifteen_is_always_the_terminal_observation(offset):
    """eval 的终端读出写死了目标下标 15。这条不成立就静默读错帧。"""
    context, targets = shifted_contract(offset)
    assert targets[15] == context[-1] == 15 + offset


@pytest.mark.parametrize("offset", [0, 1, 3, 6, 9])
def test_window_keeps_its_shape_and_stays_inside_stored_frames(offset):
    context, targets = shifted_contract(offset)
    assert len(context) == len(_BASE_CONTEXT)
    assert [b - a for a, b in zip(context, context[1:])] == [3] * 5
    assert targets[0] == offset and targets[-1] == _BASE_TARGETS[-1]
    assert len(targets) == len(_BASE_TARGETS) - offset
    assert list(targets) == sorted(set(targets))


@pytest.mark.parametrize("bad", [-1, 10, 25, 100])
def test_offset_past_the_stored_frames_is_refused(bad):
    with pytest.raises(ValueError):
        shifted_contract(bad)


@pytest.mark.parametrize("bad", [1.0, "9", None, True])
def test_non_integer_offset_is_refused(bad):
    """True is an int in Python; an offset of True would silently mean +1."""
    with pytest.raises(TypeError):
        shifted_contract(bad)


def test_times_the_trunk_sees_are_identical_at_every_offset():
    """The whole no-retraining claim in one assertion.

    get_frame subtracts time_in_seconds[context_frames[0]], so reproduce that
    arithmetic on a 30 fps clock and check the six values never move.
    """
    fps = 30.0
    baseline = None
    for offset in (0, 1, 3, 6, 9):
        context, _ = shifted_contract(offset)
        source = context[0]
        times = tuple(round((f - source) / fps, 9) for f in context)
        if baseline is None:
            baseline = times
        assert times == baseline
    assert baseline == (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)


def test_training_path_refuses_a_slid_window():
    """The training target scheduler still speaks frozen frame numbers."""
    source = (ROOT / "src" / "dataset" / "datasets.py").read_text()
    tree = ast.parse(source)
    cls = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef) and node.name == "Stream25Dataset"
    )
    init = next(
        node for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )
    assert "context_offset" in {arg.arg for arg in init.args.kwonlyargs + init.args.args}
    raises = [
        node for node in ast.walk(init)
        if isinstance(node, ast.Raise)
        and "context_offset" in ast.unparse(node)
    ]
    assert raises, "a non-zero offset on the training path must raise"


def test_getitem_anchors_source_frame_to_the_window_not_the_clip():
    """source_frame_idx must follow the window, or the times would shift."""
    source = (ROOT / "src" / "dataset" / "datasets.py").read_text()
    tree = ast.parse(source)
    cls = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef) and node.name == "Stream25Dataset"
    )
    getitem = next(
        node for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "__getitem__"
    )
    body = ast.unparse(getitem)
    assert "source_frame_idx = context_frames[0]" in body
    assert "STREAM25_CONTEXT_FRAMES[0]" not in body


# ---------------------------------------------------------------------------
# eval 侧的接线。纯静态，不 import（eval 会一路拉到 gsplat）。
#
# 存在的理由和 test_eval_report_wiring.py 一样：commit a325081 在一个顶层函数体
# 里引用了没有对应参数的名字，于是每一次 eval 都 NameError。offset 要穿过五层
# 调用，任何一层漏掉参数，指标都会**静默**地按 offset 0 去索引 GT —— 不报错，
# 只是拿终端状态去和错误的时刻比，得到一个看起来正常的数。
# ---------------------------------------------------------------------------

_EVAL_SRC = ROOT / "scripts" / "eval_stream25_base.py"

_MUST_ACCEPT_OFFSET = (
    "compute_stream25_scene_metrics",
    "evaluate_scene",
    "run_evaluation",
    "compute_balltoken_frame24_metrics",
    "_balltoken_fit_metrics",
    "compute_rendered_history_fit_metrics",
    "_finalize_and_write",
)


def _eval_tree():
    return ast.parse(_EVAL_SRC.read_text())


@pytest.mark.parametrize("name", list(_MUST_ACCEPT_OFFSET))
def test_every_link_in_the_chain_takes_the_offset(name):
    fn = next(
        node for node in ast.walk(_eval_tree())
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    params = {a.arg for a in fn.args.args + fn.args.kwonlyargs}
    assert "context_offset" in params, f"{name} would silently index GT at offset 0"


def test_eval_source_compiles_not_just_parses():
    """compile() rejects duplicate argument names; ast.parse does not."""
    compile(_EVAL_SRC.read_text(), str(_EVAL_SRC), "exec")


def test_no_bare_frame_fifteen_or_twentyfour_gt_lookups_remain():
    """GT is indexed by ABSOLUTE frame, so a literal 15/24 ignores the offset."""
    source = _EVAL_SRC.read_text()
    for bad in ('ball_position_rig"][0, 24]',
                'gt_pos15[0, 15]',
                'gt_v15[0, 15]',
                'gt_v[0, 15]',
                'gt_positions[15]',
                'target_time"][0, 24, 0]'):
        assert bad not in source, f"offset-blind GT lookup still present: {bad}"


def test_catch_metric_is_registered_everywhere_it_is_aggregated():
    """A metric absent from the name tuple is computed and then dropped."""
    assert '"catch_position",' in _EVAL_SRC.read_text()
    report = (ROOT / "src" / "utils" / "stream25_report.py").read_text()
    assert '("catch_position", "median")' in report
    compare = (ROOT / "tools" / "compare_evaluations.py").read_text()
    assert '"catch_position"' in compare
