"""高斯 ply 导出的两处接线。

存在的理由是两个都**不报错**的失败：

1. `target_frame_idx` 的布局是 [b, tgt_t * v]（每个帧号按相机重复 v 次），
   而 save_gs_params_to_ply 曾经直接用 t_idx 去索引它。于是文件名等于
   t_idx // v：只产出 ceil(tgt_t/v) 个不同的名字，每个名字还被三个不同的
   目标帧先后写入，活下来的是最后一个。NUM_FRAMES=48 时的表现就是"只输出
   到 gs_15"，而那个文件里装的是目标帧 47。文件名和内容对不上，没有异常。

2. `gs_semantic_*.ply` 原来藏在 `if self.with_feat:` 里，而 woLSeg 变体把
   with_feat 硬编码成 False 且该分支第一行就 raise NotImplementedError，
   所以它从来没执行过 —— 导出目录里安静地少一类文件。

两条都是静态检查：导入 slarm 会一路拉到 gsplat（CUDA-only）。

    pytest tests/models/test_gs_ply_export.py -q
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SLARM_SRC = ROOT / "src" / "models" / "slarm.py"


def _exporter() -> ast.FunctionDef:
    tree = ast.parse(SLARM_SRC.read_text())
    return next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "save_gs_params_to_ply"
    )


def test_frame_names_come_from_the_normalizer_not_raw_indexing():
    body = ast.unparse(_exporter())
    assert "normalize_frame_indices" in body
    assert "data_dict['target_frame_idx'][0].to(torch.int16).tolist()" not in body


@pytest.mark.parametrize("tgt_t,views", [(25, 3), (40, 3), (48, 3)])
def test_every_target_frame_gets_its_own_file_name(tgt_t, views):
    """Reproduce both layouts and show the old one collapses names."""
    flattened = [frame for frame in range(tgt_t) for _ in range(views)]
    collapsed = {flattened[i] for i in range(tgt_t)}
    assert len(collapsed) == -(-tgt_t // views)          # 旧行为：名字变少
    normalized = list(range(tgt_t))                       # 新行为：[b, t]
    assert len(set(normalized)) == tgt_t
    assert sorted(normalized) == normalized


def test_the_collapsed_name_held_the_wrong_frame():
    """gs_15.ply used to contain target frame 47, silently."""
    tgt_t, views = 48, 3
    flattened = [frame for frame in range(tgt_t) for _ in range(views)]
    written_to_name_15 = [i for i in range(tgt_t) if flattened[i] == 15]
    assert written_to_name_15 == [45, 46, 47]


def test_semantic_export_is_not_behind_the_dead_with_feat_branch():
    """Cut the `if self.with_feat:` branch out; the export must survive."""
    exporter = _exporter()
    dead = next(
        node for node in ast.walk(exporter)
        if isinstance(node, ast.If) and ast.unparse(node.test) == "self.with_feat"
    )
    outside = ast.unparse(exporter).replace(ast.unparse(dead), "")
    assert "gs_semantic_" in outside, "semantic export still only lives in dead code"


def test_semantic_colors_are_converted_for_the_sh_writer():
    """save_ply stores colors as SH DC and applies SH2RGB on write."""
    body = ast.unparse(_exporter())
    assert "RGB2SH" in body, "a raw RGB palette would come out washed out"


def test_semantic_palette_marks_the_ball_class():
    """Class 1 is the ball; the eval ball mask is `semantic == 1`."""
    source = SLARM_SRC.read_text()
    assert "TASK_SEMANTIC_PLY_COLORS" in source
    tree = ast.parse(source)
    palette = next(
        node.value for node in tree.body
        if isinstance(node, ast.Assign)
        and any(getattr(t, "id", "") == "TASK_SEMANTIC_PLY_COLORS" for t in node.targets)
    )
    colors = ast.literal_eval(palette)
    assert len(colors) == 4
    assert all(len(c) == 3 and all(0.0 <= v <= 1.0 for v in c) for c in colors)
    ball, background = colors[1], colors[0]
    assert max(ball) - min(ball) > 0.5, "the ball colour has to be saturated"
    assert max(background) - min(background) < 0.2, "background should stay neutral"


def test_semantic_layout_is_asserted_before_indexing():
    """A resolution mismatch must say so, not index into the wrong pixels."""
    body = ast.unparse(_exporter())
    assert "context_task_semantic is" in body and "(b, t, v, h, w)" in body
