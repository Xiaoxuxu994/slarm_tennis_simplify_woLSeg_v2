# -*- coding: utf-8 -*-
"""in-trunk ball token 的接线检查。

这个模块要验的不是"数值对不对"，而是**接线对不对** —— 那类错误不会抛异常，
只会让训练静默地学错东西：

  - aggregator 的 concat 顺序、patch_start_idx 的累加顺序、slarm 的取回切片顺序
    三者必须一致。错一位就会把 sky token 当成 ball token 读，而形状完全合法。
  - aggregator.ball_token 的名字命中 "aggregator." 前缀，默认会掉进 trunk 参数组
    吃 trunk_lr（head 的 1/5~1/10）。它是零初始化的新模块，那样学不起来。
  - ball_token_freeze_backbone 的前缀白名单如果漏了 in-trunk 的参数名，
    freeze 模式会把 token 本身冻住，跑起来一切正常但什么都没学。

需要 torch，本地无 torch 的环境会自动跳过。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

WORKTREE = Path(__file__).resolve().parents[2]
if str(WORKTREE) not in sys.path:
    sys.path.insert(0, str(WORKTREE))

torch = pytest.importorskip("torch")


# ---------------------------------------------------------------- 参数分组
def test_ball_token_goes_to_head_group_not_trunk():
    """aggregator.ball_token 必须按"新模块"归到 head 组。

    它住在 aggregator 里，名字会命中 "aggregator." 前缀，但用 trunk_lr 学不起来。
    """
    from src.utils.stream25_losses import _is_trunk_param

    assert not _is_trunk_param("aggregator.ball_token")
    assert not _is_trunk_param("ball_token_norm.weight")
    assert not _is_trunk_param("ball_head_intrunk.fc1.weight")
    # 其余 aggregator 参数仍然是 trunk
    assert _is_trunk_param("aggregator.sky_token")
    assert _is_trunk_param("aggregator.frame_blocks.0.attn.qkv.weight")
    assert _is_trunk_param("patch_embed.proj.weight")
    assert _is_trunk_param("aggregated_last_tokens_norm.weight")


def test_param_groups_put_ball_token_on_head_lr():
    import torch.nn as nn
    from src.utils.stream25_losses import make_stream25_param_groups

    class Toy(nn.Module):
        def __init__(self):
            super().__init__()
            self.aggregator = nn.Module()
            self.aggregator.ball_token = nn.Parameter(torch.zeros(1, 1, 8))
            self.aggregator.sky_token = nn.Parameter(torch.zeros(1, 1, 8))
            self.ball_head_intrunk = nn.Linear(8, 6)

    groups = make_stream25_param_groups(
        Toy(), head_lr=1e-4, trunk_lr=1e-5, weight_decay=0.05
    )
    by_name = {g["group_name"]: g for g in groups}
    head_params = [p for n, g in by_name.items() if n.startswith("head")
                   for p in g["params"]]
    trunk_params = [p for n, g in by_name.items() if n.startswith("trunk")
                    for p in g["params"]]
    # ball_token 是 [1,1,8]，sky_token 同形状 —— 用 shape 区分不了，用身份比对
    toy = Toy()
    groups = make_stream25_param_groups(
        toy, head_lr=1e-4, trunk_lr=1e-5, weight_decay=0.05
    )
    head_ids, trunk_ids = set(), set()
    for g in groups:
        target = head_ids if g["group_name"].startswith("head") else trunk_ids
        target.update(id(p) for p in g["params"])
    assert id(toy.aggregator.ball_token) in head_ids
    assert id(toy.aggregator.sky_token) in trunk_ids
    assert id(toy.ball_head_intrunk.weight) in head_ids


# ---------------------------------------------------------------- freeze 白名单
def test_freeze_whitelist_covers_intrunk_parameter_names():
    """main_slarm.py 的 _ball_prefixes 必须认得 in-trunk 的三组参数名。

    漏掉的话 freeze 模式会把 ball_token 本身冻住，训练照常跑完但什么都没学到。
    这里直接读源码里的元组，避免测试和实现各写一份而悄悄漂移。
    """
    import re

    src = (WORKTREE / "main_slarm.py").read_text(encoding="utf-8")
    match = re.search(r"_ball_prefixes = \(([^)]*)\)", src, re.S)
    assert match, "main_slarm.py 里找不到 _ball_prefixes"
    prefixes = tuple(re.findall(r'"([^"]+)"', match.group(1)))

    for name in ("aggregator.ball_token",
                 "ball_token_norm.weight",
                 "ball_head_intrunk.fc1.bias"):
        assert name.startswith(prefixes), f"{name} 不在 freeze 白名单里"
    # backbone 仍然要被冻住
    for name in ("aggregator.frame_blocks.0.attn.qkv.weight", "patch_embed.proj.weight"):
        assert not name.startswith(prefixes), f"{name} 不该被当成 ball token 参数"


# ---------------------------------------------------------------- 接线顺序
def test_concat_order_matches_readback_order():
    """concat 顺序、patch_start_idx 累加顺序、取回切片顺序三者必须一致。

    这是最容易出错也最难发现的一处：错一位会把 sky token 当 ball token 读出来，
    形状完全合法，loss 也在降，只是降的不是你以为的那个东西。
    """
    agg = (WORKTREE / "src/models/components/aggregator/aggregator.py").read_text(encoding="utf-8")
    slarm = (WORKTREE / "src/models/slarm.py").read_text(encoding="utf-8")

    # aggregator: patch_start_idx 的累加顺序
    idx_order = []
    for key, token in (("use_time_token", "time"), ("num_motion_tokens", "motion"),
                       ("use_affine_token", "affine"), ("use_sky_token", "sky"),
                       ("use_ball_token", "ball")):
        pos = agg.find(f"if self.{key}")
        assert pos > 0, f"aggregator 里找不到 {key} 的 patch_start_idx 分支"
        idx_order.append((pos, token))
    idx_order.sort()
    idx_seq = [t for _, t in idx_order]

    # aggregator: concat 顺序（取 torch.cat([tokens, ...]) 那几处）
    cat_order = []
    for token, needle in (("time", "time_token], dim=1"), ("motion", "motion_tokens], dim=1"),
                          ("affine", "affine_token], dim=1"), ("sky", "sky_token], dim=1"),
                          ("ball", "ball_token], dim=1")):
        pos = agg.find(needle)
        assert pos > 0, f"aggregator 里找不到 {token} 的 concat"
        cat_order.append((pos, token))
    cat_order.sort()
    cat_seq = [t for _, t in cat_order]

    assert idx_seq == cat_seq, (
        f"patch_start_idx 累加顺序 {idx_seq} 与 concat 顺序 {cat_seq} 不一致 —— "
        "patch tokens 的起点会算错")

    # patch tokens 必须在所有 special token 之后
    assert agg.find("tokens = torch.cat([tokens, patch_tokens], dim=1)") > cat_order[-1][0]

    # slarm: 取回顺序必须是 concat 的倒序
    read_order = []
    for token, needle in (("ball", "if self.use_ball_token_intrunk:"),
                          ("sky", "if self.use_sky_token:"),
                          ("affine", "if self.use_affine_token:"),
                          ("motion", "if self.num_motion_tokens > 0:")):
        pos = slarm.find(needle, slarm.find("others_last_tokens = last_tokens"))
        assert pos > 0, f"slarm.py 里找不到 {token} 的取回分支"
        read_order.append((pos, token))
    read_order.sort()
    read_seq = [t for _, t in read_order]

    expected = [t for t in reversed(cat_seq) if t in read_seq]
    assert read_seq == expected, (
        f"取回顺序 {read_seq} 不是 concat 顺序 {cat_seq} 的倒序（应为 {expected}）—— "
        "会把相邻的 special token 读串")


def test_ball_token_is_read_before_sky():
    """ball 在 concat 的最尾，所以取回时必须最先切，否则读到的是 sky token。"""
    slarm = (WORKTREE / "src/models/slarm.py").read_text(encoding="utf-8")
    start = slarm.find("others_last_tokens = last_tokens")
    ball = slarm.find("if self.use_ball_token_intrunk:", start)
    sky = slarm.find("if self.use_sky_token:", start)
    assert 0 < ball < sky, "in-trunk ball token 必须在 sky token 之前取回"


# ---------------------------------------------------------------- 形状
@pytest.mark.parametrize("intrunk", [False, True])
def test_patch_start_idx_shifts_by_one(intrunk):
    """开了 in-trunk ball token，special token 数应当且仅当多 1。"""
    from src.models.components.aggregator.aggregator import Aggregator

    common = dict(img_size=(240, 320), patch_size=8, embed_dim=64, depth=1,
                  num_register_tokens=2, use_sky_token=True, use_affine_token=False,
                  num_motion_tokens=0)
    try:
        base = Aggregator(**common, use_ball_token=False)
        with_ball = Aggregator(**common, use_ball_token=True)
    except TypeError as exc:
        pytest.skip(f"Aggregator 的构造签名与本测试不符: {exc}")
    assert with_ball.patch_start_idx == base.patch_start_idx + 1
    assert with_ball.ball_token.shape == (1, 1, 64)
    # 零初始化：接到已收敛的 ckpt 上时对 attention 的扰动最小
    assert torch.count_nonzero(with_ball.ball_token) == 0
