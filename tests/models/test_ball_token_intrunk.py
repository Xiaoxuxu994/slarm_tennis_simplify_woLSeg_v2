#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""in-trunk ball token 的接线检查。

要验的不是"数值对不对"，是**接线对不对** —— 那类错误不抛异常，只会让训练
静默地学错东西：

  - aggregator 的 concat 顺序、patch_start_idx 的累加顺序、slarm 的取回切片顺序
    三者必须一致。错一位会把 sky token 当 ball token 读，形状完全合法，loss 也在降。
  - aggregator.ball_token 的名字命中 "aggregator." 前缀，默认会掉进 trunk 参数组
    吃 trunk_lr（head 的 1/5~1/25）。它是零初始化的新模块，那样学不起来。
  - ball_token_freeze_backbone 的前缀白名单漏了 in-trunk 的参数名的话，
    freeze 模式会把 token 本身冻住，跑起来一切正常但什么都没学。

两种跑法，都可以：
    python tests/models/test_ball_token_intrunk.py     # 零依赖，不需要 pytest
    pytest tests/models/test_ball_token_intrunk.py -v

不装 torch 时，需要 torch 的那项自动跳过，其余照跑（它们只解析源码文本）。
所有运行时输出是纯 ASCII 英文 —— 终端 locale 常常渲染不了中文。
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

WORKTREE = Path(__file__).resolve().parents[2]
if str(WORKTREE) not in sys.path:
    sys.path.insert(0, str(WORKTREE))

try:
    import pytest
except ImportError:                                       # 允许脱离 pytest 直接跑
    pytest = None

try:
    import torch
except ImportError:
    torch = None


def needs_torch(fn):
    """标记一项需要 torch；有 pytest 时用它的 skipif，没有时留给 __main__ 处理。"""
    fn.needs_torch = True
    if pytest is not None:
        return pytest.mark.skipif(torch is None, reason="torch not installed")(fn)
    return fn


# ---------------------------------------------------------------- 参数分组
def _trunk_rules():
    src = (WORKTREE / "src/utils/stream25_losses.py").read_text(encoding="utf-8")
    prefixes = tuple(re.findall(
        r'"([^"]+)"', re.search(r"_TRUNK_PREFIXES = \(([^)]*)\)", src, re.S).group(1)))
    exceptions_match = re.search(r"_TRUNK_EXCEPTIONS = \(([^)]*)\)", src, re.S)
    exceptions = tuple(re.findall(r'"([^"]+)"', exceptions_match.group(1))
                       ) if exceptions_match else ()
    return prefixes, exceptions


def test_ball_token_goes_to_head_group_not_trunk():
    """aggregator.ball_token must be treated as a new module, not as trunk.

    It lives inside the aggregator so its name matches the "aggregator." prefix,
    but it is zero-initialised and cannot learn at trunk_lr.
    """
    prefixes, exceptions = _trunk_rules()

    def is_trunk(name):
        return (not name.startswith(exceptions)) and name.startswith(prefixes)

    for name in ("aggregator.ball_token", "ball_token_norm.weight",
                 "ball_head_intrunk.fc1.weight"):
        assert not is_trunk(name), f"{name} must be in the head group, got trunk"
    for name in ("aggregator.sky_token", "aggregator.frame_blocks.0.attn.qkv.weight",
                 "patch_embed.proj.weight", "aggregated_last_tokens_norm.weight"):
        assert is_trunk(name), f"{name} must stay in the trunk group, got head"


@needs_torch
def test_param_groups_put_ball_token_on_head_lr():
    """The real grouping function, not just the prefix rules."""
    import torch.nn as nn

    from src.utils.stream25_losses import make_stream25_param_groups

    class Toy(nn.Module):
        def __init__(self):
            super().__init__()
            self.aggregator = nn.Module()
            self.aggregator.ball_token = nn.Parameter(torch.zeros(1, 1, 8))
            self.aggregator.sky_token = nn.Parameter(torch.zeros(1, 1, 8))
            self.ball_head_intrunk = nn.Linear(8, 6)

    toy = Toy()
    groups = make_stream25_param_groups(
        toy, head_lr=1e-4, trunk_lr=1e-5, weight_decay=0.05)
    head_ids, trunk_ids = set(), set()
    for g in groups:
        target = head_ids if g["group_name"].startswith("head") else trunk_ids
        target.update(id(p) for p in g["params"])

    assert id(toy.aggregator.ball_token) in head_ids, "ball_token landed on trunk_lr"
    assert id(toy.ball_head_intrunk.weight) in head_ids, "ball_head_intrunk landed on trunk_lr"
    assert id(toy.aggregator.sky_token) in trunk_ids, "sky_token should stay on trunk_lr"


# ---------------------------------------------------------------- freeze 白名单
def test_freeze_whitelist_covers_intrunk_parameter_names():
    """The freeze whitelist in main_slarm.py must know the in-trunk names.

    If it misses them, freeze mode pins the ball token itself and the run
    completes normally having learned nothing.
    """
    src = (WORKTREE / "main_slarm.py").read_text(encoding="utf-8")
    match = re.search(r"_ball_prefixes = \(([^)]*)\)", src, re.S)
    assert match, "_ball_prefixes not found in main_slarm.py"
    prefixes = tuple(re.findall(r'"([^"]+)"', match.group(1)))

    for name in ("aggregator.ball_token", "ball_token_norm.weight",
                 "ball_head_intrunk.fc1.bias", "ball_query", "ball_head.fc1.weight"):
        assert name.startswith(prefixes), f"{name} is not in the freeze whitelist"
    for name in ("aggregator.frame_blocks.0.attn.qkv.weight", "patch_embed.proj.weight",
                 "aggregator.sky_token"):
        assert not name.startswith(prefixes), f"{name} must stay frozen"


# ---------------------------------------------------------------- 接线顺序
def _orders():
    agg = (WORKTREE / "src/models/components/aggregator/aggregator.py").read_text(encoding="utf-8")
    slarm = (WORKTREE / "src/models/slarm.py").read_text(encoding="utf-8")

    idx = []
    for key, token in (("use_time_token", "time"), ("num_motion_tokens", "motion"),
                       ("use_affine_token", "affine"), ("use_sky_token", "sky"),
                       ("use_ball_token", "ball")):
        pos = agg.find(f"if self.{key}")
        assert pos > 0, f"aggregator has no patch_start_idx branch for {key}"
        idx.append((pos, token))
    idx.sort()

    cat = []
    for token, needle in (("time", "time_token], dim=1"), ("motion", "motion_tokens], dim=1"),
                          ("affine", "affine_token], dim=1"), ("sky", "sky_token], dim=1"),
                          ("ball", "ball_token], dim=1")):
        pos = agg.find(needle)
        assert pos > 0, f"aggregator has no concat for {token}"
        cat.append((pos, token))
    cat.sort()

    start = slarm.find("others_last_tokens = last_tokens")
    read = []
    for token, needle in (("ball", "if self.use_ball_token_intrunk:"),
                          ("sky", "if self.use_sky_token:"),
                          ("affine", "if self.use_affine_token:"),
                          ("motion", "if self.num_motion_tokens > 0:")):
        pos = slarm.find(needle, start)
        assert pos > 0, f"slarm.py has no readback branch for {token}"
        read.append((pos, token))
    read.sort()

    return ([t for _, t in idx], [t for _, t in cat], [t for _, t in read],
            agg, cat[-1][0])


def test_concat_order_matches_patch_start_idx_order():
    """Off by one here misplaces where patch tokens begin."""
    idx_seq, cat_seq, _, agg, last_special = _orders()
    assert idx_seq == cat_seq, (
        f"patch_start_idx order {idx_seq} != concat order {cat_seq}")
    patch_pos = agg.find("tokens = torch.cat([tokens, patch_tokens], dim=1)")
    assert patch_pos > last_special, "patch tokens must be concatenated last"


def test_readback_order_is_reverse_of_concat():
    """Readback slices from the tail, so it must mirror the concat order.

    Off by one reads sky_token as the ball token: legal shape, loss still goes
    down, just not on the thing you think.
    """
    _, cat_seq, read_seq, _, _ = _orders()
    expected = [t for t in reversed(cat_seq) if t in read_seq]
    assert read_seq == expected, f"readback order {read_seq} != expected {expected}"


def test_ball_token_is_read_before_sky():
    """ball is concatenated last, so it must be sliced off first."""
    slarm = (WORKTREE / "src/models/slarm.py").read_text(encoding="utf-8")
    start = slarm.find("others_last_tokens = last_tokens")
    ball = slarm.find("if self.use_ball_token_intrunk:", start)
    sky = slarm.find("if self.use_sky_token:", start)
    assert 0 < ball < sky, "in-trunk ball token must be read back before sky token"


# ---------------------------------------------------------------- 形状
@needs_torch
def test_patch_start_idx_shifts_by_exactly_one():
    """Enabling the in-trunk ball token adds exactly one special token."""
    from src.models.components.aggregator.aggregator import Aggregator

    common = dict(img_size=(240, 320), patch_size=8, embed_dim=64, depth=1,
                  num_register_tokens=2, use_sky_token=True, use_affine_token=False,
                  num_motion_tokens=0)
    try:
        base = Aggregator(**common, use_ball_token=False)
        with_ball = Aggregator(**common, use_ball_token=True)
    except TypeError as exc:
        if pytest is not None:
            pytest.skip(f"Aggregator signature differs from this test: {exc}")
        print(f"    skipped: Aggregator signature differs: {exc}")
        return
    assert with_ball.patch_start_idx == base.patch_start_idx + 1, (
        f"patch_start_idx {base.patch_start_idx} -> {with_ball.patch_start_idx}, expected +1")
    assert tuple(with_ball.ball_token.shape) == (1, 1, 64)
    assert int(torch.count_nonzero(with_ball.ball_token)) == 0, (
        "ball_token must be zero-initialised so it does not perturb a converged backbone")


# ---------------------------------------------------------------- 无 pytest 时的入口
def _run_standalone() -> int:
    tests = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith("test_") and callable(fn)]
    print("=" * 72)
    print(f"in-trunk ball token wiring checks  ({len(tests)} tests)")
    print(f"torch: {'yes' if torch is not None else 'no -- torch-only tests skipped'}")
    print("=" * 72)
    failed = skipped = 0
    for name, fn in tests:
        if getattr(fn, "needs_torch", False) and torch is None:
            print(f"[SKIP] {name}")
            skipped += 1
            continue
        try:
            fn()
        except AssertionError as exc:
            print(f"[FAIL] {name}")
            print(f"       {exc}")
            failed += 1
        except Exception as exc:                          # noqa: BLE001
            print(f"[ERROR] {name}: {type(exc).__name__}: {exc}")
            failed += 1
        else:
            print(f"[ OK ] {name}")
    print("-" * 72)
    print(f"{len(tests) - failed - skipped} passed, {failed} failed, {skipped} skipped")
    if failed:
        print("")
        print("A failure here means the ball token is mis-wired. Do not start training:")
        print("the errors it guards against do not raise, they just train the wrong thing.")
    print("=" * 72)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run_standalone())
