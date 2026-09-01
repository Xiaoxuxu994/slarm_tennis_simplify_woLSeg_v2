#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""开跑前的模型初始化自检：ckpt 到底加载进去了多少，参数分到哪个 LR 组。

为什么需要这个
--------------
``misc.load_model`` 用 ``strict=False`` 加载，而它对 missing / unexpected keys
的处理是 ``if forbidden_missing: pass`` —— **权重没对上不会报错**。
``validate_stream25_checkpoint_contract`` 同样全是 ``pass``。

于是这两类事故都不会有任何提示：

  - 改了模型结构（比如加一个 special token），某些权重悄悄没加载进去，
    你以为在微调，其实半个网络是随机初始化的；
  - config 的 num_context_timesteps / timespan / dataset 与 ckpt 里存的契约不符，
    训出来的模型和你以为的不是一个东西。

还有一类更隐蔽的：新加的模块名字命中了 trunk 前缀，被分去吃 trunk_lr
（head 的 1/5 ~ 1/25），跑完发现"这个方法没用"，其实是它压根没学。

这个脚本只构建模型 + 加载 ckpt + 打印分组，不跑 forward，几十秒出结果。

用法
----
    python tools/check_model_init.py --config configs/xxx.yml
    python tools/check_model_init.py --config configs/xxx.yml --checkpoint path/to.pth

不给 --checkpoint 时用 config 里的 load_from。所有输出是纯 ASCII 英文。
"""
from __future__ import annotations

import argparse
import sys
from collections import OrderedDict
from pathlib import Path

WORKTREE = Path(__file__).resolve().parent.parent
if str(WORKTREE) not in sys.path:
    sys.path.insert(0, str(WORKTREE))


def _group_names(names, max_show=12):
    """把一堆参数名按模块前缀折叠，避免刷屏。"""
    buckets = OrderedDict()
    for n in names:
        head = n.split(".")[0]
        if head == "aggregator" and n.count(".") >= 1:
            head = ".".join(n.split(".")[:2])
        buckets.setdefault(head, []).append(n)
    out = []
    for head, items in buckets.items():
        if len(items) == 1:
            out.append(f"{items[0]}")
        else:
            out.append(f"{head}.* ({len(items)} tensors)")
    if len(out) > max_show:
        return out[:max_show] + [f"... and {len(out) - max_show} more groups"]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Check what a run will actually start from: which checkpoint tensors "
                    "load, which parameters are new, which learning-rate group each lands "
                    "in, and whether the checkpoint contract matches the config.")
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--checkpoint", type=Path, default=None,
                    help="defaults to load_from in the config")
    ap.add_argument("--show-all", action="store_true",
                    help="list every parameter name instead of folding by module")
    args_cli = ap.parse_args()

    import torch  # noqa: F401  (import here so --help works without torch)
    from engine_tools import build_model
    from src.utils import misc
    from src.utils.misc import prepare_checkpoint_state_for_model
    from src.utils.stream25_losses import make_stream25_param_groups
    from tools.stream25_runtime import load_stream25_args

    args = load_stream25_args(args_cli.config,
                              checkpoint_path=args_cli.checkpoint,
                              checkpoint_role="initialization")
    ckpt_path = args_cli.checkpoint or getattr(args, "load_from", None)

    print("=" * 78)
    print("Model initialization check")
    print("=" * 78)
    print(f"config     : {args_cli.config}")
    print(f"checkpoint : {ckpt_path}")
    print(f"exp_name   : {getattr(args, 'exp_name', '?')}")

    model = build_model(args)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"model      : {n_total / 1e6:.2f}M parameters")

    agg = getattr(model, "aggregator", None)
    if agg is not None and hasattr(agg, "patch_start_idx"):
        flags = []
        for name in ("use_time_token", "num_motion_tokens", "use_affine_token",
                     "use_sky_token", "use_ball_token"):
            val = getattr(agg, name, None)
            if val:
                flags.append(f"{name}={val}")
        print(f"aggregator : patch_start_idx={agg.patch_start_idx}"
              + (f"  [{', '.join(flags)}]" if flags else ""))

    # ---------------- checkpoint 加载 ----------------
    print("")
    print("-" * 78)
    print("Checkpoint load")
    print("-" * 78)
    if not ckpt_path or not Path(ckpt_path).exists():
        print(f"[FAIL] checkpoint not found: {ckpt_path}")
        print("       The run would start from random initialization.")
        return 2

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    raw_state = ckpt.get("model", ckpt)

    # ★ 必须走 load_model 用的同一条预处理，否则这个自检会给出假阴性。
    #   裸的 load_state_dict 会在 aggregator.affine_token 上报 size mismatch
    #   （三视图 ckpt -> 双视图模型是 [1,3,768] vs [1,2,768]），而真实训练路径
    #   会按相机名 index_select 把它接上；分辨率相关的 plucker buffer 同理。
    #   这里曾经直接调 load_state_dict，导致本来能训的配置被报成不兼容。
    state, camera_report = prepare_checkpoint_state_for_model(
        ckpt, model.state_dict(), args, checkpoint_path=ckpt_path
    )
    if camera_report.get("message"):
        print(f"[Model-init] {camera_report['message']}")
    else:
        print("[Model-init] no camera remapping needed (same camera list)")

    msg = model.load_state_dict(state, strict=False)
    missing, unexpected = list(msg.missing_keys), list(msg.unexpected_keys)
    loaded = len(state) - len(unexpected)

    print(f"tensors in checkpoint : {len(state)}")
    print(f"loaded into the model : {loaded}")
    print(f"missing (new/random)  : {len(missing)}")
    print(f"unexpected (dropped)  : {len(unexpected)}")

    if missing:
        print("")
        print("NEW parameters, randomly initialized -- confirm this is exactly what you")
        print("intended to add. Anything unexpected here means part of the network is not")
        print("being fine-tuned but trained from scratch, and load_model will not warn you:")
        for line in (missing if args_cli.show_all else _group_names(missing)):
            print(f"    {line}")
    if unexpected:
        print("")
        print("DROPPED checkpoint tensors -- these exist in the checkpoint but not in the")
        print("model. Fine when you deliberately removed a module; otherwise it means the")
        print("architecture drifted and you are silently losing trained weights:")
        for line in (unexpected if args_cli.show_all else _group_names(unexpected)):
            print(f"    {line}")
    if not missing and not unexpected:
        print("")
        print("Every tensor matched. This is a pure resume of the same architecture.")

    # ---------------- 契约对比 ----------------
    stored = ckpt.get("stream25_contract")
    if stored:
        print("")
        print("-" * 78)
        print("Checkpoint contract vs config")
        print("-" * 78)
        current = misc.stream25_checkpoint_contract(args)
        diffs = [(k, stored.get(k), current.get(k))
                 for k in sorted(set(stored) | set(current))
                 if stored.get(k) != current.get(k)]
        if diffs:
            for k, a, b in diffs:
                print(f"  [DIFF] {k}")
                print(f"         checkpoint: {a}")
                print(f"         config    : {b}")
            print("")
            print("validate_stream25_checkpoint_contract is a no-op (every branch is `pass`),")
            print("so none of these would be reported at training time. A difference in")
            print("context_frames, terminal_frame or timespan means the model is being asked")
            print("to do a different task than it was trained for.")
        else:
            print("  identical")

    # ---------------- 参数分组 ----------------
    print("")
    print("-" * 78)
    print("Learning-rate groups")
    print("-" * 78)
    head_lr = float(getattr(args, "lr", 0.0) or 0.0)
    trunk_lr = float(getattr(args, "stream25_trunk_lr", head_lr) or head_lr)
    groups = make_stream25_param_groups(
        model, head_lr=head_lr, trunk_lr=trunk_lr,
        weight_decay=float(getattr(args, "weight_decay", 0.05) or 0.05))
    by_id = {}
    for g in groups:
        for p in g["params"]:
            by_id[id(p)] = ("trunk" if g["group_name"].startswith("trunk") else "head")
    tot = {"trunk": 0, "head": 0}
    for _, p in model.named_parameters():
        if p.requires_grad and id(p) in by_id:
            tot[by_id[id(p)]] += p.numel()
    print(f"  trunk lr={trunk_lr:.2e}  {tot['trunk'] / 1e6:8.2f}M params")
    print(f"  head  lr={head_lr:.2e}  {tot['head'] / 1e6:8.2f}M params")
    if trunk_lr > 0:
        print(f"  ratio head/trunk = {head_lr / trunk_lr:.1f}x")

    if missing:
        print("")
        print("Where the NEW parameters landed (a new module in the trunk group gets the")
        print("small lr and usually fails to learn at all):")
        miss = set(missing)
        for name, p in model.named_parameters():
            if name in miss and id(p) in by_id:
                print(f"    {by_id[id(p)]:<5} lr="
                      f"{(trunk_lr if by_id[id(p)] == 'trunk' else head_lr):.2e}  {name}")

    # ---------------- freeze ----------------
    if getattr(args, "ball_token_freeze_backbone", False):
        print("")
        print("-" * 78)
        print("ball_token_freeze_backbone is ON")
        print("-" * 78)
        prefixes = ("ball_query", "ball_block", "ball_head",
                    "ball_token_norm", "aggregator.ball_token")
        trainable = [(n, p.numel()) for n, p in model.named_parameters()
                     if n.startswith(prefixes)]
        n_tr = sum(c for _, c in trainable)
        print(f"  trainable: {n_tr / 1e6:.4f}M across {len(trainable)} tensors")
        for n, c in trainable:
            print(f"    {n}  ({c})")
        if getattr(args, "use_ball_token_intrunk", False):
            print("")
            print("  WARNING: freeze combined with the in-trunk ball token. Putting the token")
            print("  inside the trunk exists so gradients reach the backbone; freezing the")
            print("  backbone turns it into a probe weaker than the external variant.")

    print("")
    print("=" * 78)
    return 1 if (missing and not ckpt_path) else 0


if __name__ == "__main__":
    raise SystemExit(main())
