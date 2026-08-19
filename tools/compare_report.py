#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
精度无损对比 —— 报告脚本（独立，不依赖任何 repo 代码，只要有 torch）。

把 compare_dump.py 产出的 dump 文件传进来，第一个作为参考基线，
其余每个与基线逐 key 对比：形状 / 最大绝对误差 / 平均绝对误差 / 是否逐比特相等 / 是否 allclose，
并对关键渲染量按帧（含外推帧 >=16）分桶打印。

用法：
  # v2 vs v1 —— 期望「逐比特相等」(exact=True)，因为 v2 只是删死代码 + 重构 split
  python compare_report.py dump_woLSeg.pt dump_v2.pt

  # v1 vs baseline —— 期望除 feat 相关 key 外全部 allclose（~1e-5 级浮点误差）
  python compare_report.py dump_baseline.pt dump_woLSeg.pt

  # 三者一起（都与 baseline 比）
  python compare_report.py dump_baseline.pt dump_woLSeg.pt dump_v2.pt

判定标准：
  exact=True                 -> 逐比特一致（纯代码重构应达到，如 v2 vs v1）
  exact=False & allclose=True-> 数值等价（浮点顺序差异，如 v1 vs baseline 的非 feat 输出）
  allclose=False             -> 有实质差异，需排查
"""
import sys
import torch

ATOL, RTOL = 1e-4, 1e-3
# 关注的关键轨迹 / 渲染量（按帧分桶时只看这些，避免刷屏）
FOCUS = ["rendered_image", "rendered_depth", "rendered_target_ms3",
         "rendered_flow", "rendered_task_semantic_logits"]


def diff_stats(ref, cur):
    if ref.shape != cur.shape:
        return None
    d = (ref.double() - cur.double()).abs()
    denom = ref.double().abs().max().item() + 1e-12
    return {
        "max": d.max().item(),
        "mean": d.mean().item(),
        "rel": d.max().item() / denom,
        "exact": bool(torch.equal(ref, cur)),
        "allclose": bool(torch.allclose(ref, cur, atol=ATOL, rtol=RTOL)),
    }


def per_frame(ref, cur, frame_ids):
    t = len(frame_ids)
    if ref.dim() < 2 or ref.shape[1] != t or cur.shape[1] != t:
        return None
    lines = []
    for i, fid in enumerate(frame_ids):
        d = (ref[:, i].double() - cur[:, i].double()).abs().max().item()
        tag = "extrap" if fid >= 16 else ("anchor" if fid == 15 else "ctx/interp")
        lines.append((int(fid), tag, d))
    return lines


def main():
    files = sys.argv[1:]
    if len(files) < 2:
        print("用法: python compare_report.py <ref.pt> <other.pt> [other2.pt ...]")
        sys.exit(1)

    dumps = [torch.load(f, map_location="cpu") for f in files]
    ref_name, ref = files[0], dumps[0]

    frame_ids = None
    if "meta.target_frame_idx" in ref:
        fi = ref["meta.target_frame_idx"]
        frame_ids = (fi[0] if fi.dim() > 1 else fi).tolist()
        print(f"[info] target_frame_idx = {frame_ids}")

    worst_overall = 0.0
    for name, cur in zip(files[1:], dumps[1:]):
        print("\n" + "=" * 92)
        print(f"参考基线  {ref_name}   vs   {name}")
        print("=" * 92)

        rk, ck = set(ref), set(cur)
        only_ref = sorted(k for k in rk - ck if not k.startswith("meta."))
        only_cur = sorted(k for k in ck - rk if not k.startswith("meta."))
        if only_ref:
            print("仅参考基线有（本版缺失，woLSeg 版这里应恰好是 feat 相关渲染项）:")
            for k in only_ref:
                print("   -", k)
        if only_cur:
            print("仅本版有:")
            for k in only_cur:
                print("   +", k)

        shared = sorted(k for k in rk & ck if not k.startswith("meta."))
        print(f"\n{'key':46s} {'shape':22s} {'max_abs':>10s} {'mean_abs':>10s} "
              f"{'exact':>6s} {'close':>6s}")
        print("-" * 92)
        worst = 0.0
        for k in shared:
            s = diff_stats(ref[k], cur[k])
            if s is None:
                print(f"{k:46s} 形状不一致 {tuple(ref[k].shape)} vs {tuple(cur[k].shape)}")
                continue
            worst = max(worst, s["max"])
            flag = "" if s["allclose"] else "   <<< 差异!"
            print(f"{k:46s} {str(tuple(ref[k].shape)):22s} {s['max']:10.2e} "
                  f"{s['mean']:10.2e} {str(s['exact']):>6s} {str(s['allclose']):>6s}{flag}")
        worst_overall = max(worst_overall, worst)
        print("-" * 92)
        print(f"共享 key 最大绝对误差 = {worst:.3e}  "
              f"({'全部 allclose，通过' if worst <= ATOL else '存在超阈值差异，需排查'})")

        if frame_ids is not None:
            for k in shared:
                if not any(t in k for t in FOCUS):
                    continue
                lines = per_frame(ref[k], cur[k], frame_ids)
                if not lines:
                    continue
                print(f"\n[按帧] {k}")
                for fid, tag, d in lines:
                    mark = "  <<<" if d > ATOL else ""
                    print(f"   frame {fid:2d} [{tag:10s}] max_abs_diff = {d:.3e}{mark}")

    print("\n" + "#" * 92)
    print(f"# 全部对比里的最大绝对误差 = {worst_overall:.3e}")
    print("#" * 92)


if __name__ == "__main__":
    main()
