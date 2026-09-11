"""关键指标提取与报表。

被 scripts/eval_stream25_base.py（单次 eval 摘要表）和 tools/compare_eval.py
（两次实验对比表）复用。只依赖标准库；阈值表按需延迟 import，避免拉起 torch。

metrics 结构约定（= evaluation.json 的顶层 "metrics"，即 aggregate scope）：
  metrics[<scalar>][<bucket>]                 e.g. rgb_psnr / semantic_miou / ball_iou / rgb_psnr_p10 / depth_absrel
  metrics["ball_depth_error_median"][<bucket>], metrics["ball_depth_error_p95"][<bucket>]
  metrics["ms3_ball_velocity"]["median"/"p95"]（accel/jerk/static/context 同理）
  metrics["frame24_position"]["median"/"p95"]
  metrics["frame24_position_balltoken"/"ball_pos15_error"/"ball_vel15_error"]["median"/"p95"]
      —— 仅当模型带内建 ball token (use_ball_token) 时存在；缺席时表格显示 n/a
bucket ∈ {anchor, interpolation, near, mid, far, farthest}
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

# (label, direction, [(metric_key, subkey), ...], threshold_key)
#   direction: "up" = 越大越好, "down" = 越小越好
#   threshold_key: ACCEPTANCE_TABLE 里的键；None 表示不设门
KEY_METRICS: List[Tuple[str, str, List[Tuple[str, str]], Optional[str]]] = [
    ("frame24 position med / p95",    "down", [("frame24_position", "median"), ("frame24_position", "p95")], "frame24_position"),
    # 内建 ball token 的并列口径（无 ball token 的模型这三行显示 n/a；不参与 acceptance gate）
    ("frame24 balltoken med / p95",   "down", [("frame24_position_balltoken", "median"), ("frame24_position_balltoken", "p95")], None),
    ("balltoken pos15 med / p95",     "down", [("ball_pos15_error", "median"), ("ball_pos15_error", "p95")], None),
    ("balltoken vel15 med / p95",     "down", [("ball_vel15_error", "median"), ("ball_vel15_error", "p95")], None),
    # 多帧弹道拟合读出：用历史位置 + 已知重力反解 (pos15, v15)，代替 head 直接回归速度。
    # 只在 ball_prefix_supervision 打开的 ckpt 上有值。与上面三行并排看：
    #   fit 的 vel15 明显更好而 pos15 持平 -> 直接回归速度是瓶颈，应换成拟合读出。
    #   两者都没变好 -> 位置的逐帧误差本身就是瓶颈，拟合救不了。
    ("balltoken fit frame24 med / p95", "down", [("frame24_position_balltoken_fit", "median"), ("frame24_position_balltoken_fit", "p95")], None),
    ("balltoken fit pos15 med / p95",   "down", [("ball_pos15_error_balltoken_fit", "median"), ("ball_pos15_error_balltoken_fit", "p95")], None),
    ("balltoken fit vel15 med / p95",   "down", [("ball_vel15_error_balltoken_fit", "median"), ("ball_vel15_error_balltoken_fit", "p95")], None),
    # 逐 prefix 帧的位置误差。决定拟合该用几帧 —— 早期帧劣化超过后期约 1.6 倍时，
    # 缩短时间跨度的代价就盖过去掉坏样本的好处（见 stream25_metrics.fit_ballistic_state）。
    # 逐帧速度读出去重力后平均。spread 是这几个估计之间的散布：
    #   spread ~ 0    -> 各帧读出实质相同，平均无用
    #   spread 大且 vavg 误差明显低于 balltoken vel15 -> 逐帧读出带独立信息
    ("balltoken vavg frame24 / vel15",  "down", [("frame24_position_balltoken_vavg", "median"), ("ball_vel15_error_balltoken_vavg", "median")], None),
    ("balltoken vel15 spread",          "down", [("ball_vel15_spread_balltoken", "median")], None),
    ("balltoken prefix pos f0 / f15",   "down", [("ball_prefix_pos_error_frame0", "median"), ("ball_pos15_error", "median")], None),
    # 像素路径的多帧弹道拟合读出。和最上面的 frame24 position 直接可比（同为三目取最差）。
    #   fit vel15 明显好 -> MS3 直接预测速度不是最优，换拟合读出，零训练。
    #   constant >> scatter -> 逐帧误差主要是恒定偏置，拟合能赢很多。
    #   constant ≈ scatter -> 误差主要是逐帧抖动，拟合只能小赢。
    # 接球帧落点。★ 这是唯一跨 context offset 可比的数：frame24_* 的外推时长
    # 随窗口滑动而变，catch_* 的目标固定在绝对时刻 catch_frame 上，所以
    # "把观测窗口后移" 买到了多少必须看这两行。catch horizon 打印的是
    # (catch_frame - 终端帧) 的秒数，滑窗后它会变短 —— 那正是收益的来源。
    ("catch position med / p95",        "down", [("catch_position", "median"), ("catch_position", "p95")], None),
    ("catch position balltoken",        "down", [("catch_position_balltoken", "median")], None),
    ("catch horizon s",                 "down", [("catch_horizon_s", "median")], None),
    ("pixel fit frame24 med / p95",     "down", [("frame24_position_fit", "median"), ("frame24_position_fit", "p95")], None),
    ("pixel fit pos15 med / p95",       "down", [("ball_pos15_error_fit", "median"), ("ball_pos15_error_fit", "p95")], None),
    ("pixel fit vel15 med / p95",       "down", [("ball_vel15_error_fit", "median"), ("ball_vel15_error_fit", "p95")], None),
    ("pixel pos err const / scatter",   "down", [("pixel_pos_error_constant_m", "median"), ("pixel_pos_error_scatter_m", "median")], None),
    ("pixel pos err f0 / f15",          "down", [("pixel_pos_error_frame0", "median"), ("pixel_pos_error_frame15", "median")], None),
    ("ball velocity med / p95",       "down", [("ms3_ball_velocity", "median"), ("ms3_ball_velocity", "p95")], "ms3_ball_velocity"),
    ("ball acceleration med / p95",   "down", [("ms3_ball_acceleration", "median"), ("ms3_ball_acceleration", "p95")], "ms3_ball_acceleration"),
    ("ball jerk med / p95",           "down", [("ms3_ball_jerk", "median"), ("ms3_ball_jerk", "p95")], "ms3_ball_jerk"),
    ("ball depth farthest med / p95", "down", [("ball_depth_error_median", "farthest"), ("ball_depth_error_p95", "farthest")], None),
    ("ball IoU anchor",               "up",   [("ball_iou", "anchor")], "ball_iou"),
    ("ball IoU farthest",             "up",   [("ball_iou", "farthest")], "ball_iou"),
    ("ball RGB PSNR anchor",          "up",   [("ball_rgb_psnr", "anchor")], "ball_rgb_psnr"),
    ("semantic mIoU anchor",          "up",   [("semantic_miou", "anchor")], "semantic_miou"),
    ("semantic mIoU farthest",        "up",   [("semantic_miou", "farthest")], "semantic_miou"),
    ("RGB PSNR anchor",               "up",   [("rgb_psnr", "anchor")], "rgb_psnr"),
    ("RGB PSNR farthest",             "up",   [("rgb_psnr", "farthest")], "rgb_psnr"),
    ("RGB p10 farthest",              "up",   [("rgb_psnr_p10", "farthest")], "rgb_psnr_p10"),
    ("depth absrel farthest",         "down", [("depth_absrel", "farthest")], "depth_absrel"),
]


def _get(metrics: Dict[str, Any], key: str, sub: str) -> float:
    node = metrics.get(key) if isinstance(metrics, dict) else None
    if isinstance(node, dict):
        try:
            return float(node.get(sub))
        except (TypeError, ValueError):
            return float("nan")
    return float("nan")


def _fmt(v: float) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "n/a"
    return f"{v:.3f}"


def extract_rows(metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = []
    for label, direction, paths, thr_key in KEY_METRICS:
        vals = [_get(metrics, k, s) for (k, s) in paths]
        rows.append({
            "label": label, "direction": direction,
            "values": vals, "paths": paths, "threshold_key": thr_key,
        })
    return rows


def _threshold(thr_key: Optional[str], sub: str):
    if not thr_key:
        return None
    from src.utils.stream25_metrics import ACCEPTANCE_TABLE  # 延迟 import（避免拉 torch）
    row = ACCEPTANCE_TABLE.get(thr_key)
    return row.get(sub) if isinstance(row, dict) else None


def render_single_markdown(metrics: Dict[str, Any]) -> str:
    """单次 eval 关键指标摘要：指标 | 值 | 阈值 | 达标。"""
    out = ["| 指标 | 值 | 阈值 | 达标 |", "| --- | ---: | ---: | :---: |"]
    for row in extract_rows(metrics):
        vals, direction, paths = row["values"], row["direction"], row["paths"]
        val_str = " / ".join(_fmt(v) for v in vals)
        thr = _threshold(row["threshold_key"], paths[0][1])
        thr_str = _fmt(thr) if thr is not None else "—"
        ok = "—"
        if thr is not None and not (isinstance(vals[0], float) and math.isnan(vals[0])):
            passed = (vals[0] >= thr) if direction == "up" else (vals[0] <= thr)
            ok = "✅" if passed else "❌"
        out.append(f"| {row['label']} | {val_str} | {thr_str} | {ok} |")
    return "\n".join(out)


def _verdict(base: float, cand: float, direction: str, tol: float = 0.02) -> str:
    if math.isnan(base) or math.isnan(cand):
        return "—"
    denom = abs(base) if base != 0 else 1e-9
    rel = (cand - base) / denom
    better = (rel > tol) if direction == "up" else (rel < -tol)
    worse = (rel < -tol) if direction == "up" else (rel > tol)
    if better:
        return "✅ 改善"
    if worse:
        return "❌ 明显退化" if abs(rel) > 0.5 else "❌ 退化"
    return "≈ 基本一致"


def render_compare_markdown(metrics_a: Dict[str, Any], metrics_b: Dict[str, Any],
                            label_a: str = "A", label_b: str = "B") -> str:
    """两次实验对比：指标 | A | B | 判断（含 p95 长尾提示）。"""
    out = [f"| 指标 | {label_a} | {label_b} | 判断 |", "| --- | ---: | ---: | --- |"]
    for ra, rb in zip(extract_rows(metrics_a), extract_rows(metrics_b)):
        va = " / ".join(_fmt(v) for v in ra["values"])
        vb = " / ".join(_fmt(v) for v in rb["values"])
        verdict = _verdict(ra["values"][0], rb["values"][0], ra["direction"])
        note = ""
        if len(ra["values"]) > 1:
            v_med = _verdict(ra["values"][0], rb["values"][0], ra["direction"])
            v_p95 = _verdict(ra["values"][1], rb["values"][1], ra["direction"])
            if "退化" in v_p95 and "退化" not in v_med:
                note = "（p95 长尾）"
            elif "明显" in v_p95 and "明显" not in v_med:
                note = "（p95 长尾更重）"
        out.append(f"| {ra['label']} | {va} | {vb} | {verdict}{note} |")
    return "\n".join(out)
