#!/usr/bin/env bash
set -uo pipefail

# 用 verify_physics_extrapolation.py 扫一串 ckpt，看指标随训练步数怎么走。
#
# 用途：判断一次训练到底收敛了没有。单看两个 ckpt 分不清「持续下降」和
# 「某个点恰好是低谷」，多打几个点就清楚了。比完整 eval 快一个量级
# （--limit 40，几分钟一个），足够看趋势。
#
# 用法：
#   bash run_sh/verify_sweep.sh              # 按下面 ITERS 跑
#   bash run_sh/verify_sweep.sh 029999 039999  # 或命令行直接给迭代号
#
# 跑完屏幕上有对比表，同时在输出目录留一份 summary.md（可直接粘贴）
# 和每个 ckpt 的完整日志。

# ============================================================
# 改这里
# ============================================================

GPU="4"
CONFIG="configs/exp0825_002_slarm_stream25_6.5cm_triview_window6_nolseg_loadpre.yml"
CKPT_DIR="work_dirs/slarm/exp0825_002_slarm_stream25_6.5cm_triview_window6_nolseg_loadpre/checkpoints"

# 要扫的迭代号。"auto" = 自动发现 CKPT_DIR 下所有 ckpt_*.pth 并按步数排序
ITERS="029999 033999 035999 037999 039999"
# ITERS="auto"

LIMIT=40                 # 场景数。40 与 docs/BALL_LANDING_FINDINGS.md 的基线同口径
MASK_SOURCE="pred"       # pred / gt / both。已确认 pred≈gt，扫趋势用 pred 即可，省一半时间
GRAVITY="0,0,-9.81"
SPLIT="validation"

# ============================================================

cd "$(dirname "${BASH_SOURCE[0]}")/.."

[ -f "${CONFIG}" ] || { echo "config 不存在: ${CONFIG}"; exit 1; }
[ -d "${CKPT_DIR}" ] || { echo "ckpt 目录不存在: ${CKPT_DIR}"; exit 1; }

# 命令行参数优先于上面的 ITERS
if [ "$#" -gt 0 ]; then
    ITERS="$*"
fi

if [ "${ITERS}" = "auto" ]; then
    ITERS="$(ls "${CKPT_DIR}"/ckpt_*.pth 2>/dev/null \
             | sed 's#.*/ckpt_##; s#\.pth##' | grep -E '^[0-9]+$' | sort -n | tr '\n' ' ')"
    [ -n "${ITERS}" ] || { echo "在 ${CKPT_DIR} 下没找到 ckpt_*.pth"; exit 1; }
    echo "auto 发现 $(echo ${ITERS} | wc -w) 个 ckpt"
fi

EXP_NAME="$(basename "${CKPT_DIR%/checkpoints}")"
OUT_DIR="work_dirs/slarm/verify_sweep/${EXP_NAME}"
mkdir -p "${OUT_DIR}"

echo "════════════════════════════════════════════════════════"
echo "config : ${CONFIG}"
echo "ckpts  : ${CKPT_DIR}"
echo "iters  : ${ITERS}"
echo "limit  : ${LIMIT}   mask_source: ${MASK_SOURCE}   split: ${SPLIT}"
echo "out    : ${OUT_DIR}"
echo "GPU    : ${GPU}"
echo "════════════════════════════════════════════════════════"
echo ""

DONE_LIST=""
for it in ${ITERS}; do
    CKPT="${CKPT_DIR}/ckpt_${it}.pth"
    LOG="${OUT_DIR}/verify_${it}.log"

    if [ ! -f "${CKPT}" ]; then
        echo "[skip] ckpt_${it}.pth 不存在"
        continue
    fi

    echo "──────── ckpt_${it} ────────"
    START=$(date +%s)

    CUDA_VISIBLE_DEVICES="${GPU}" SLARM_SINGLE_PROCESS=1 \
    python tools/verify_physics_extrapolation.py \
        --config "${CONFIG}" \
        --checkpoint "${CKPT}" \
        --split "${SPLIT}" \
        --ball-mask-source "${MASK_SOURCE}" \
        --gravity "${GRAVITY}" \
        --limit "${LIMIT}" \
        > "${LOG}" 2>&1

    RC=$?
    ELAPSED=$(( $(date +%s) - START ))

    if [ ${RC} -ne 0 ]; then
        echo "  失败（退出码 ${RC}），最后几行："
        tail -5 "${LOG}" | sed 's/^/    /'
        continue
    fi

    echo "  完成，用时 ${ELAPSED}s  ->  ${LOG}"
    grep -E "pos15_error|phys\(gravity\)" "${LOG}" | sed 's/^/    /'
    DONE_LIST="${DONE_LIST} ${it}"
done

echo ""
echo "════════════════════════════════════════════════════════"

python3 - "${OUT_DIR}" ${DONE_LIST} <<'PY'
"""把各 ckpt 的 verify 日志解析成一张对比表。

verify 的输出是固定宽度的表格，形如：
    region   metric               median        p95  n_valid
    pred     pos15_error          0.0364     0.0863       40
    pred     phys(gravity)        0.0417     0.1062       40
按 (region, metric) 抓 median / p95 两列。
"""
import re, sys, pathlib

out_dir = pathlib.Path(sys.argv[1])
iters = sys.argv[2:]
if not iters:
    print("没有成功完成的 ckpt，无法汇总")
    raise SystemExit(0)

WANT = [
    ("pred", "pos15_error",   "pos15"),
    ("pred", "v15_error",     "v15"),
    ("pred", "phys(gravity)", "frame24_phys"),
    ("pred", "free(current)", "frame24_free"),
]

def parse(log_path):
    got = {}
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split()
        if len(parts) < 4:
            continue
        region, metric = parts[0], parts[1]
        for r, m, key in WANT:
            if region == r and metric == m:
                try:
                    got[key] = (float(parts[2]), float(parts[3]))
                except ValueError:
                    pass
    return got

rows = []
for it in iters:
    p = out_dir / f"verify_{it}.log"
    if p.exists():
        rows.append((it, parse(p)))

if not rows:
    print("日志都解析不出数字，检查 verify 输出格式是否变了")
    raise SystemExit(0)

hdr = f"{'ckpt':>9} | {'pos15 med':>10} {'pos15 p95':>10} | {'v15 med':>9} | {'f24 med':>9} {'f24 p95':>9}"
sep = "-" * len(hdr)
print(hdr); print(sep)
lines_md = ["| ckpt | pos15 med | pos15 p95 | v15 med | frame24 med | frame24 p95 |",
            "| --- | ---: | ---: | ---: | ---: | ---: |"]
for it, g in rows:
    def f(key, idx):
        return f"{g[key][idx]:.4f}" if key in g else "—"
    print(f"{it:>9} | {f('pos15',0):>10} {f('pos15',1):>10} | {f('v15',0):>9} |"
          f" {f('frame24_phys',0):>9} {f('frame24_phys',1):>9}")
    lines_md.append(f"| {it} | {f('pos15',0)} | {f('pos15',1)} | {f('v15',0)} "
                    f"| {f('frame24_phys',0)} | {f('frame24_phys',1)} |")
print(sep)

# 趋势判读：只在有 3 个以上点、且 pos15 都拿到时给（pos15 越小越好）
vals = [(it, g["pos15"][0]) for it, g in rows if "pos15" in g]
if len(vals) >= 3:
    first, last = vals[0][1], vals[-1][1]
    drops = sum(1 for a, b in zip(vals, vals[1:]) if b[1] < a[1])
    change = (1 - last / first) * 100
    verb = "改善" if change > 0 else "退化"
    print(f"\n首 {vals[0][0]} = {first:.4f}   末 {vals[-1][0]} = {last:.4f}   "
          f"{verb} {abs(change):.1f}%")
    print(f"相邻下降的段数：{drops}/{len(vals)-1}")

    # 异常点：与左右邻居均值偏离超过 50%，多半是训练抖动而非趋势
    outliers = []
    for i in range(1, len(vals) - 1):
        neighbour = (vals[i - 1][1] + vals[i + 1][1]) / 2
        if neighbour > 0 and vals[i][1] > neighbour * 1.5:
            outliers.append((vals[i][0], vals[i][1], neighbour))
    for it_o, v_o, nb in outliers:
        print(f"[!] {it_o} = {v_o:.4f}，而左右邻居均值只有 {nb:.4f}（高 {v_o/nb:.1f}×）"
              f" —— 多半是训练抖动，不要拿它当趋势的一部分解读")

    clean = [v for v in vals if v[0] not in {o[0] for o in outliers}]
    tail = clean if len(clean) >= 3 else vals
    tail_drops = sum(1 for a, b in zip(tail, tail[1:]) if b[1] < a[1])
    if tail_drops == len(tail) - 1:
        print("→ 剔除异常点后单调下降：尚未收敛，继续训 stage1 的收益可能高于架构改动")
    elif tail[-1][1] <= min(v for _, v in tail):
        print("→ 末点是（剔除异常点后的）全程最优，且中间有小幅反弹：接近收敛，"
              "剩余波动来自训练噪声")
    else:
        best = min(tail, key=lambda x: x[1])
        print(f"→ 最优在 {best[0]}（{best[1]:.4f}）而非末点：曲线已平，末点未必最好；"
              f"后续实验建议用 {best[0]} 作为起点")

md = out_dir / "summary.md"
md.write_text("\n".join(lines_md) + "\n", encoding="utf-8")
print(f"\nmarkdown 表已存到 {md}（可直接粘贴）")
PY

echo ""
echo "完整日志在: ${OUT_DIR}/"
