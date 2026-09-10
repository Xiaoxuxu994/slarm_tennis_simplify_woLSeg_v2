# Ball-token 联合微调：B + 历史读取 + 因果前缀监督

日期：2026-09-10。优先启动 `exp0910_004`。这是一组追求尽快验证综合收益的联合实验，不是单因素消融，也不承诺必然提高精度。

改动前后的架构图、因果窗口和 loss 图见 [Ball-token 对比图解](BALL_TEMPORAL_COMPARISON.md)。

训练后的 token、轨迹、attention 热图和视频见 [Ball-token 可视化脚本](BALL_TOKEN_VISUALIZATION.md)。

## 本次组合

| 项目 | 原 B | 联合版 004 |
| --- | --- | --- |
| 输入观测 | 三目 `(0,3,6,9,12,15)` | 不变，不加入晚观测 |
| 末帧位置监督 | 每目同一 rig 球心 GT | 保留，但读取增强后的 tokens |
| 主 ball-token 读出 | 原 token 平均后回归 p/v | 轻量历史 patch 读取后的 token 平均后回归 p/v |
| 历史 token 状态监督 | 无直接监督 | frame6/9/12 的位置、速度、同一接球时刻位置 |
| `ball_latents` 导出 | 原末帧三目 tokens | 真正用于主 p/v 的增强三目 tokens |
| 新参数 | 0 | 约 191 万，hidden256、4 heads |

现有 C 的不同点：C 是完整 1536 维 block、仅用于辅助位置分支。本方案是 256 维时序模块，读取已有历史并直接进入主输出，不是简单把 C 的辅助 loss 再加大。

```text
固定六帧三目 RGB + 相机几何/时间
                |
       原因果 SLARM aggregator
          /                   \
  每时刻三目 ball tokens     每时刻三目 patches
          |                   |
          +--> 256维 query     +--> 投影后的历史 K/V memory
                       \     /
                 因果 cross-attention
                 仅访问当前及更早观测
                          |
               零初始化输出投影 + 原 token
                          |
                 增强后的三目 tokens
              /            |             \
       逐目共享头       mean ->共享头       导出 ball_latents
       p[t,view]        p_t, v_t              (frame15)
           |               |
   逐目球心监督       速度/轨迹/接球点监督
```

这里 `view` 表示视角编号，`v_t` 表示速度。未新增独立位置/运动角色 query；主干仍做跨视角融合。

## Loss 的具体含义

末帧仍使用 B 的逐目位置监督，原 pooled velocity、trajectory、landing 和所有重建损失保持原权重。

```text
L_joint = L_existing_B
        + 0.25 * L_prefix_pos
        + 0.25 * L_prefix_vel
        + 0.25 * L_prefix_landing
```

- prefix 选 frame6/9/12，内部权重为 `(1,2,4)/7`，不是直接求和。frame15 已有主监督，不重复添加；frame0/3 不直接强压速度监督。
- `L_prefix_pos`：每个被选时刻，每目增强 token 经共享头预测对应 rig 球心，与该时刻同一个 GT 比较，按 batch/view/坐标平均。
- `L_prefix_vel`：同一时刻增强 tokens 的 mean 经共享头预测速度，与对应速度 GT 比较。
- `L_prefix_landing`：每个 prefix 都预测同一个绝对接球时刻。`dt_t = catch_dt_from_frame15 + timestamp15 - timestamp_t`，不是给所有 prefix 用 frame15 的 dt。
- position/landing 仍用 0.1 m 归一化，velocity 沿用原主速度损失尺度；新增几何损失以 FP32 计算。
- GT 只进损失。新 attention 使用 `context_time`、视角标识和预测特征，不读取 GT 球心、速度、mask、目标图像或未来 latent。

这不是“标注变多就必然更准”：新增约束要求不同因果前缀的特征都能读出物理状态。联合收益也可能来自新读出容量，需要备用消融判断。

## 初始化、输出和流式接口

新增模块输出投影为零，开始时增强 token 等于原 token。旧 `007999` 加载后，只应缺少 `ball_temporal.*` 新参数；已有 ball head 和主干都应加载。不要使用旧 config 评估新 checkpoint，否则可能忽略新模块。

修复了一个初始化问题：SLARM 最后的统一 Linear 初始化会覆盖 C 之前设置的零残差。本次在统一初始化后重新置零 C 和新模块的残差输出。已有 C checkpoint 的训练权重仍会在加载时覆盖初始化值，不会被强制清零。

| 输出 | 形状 | 含义 |
| --- | --- | --- |
| `ball_pos15`, `ball_v15` | `[B,3]` | 当前末观测时刻的主状态 |
| `ball_latents` | `[B,3,1536]` | 主状态实际使用的增强末帧 tokens |
| `ball_latents_raw` | `[B,3,1536]` | 增强前 tokens，用于诊断，仅 temporal 开启时有 |
| `ball_pos15_per_view` | `[B,3,3]` | 末时刻逐目位置读出 |
| `ball_prefix_states` | `[B,T,6]` | 每个因果 prefix 的 pooled p/v |
| `ball_prefix_positions_per_view` | `[B,T,3,3]` | 每个 prefix 的逐目位置 |

标准 StreamSession 累积 prefix 的时间轴，但覆盖末时刻状态/latent；`clear()` 同时清除新 K/V memory。完整六帧与逐帧输入共享时序读取实现，新增检查拒绝漏传历史、不同步缓存和第七次观测。frame15 之后不生成新的观测 token。

缓存只保存投影后的 K/V，不保存原始历史 patches；仍有额外显存和计算，不是零成本。默认尺寸下最终 K/V 的 FP32 存储约 44 MB/样本，训练还需中间激活，实际峰值和吞吐必须 GPU 实测。

## 配置选择

- [004 联合版](../configs/exp0910_004_balltoken_temporal_joint.yml)：本次推荐，B + 历史读取 + 前缀监督。
- [005 prefix-only](../configs/exp0910_005_balltoken_prefix_only.yml)：只增加前缀监督、不增加参数；作为省显存备用或区分历史读取收益的消融。
- 原 `001 control / 002 B / 003 C` 配置不变。新开关与新 loss 默认关闭。

两组新配置均从 `exp0908_003/ckpt_007999.pth` 初始化；4 GPU、batch size 1/GPU、4000 steps、warmup 200、cosine，head LR `1e-5`、trunk LR `1e-6`。其余数据、seed、损失与已有 B 一致；不恢复旧 optimizer。

## 完整运行命令

在具有 checkpoint、数据和 CUDA SLARM 环境的机器上，进入仓库根目录。先确认初始化：

```bash
SLARM_SINGLE_PROCESS=1 python tools/check_model_init.py \
  --config configs/exp0910_004_balltoken_temporal_joint.yml
```

`run_sh/train.sh` 已默认选用 004、GPU 0/1/2/3、`RESUME=0`。直接启动：

```bash
bash run_sh/train.sh
```

可以直接使用之前训练好的 `exp0908_003/ckpt_007999.pth`，不必先跑 B/C。本次是 `load_from` 权重初始化，004 从第 0 步开始新优化器和 4000-step 日程；不要用 `--resume_from` 指向旧实验。

若另一台机器的权重放在不同位置：

```bash
bash run_sh/train.sh --load_from /absolute/path/to/ckpt_007999.pth
```

这里应使用相同架构的三目 in-trunk ball-token checkpoint，不要替换成纯像素或外挂 ball-token 权重。初始化检查中，旧主干、`aggregator.ball_token`、`ball_token_norm.*` 和 `ball_head_intrunk.*` 都应加载；预期新增缺失项仅为 `ball_temporal.*`。

换四张 GPU：`GPUS=4,5,6,7 bash run_sh/train.sh`。只在恢复已经中断的 **004 自己的运行** 时使用 `RESUME=1 bash run_sh/train.sh`；不能用它续旧 007999 实验。

等价的直接启动命令（不启用脚本额外的 TensorBoard）：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 main_slarm.py \
  --config configs/exp0910_004_balltoken_temporal_joint.yml
```

先以 1k checkpoint 检查主输出是否明显退化，不把未跑完 cosine 的结果当作最终收敛：

```bash
CUDA_VISIBLE_DEVICES=0 SLARM_SINGLE_PROCESS=1 python -u tools/verify_physics_extrapolation.py \
  --config configs/exp0910_004_balltoken_temporal_joint.yml \
  --checkpoint work_dirs/slarm/exp0910_004_balltoken_temporal_joint/checkpoints/ckpt_000999.pth \
  --split validation --limit 20 --target-frames 24,45 --catch-frame 45 \
  --ball-mask-source both --ball-radius-compensation 0
```

最终同一批 100 场景评估：

```bash
CUDA_VISIBLE_DEVICES=0 SLARM_SINGLE_PROCESS=1 python -u tools/verify_physics_extrapolation.py \
  --config configs/exp0910_004_balltoken_temporal_joint.yml \
  --checkpoint work_dirs/slarm/exp0910_004_balltoken_temporal_joint/checkpoints/ckpt_003999.pth \
  --split validation --limit 100 --target-frames 24,45 --catch-frame 45 \
  --ball-mask-source both --ball-radius-compensation 0
```

备用配置启动命令：

```bash
CONFIG=configs/exp0910_005_balltoken_prefix_only.yml bash run_sh/train.sh
```

评估备用实验时同时替换 config 和 checkpoint 路径。若训练占满四卡，不要在同卡同时启动评测导致显存不足；使用空闲卡或在 checkpoint 保存后另行安排。

## 验收和当前验证范围

训练日志应出现 `stream25_ball_prefix_pos_loss`、`stream25_ball_prefix_vel_loss`、`stream25_ball_prefix_landing_loss`，以及各 frame6/9/12 的米制诊断。辅助项下降不是胜出判据。

主判据仍是相同 validation 场景的 balltoken frame45 median/p95，并检查 pos15、vel15、pixel-path 和球重建是否退化。当前 verify 最后一个 `balltoken frame45` 行读取的是增强后的主状态，但它不会逐项打印新 prefix 诊断。frame45 GT 是解析弹道参考，不是实际机器人接球成功。

CPU 测试覆盖实际读出/初始化/Session 方法、旧权重兼容、主状态与导出一致、因果梯度、BF16 时序模块完整/流式一致、缓存重置、配置接线和 prefix 损失：

```bash
python -m pytest tests/models/test_ball_temporal.py \
  tests/models/test_ball_temporal_integration.py tests/models/test_ball_prefix_losses.py \
  tests/models/test_ball_position_ablation.py tests/models/test_ball_latent_export.py \
  tests/utils/test_ball_trajectory_losses.py -q
```

CPU 测试通过不等于已完成真实 CUDA 训练验证。本地没有训练数据、权重和完整 CUDA renderer 环境，真实模型 checkpoint 前向、显存、吞吐和精度仍需训练机确认。

训练改动未包含二维球心检测/三角化、背景配对一致性、球级 GS 重构或端点重参数化。后续补充的独立可视化工具见上方链接，不改变训练路径。
