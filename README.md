# SLARM 流式重建系统文档（woLSeg_v2）

本仓是 SLARM 的精简版：只保留**流式重建（Stream25 base）**的训练 / 评估 / 推理，
移除了下游接球模块（CatchStateReader）、STORM、data_gen 采集链路，以及 LSeg 特征监督。

## 1. 系统目标

系统接收六次同步三目观测：

```text
frame 0 → 3 → 6 → 9 → 12 → 15
每次观测：front_left + front_right + lower_front 的 RGB + 相机内外参 + 物理时间
```

端到端流程：

```text
三目稀疏观测(6 次)
  → ViT/Aggregator 因果时空聚合(window_6)
  → 逐帧 Gaussian + MS3 运动
  → 渲染 25 帧:RGB / metric depth / 四类语义 / MS3
  → frame 16–24 由 frame-15 terminal context 外推
```

即：用六次观测重建 `[0,25)` 共 25 帧的 RGB、metric depth、四类语义和 MS3，并从最后一次观测外推到 frame 24。

## 2. 冻结的三目与时序合同

### 2.1 相机

相机顺序不可改变：

| 顺序 |     名称      | rig/FLU offset（m）  | Pitch  |
| :--: | :-----------: | :------------------: | :----: |
|  0   | `front_left`  | `[0.00,+0.20,0.00]`  |  `0°`  |
|  1   | `front_right` | `[0.00,-0.20,0.00]`  |  `0°`  |
|  2   | `lower_front` | `[+0.30,0.00,-1.00]` | `+27°` |

`front_left` 是 canonical/reference camera。Pitch 只属于 `lower_front` 的相机外参，rig/FLU 坐标系本身没有旋转。rig 原点位于离地 1.5 m，因此 `lower_front` 的世界高度为 0.5 m。

### 2.2 图像与 token

| 项目                 |              数值 |
| -------------------- | ----------------: |
| 输入高×宽            |         `320×240` |
| Patch size           |               `8` |
| 每目 patch           |      `40×30=1200` |
| Aggregator embed dim |             `768` |

### 2.3 时间

- 原始仿真以 30 FPS 记录 30 帧，用于接触边界和数据完整性审计；
- SLARM 训练/评估使用半开区间 `[0,25)`，即 frame 0–24；
- 只有 `[0,3,6,9,12,15]` 进入因果上下文；
- 每个训练 step 采样 7 个 target：1 anchor、2 interpolation、4 extrapolation；
- validation/test 对 25 帧完整评估；
- frame 16–24 由 frame-15 terminal context 独占外推责任。

## 3. 相对原始 SLARM 的网络改动

### 3.1 真流式 `window_6`

[`src/models/stream_session.py`](src/models/stream_session.py) 提供真正的六步因果会话：每次只接收 `[B,1,V,C,H,W]`，维护 Aggregator/CameraHead KV cache，并强制帧序为 `[0,3,6,9,12,15]`。

在 `terminal_context_extrapolation=True` 时，前五次调用只累计上下文和 Gaussian；第六次调用后才统一渲染全部 targets。

> 注：训练与 `render_stream25_base.py` 走「整段一次前向 `model(input_dict)`」，`eval_stream25_base.py` 与 `inference_stream.py` 走「StreamSession 逐帧」。两条路径靠 window 因果 mask 设计上等价，可用 [`tools/check_render_vs_stream.py`](tools/check_render_vs_stream.py) 校验一致性。

### 3.2 Terminal-context 外推

[`src/models/slarm.py`](src/models/slarm.py) 的 `terminal_context_extrapolation`：

- anchor/interpolation 仍由相应上下文表征负责；
- frame 16–24 的动态 Gaussian 由 frame-15 表征推进；
- MS3 对 Gaussian 的位置进行连续时间更新；
- `render_target_chunk_size` 支持按 target 分块渲染，降低峰值显存。

### 3.3 四类任务语义

`enable_task_semantic_head` 让每个 Gaussian/patch 预测四类 logits：

```text
0 background   1 ball   2 floor   3 obstacle
```

语义 logits 与 Gaussian 使用同一几何和 opacity 渲染，因此 RGB、深度、语义和 MS3 在像素上保持对齐。类别权重必须由正式 train split 的像素频率计算，采用 inverse-square-root weighting，并 cap 到 10。

### 3.4 Dense MS3 重建

MS3 使用 9 个通道：

```text
[vx,vy,vz, ax,ay,az, jx,jy,jz]
```

球区域监督仿真速度、重力 `[0,0,-9.81]` 和零 jerk；有效静态区域监督零运动。
运动状态随同 Gaussian 几何一起渲染到 target view，而不是单独预测一张无几何约束的运动图。

## 4. 仓库布局

```text
main_slarm.py          训练主程序（入口 + 全部 argparse + 训练循环）
engine_tools.py        build_model / evaluate 等共享库（被多处 import，留在根）

scripts/               面向使用的入口 py
  ├─ train_stream25_base.py    stereo/triview 预设启动器（exec main_slarm.py）
  ├─ eval_stream25_base.py     流式重建评估（acceptance 指标）
  ├─ render_stream25_base.py   生成指定帧长的重建视频（整段前向）
  └─ inference_stream.py       StreamSession 流式推理演示

run_sh/                一键启动 sh（内部 cd 到仓根后调用上面的入口）
  ├─ train.sh                  训练：单卡 python / 多卡 torchrun 自动
  ├─ eval.sh                   评估：输出目录按 config/ckpt 名自动生成
  ├─ train_stream25_base.sh    预设启动器的 sh 包装
  ├─ eval_stream25_base.sh     eval.sh 调用的底层评估 sh
  └─ render_stream25_base.sh   渲染重建视频入口

tools/                 辅助库与回归工具
  ├─ compare_dump.py / compare_report.py   三套代码前向逐比特无损对比
  ├─ check_render_vs_stream.py             整段前向 vs 流式逐帧 一致性检查
  └─ stream25_runtime.py / export_ply.py / ...

src/                   核心代码：models / dataset / utils / visualization
configs/               实验 YAML
```

## 5. 关键文件

| 文件 | 用途 |
|---|---|
| `configs/slarm_stream25_24cm_nopitch_window6.yaml` | 双目 base |
| `configs/slarm_stream25_24cm_triview_window6.yaml` | 三目 base |
| `configs/slarm_stream25_24cm_triview_stereo_subset_window6.yaml` | 三目（stereo 子集）base |
| `run_sh/train.sh` | 训练启动（单/多卡自动） |
| `run_sh/eval.sh` | 评估启动（输出路径自动） |
| `src/models/slarm.py` | 主模型（Gaussian / MS3 / terminal 外推） |
| `src/models/stream_session.py` | 六步因果流式会话 |
| `src/utils/stream25_losses.py` | 重建损失组装 |
| `src/utils/stream25_metrics.py` | 评估指标 / acceptance 门 |

## 6. 安装

```bash
# 创建 conda 环境
conda create -n SLARM python=3.10 -y
conda activate SLARM

# 更快的依赖求解（可选）
conda install mamba -n base -c conda-forge

# 可选：gsplat 因 g++/CUDA 编译失败时，在环境内装 CUDA 12.1
mamba install nvidia/label/cuda-12.1.1::cuda-toolkit -c nvidia/label/cuda-12.1.1
# export CUDA_HOME=$CONDA_PREFIX

# PyTorch (CUDA 12.1)
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121

# 可选：可微体素化
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.3.1+cu121.html

# gsplat（torch 2.3.1 + cuda 12.1 对应版本，用环境内 cuda 12.1 编译）
pip install git+https://github.com/nerfstudio-project/gsplat.git@937e29912570c372bed6747a5c9bf85fed877bae --no-build-isolation

# python 依赖
pip install -r requirements.txt
```

## 7. 训练

### 7.1 配置

| 字段 | 含义 |
|---|---|
| `dataset`, `data_root` | 数据集注册名和数据根目录 |
| `input_size` | `[H,W]`，当前 `[320,240]` |
| `patch_size` | ViT patch 边长，当前 8 |
| `num_max_cameras` | 双目为 2，三目为 3 |
| `num_context_timesteps` | 每个 episode 的流式观察数，当前 6 |
| `num_target_timesteps` | 每步监督 target 数，当前 7 |
| `context_stride` | 观察帧之间相差 3 个仿真帧 |
| `timespan` | frame 0 到 frame 24 的物理时间，0.8 秒 |
| `mode` | 必须是 `window_6` |
| `terminal_context_extrapolation` | frame 15 负责向 frame 24 外推 |
| `stream25_*_weight` | 各重建损失进入总损失的权重 |
| `stream25_ms3_*_scale` | 速度、加速度、jerk 的物理归一化尺度 |
| `stream25_semantic_class_weights` | 可选的四类语义权重，只能来自训练集统计 |
| `lr`, `stream25_trunk_lr` | head 与共享 trunk 的学习率 |
| `load_from` | 只加载模型权重的初始化 checkpoint |
| `train_annotation`, `eval_annotation` | full train/validation manifest |
| `num_iterations` | 双目 20k，三目 40k |
| `ckpt_every_n_iters` | checkpoint 保存间隔 |

### 7.2 启动

编辑 `run_sh/train.sh` 顶部的 `GPUS` 和 `CONFIG`，然后：

```bash
bash run_sh/train.sh
```

- `GPUS="0"`（单卡）自动用 `python`，`GPUS="4,5,6,7"`（多卡）自动用 `torchrun`；
- 命令行额外参数会透传给 `main_slarm.py`，例如：

```bash
bash run_sh/train.sh --num_iterations 30000
```

训练输出落在 `work_dirs/<project>/<exp_name>/`，由 config 里的 `exp_name` 决定。

> 另有预设启动器 `run_sh/train_stream25_base.sh stereo|triview`，内置 stereo/triview 两套 config 的校验后再 exec `main_slarm.py`，按需选用。

### 7.3 loss

[`src/utils/stream25_losses.py`](src/utils/stream25_losses.py) 隔离计算以下重建 loss（woLSeg 版已移除 LSeg 特征监督）：

| Loss                   |   权重 |
| ---------------------- | -----: |
| full RGB               | `1.00` |
| LPIPS                  | `0.05` |
| ball RGB               | `0.50` |
| full depth relative    | `1.00` |
| ball metric depth      | `0.02` |
| four-class semantic    | `1.00` |
| ball MS3               | `1.00` |
| static MS3             | `0.25` |
| opacity regularization | `0.10` |

球在 frame 16–24 某一目离屏时，只把该 frame-eye 的 ball-region loss/metric 记为 N/A；全图 loss 仍有效，也不会冻结该视角后续全部帧。

## 8. 评估（流式重建）

该命令是独立评估，不会继续训练。它按 `[0,3,6,9,12,15]` 真流式（StreamSession）输入，重建 frame 0–24，输出 RGB、depth、四类语义/ball IoU、MS3 速度/加速度/jerk、frame-24 位置，以及 anchor/interpolation/extrapolation 分组指标。

编辑 `run_sh/eval.sh` 顶部的 `CONFIG` 和 `CKPT`，然后：

```bash
bash run_sh/eval.sh
```

输出目录自动为 `work_dirs/slarm/stream25_eval/<config名>/<ckpt名>/`（切换 config 或 ckpt 不会互相覆盖），内含 `evaluation.json` 和 `evaluation.md`。

## 9. 推理：生成重建视频

对指定场景整段前向渲染 `[0,N)` 帧（`N` 可 >25 做外推）生成重建视频：

```bash
bash run_sh/render_stream25_base.sh \
  --config configs/slarm_stream25_24cm_triview_window6.yaml \
  --checkpoint ckpts/triview_stage_a.pth \
  --data_root data/SLARM_data \
  --scene_ids 0,1,2 \
  --num_frames 40 \
  --output_dir output/stream25_triview
```

| 参数 | 含义 |
|---|---|
| `--config` | 与 checkpoint 完全匹配的双目或三目 Stream25 配置 |
| `--checkpoint` | 已训练的重建 checkpoint |
| `--data_root` | 含 `scene_list/`、`annotations/` 和各模态数据的根目录 |
| `--scene_ids` | validation manifest 内的局部下标，不是 annotation 中的全局 scene 编号 |
| `--num_frames` | 从 frame 0 开始重建的总帧数，必须大于 0；默认 25 |
| `--output_dir` | 每个局部下标生成一个 `scene_XXXX.mp4` |
