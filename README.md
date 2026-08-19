# SLARM 流式重建与接球状态系统文档

## 1. 系统目标与完整数据流

系统接收六次同步三目观测：

```text
frame 0 → 3 → 6 → 9 → 12 → 15
```

每次观测包含：

```text
front_left + front_right + lower_front
RGB + 相机内外参 + 物理时间
```

端到端流程如下：

```text
三目稀疏观测
  → ViT/Aggregator 因果时空聚合
  → frame-15 terminal perception tokens [B,3,1200,1536]
  → 固定 CLIP("the tennis ball") 条件的四层 CatchStateReader
  → catch position_rig [B,3]
  → 外部动作模块
```

SLARM 同时保留完整重建能力：用六次观测重建 25 帧中的 RGB、metric depth、四类语义和 MS3，并从最后一次观测外推。

## 2. 冻结的三目与时序合同

### 2.1 相机

相机顺序不可改变：

| 顺序 |      |     名称      | rig/FLU offset（m）  | Pitch  |
| :--: | ---: | :-----------: | :------------------: | :----: |
|  0   |      | `front_left`  | `[0.00,+0.20,0.00]`  |  `0°`  |
|  1   |      | `front_right` | `[0.00,-0.20,0.00]`  |  `0°`  |
|  2   |      | `lower_front` | `[+0.30,0.00,-1.00]` | `+27°` |

`front_left` 是 canonical/reference camera。Pitch 只属于 `lower_front` 的相机外参，rig/FLU 坐标系本身没有旋转。rig 原点位于离地 1.5 m，因此`lower_front` 的世界高度为 0.5 m。

### 2.2 图像与 token

| 项目                 |              数值 |
| -------------------- | ----------------: |
| 输入高×宽            |         `320×240` |
| Patch size           |               `8` |
| 每目 patch           |      `40×30=1200` |
| Aggregator embed dim |             `768` |
| terminal token dim   |            `1536` |
| terminal token shape | `[B,3,1200,1536]` |

### 2.3 时间

- 原始仿真以 30 FPS 记录 30 帧，用于接触边界和数据完整性审计；
- SLARM 训练/评估使用半开区间 `[0,25)`，即 frame 0–24；
- 只有 `[0,3,6,9,12,15]` 进入因果上下文；
- 每个训练 step 采样 7 个 target：1 anchor、2 interpolation、4 extrapolation；
- validation/test 对 25 帧完整评估；
- frame 16–24 由 frame-15 terminal context 独占外推责任。

## 3. 相对原始 SLARM 的网络改动

### 3.1 真流式 `window_6`

[`src/models/stream_session.py`](src/models/stream_session.py) 增加了真正的六步因果会话：每次只接收 `[B,1,V,C,H,W]`，维护 Aggregator/CameraHead KV cache，并强制帧序为 `[0,3,6,9,12,15]`。

在 `terminal_context_extrapolation=True` 时，前五次调用只累计上下文和 Gaussian；第六次调用后才统一渲染全部 targets。

### 3.2 Terminal-context 外推

[`src/models/slarm.py`](src/models/slarm.py) 新增 `terminal_context_extrapolation`：

- anchor/interpolation 仍由相应上下文表征负责；
- frame 16–24 的动态 Gaussian 由 frame-15 表征推进；
- MS3 对 Gaussian 的位置进行连续时间更新；
- `render_target_chunk_size` 支持按 target 分块渲染，降低峰值显存。

### 3.3 四类任务语义

增加 `enable_task_semantic_head`。每个 Gaussian/patch 预测四类 logits：

```text
0 background
1 ball
2 floor
3 obstacle
```

语义 logits 与 Gaussian 使用同一几何和 opacity 渲染，因此 RGB、深度、语义和 MS3 在像素上保持对齐。类别权重必须由正式 train split 的像素频率计算，采用 inverse-square-root weighting，并 cap 到 10。

### 3.4 Dense MS3 重建

MS3 使用 9 个通道：

```text
[vx,vy,vz, ax,ay,az, jx,jy,jz]
```

球区域监督仿真速度、重力 `[0,0,-9.81]` 和零 jerk；有效静态区域监督零运动。
运动状态随同 Gaussian 几何一起渲染到 target view，而不是单独预测一张无几何约束的运动图。

### 3.5 Terminal perception tokens

`emit_terminal_perception_tokens=True` 时，SLARM 从归一化后的最后层 Aggregator patch tokens 中提取最后一次观测的三个视角，输出：

```python
predictions["latest_perception_tokens"]  # [B,3,1200,1536]
```

这些 token 虽然索引属于 frame 15，但已经通过 KV cache/attention 汇总前六次观测，不能理解成“只看 frame 15 的单帧特征”。

## 4. 新增的下游接球模块

### 4.1 固定 CLIP prompt

[`src/models/fixed_clip_prompt.py`](src/models/fixed_clip_prompt.py) 使用本地 OpenAI CLIP ViT-B/32 将固定文本：

```text
"the tennis ball"
```

编码为 512D、L2-normalized embedding。CLIP 和 embedding 都冻结并记录 SHA256。该文本只是固定找球先验，不代表当前系统支持任意语言选择物体。

### 4.2 CatchStateReader

[`src/models/catch_state_reader.py`](src/models/catch_state_reader.py) 的正式结构：

```text
visual tokens [B,3,1200,1536]
  → LayerNorm(1536)
  → Linear(1536,512)
  → 加三目 view embedding
  → flatten 为 [B,3600,512]

CLIP query [B,1,512]
  → 4 × (pre-norm cross-attention + FFN)
  → fused query [B,512]
  └── position head: 512 → 256 → 2
```

每层都有独立 Q/K/V；第 2–4 层的 Q 来自前一层融合后的 query。

可选的 `[9,12,15]` 单变量实验不会直接拼原始 RGB。三帧分别经过同一个
冻结 SLARM，取归一化后的 post-Aggregator perception tokens：

```text
[B,3 frames,3 views,1200 patches,1536]
  → 加 frame/view embedding
  → time-major × view-major × patch-major flatten
  → [B,10800,512]
  → 原四层 Reader
```

frame 15 就是 terminal tokens，不会再追加一次。第一层 attention 和 ball
patch target 相应扩展为 10800，并让三个时刻各占 `1/3` 监督质量。

输出定义：

```python
{
    "catch_xy_rig": ...,              # [B,2]
    "catch_position_rig": ...,        # [B,3]，z_rig 固定为 -0.5 m
    "first_layer_attention": ...,     # [B,8,3600]
}
```

目标是球在 frame 15 之后首次下降穿过 `z_world=1.0 m` 时的连续状态，不是 frame-15 当前状态。

### 4.3 Attention target

[`src/utils/catch_attention.py`](src/utils/catch_attention.py) 将 frame-15 三目 ball mask 用 `8×8` average pooling 变成每目 `40×30` patch distribution。
每目独立归一化后分配 `1/3` 总质量，再按 view-major、row-major flatten 为 3600 维 target。

### 4.4 Frozen streaming wrapper

[`src/models/streaming_catch_model.py`](src/models/streaming_catch_model.py) 组合：

```text
frozen SLARM + frozen CLIP prompt + trainable/frozen Reader
```

它提供 `reset()` 和 `observe()`，按约定的帧序和相机顺序接收数据，并保证 `.train()` 不会把 SLARM 或 CLIP 切回训练状态。Reader 训练时 optimizer 只包含 Reader 参数。当前 POC 假定调用方提供匹配的配置、checkpoint 和 prompt，不额外执行兼容性门控。

## 5. 关键文件

| 文件 | 用途 |
|---|---|
| `data_gen/configs/ball_catch_24cm_triview_reader_extra_v1.yaml` | 三目 Isaac 采集、物理参数和输出路径 |
| `tools/prepare_reader_ablation_data.py` | 转换新增 raw 场景、执行硬审计并发布训练清单 |
| `configs/slarm_stream25_24cm_nopitch_window6.yaml` | 双目 full base |
| `configs/slarm_stream25_24cm_triview_window6.yaml` | 三目 full base |
| `run_sh/train_stream25_base.sh` | 选择 `stereo` 或 `triview` 并启动 base |
| `scripts/run_stream25_inference.py` | 按六帧契约生成用户指定时长的重建视频 |
| `run_sh/run_streaming_reconstruction.sh` | 重建视频的一键入口 |
| `tools/build_catch_prompt_artifact.py` | 从本地 CLIP 权重生成固定 prompt artifact |
| `tools/build_catch_state_cache.py` | 用三目 base 生成 train/validation token cache |
| `tools/build_temporal_catch_state_cache.py` | 生成 frame 9/12/15 perception-token cache |
| `scripts/run_reader_temporal_9_12_15_pipeline.sh` | 双卡并行建 temporal cache，再训练 30k Reader |
| `configs/slarm_catch_state_reader_full.yaml` | Reader 30k full 配置 |
| `scripts/train_catch_state_reader.sh` | 两卡启动 Reader |
| `tools/eval_catch_state_reader.py` | 在固定 token cache 上独立重评 Reader checkpoint |
| `scripts/eval_catch_state_reader.sh` | Reader 独立评估的一键入口 |
| `src/models/streaming_catch_model.py` | 在线六次 observe 后输出 CatchState |
| `tools/run_streaming_catch_inference.py` | 加载正式产物并输出接球位置 JSON |
| `scripts/run_streaming_catch_inference.sh` | 六观测接球位置推理入口 |

## 6. Install

```bash
# create conda environment
conda create -n SLARM python=3.10 -y
conda activate SLARM

# Install mamba for faster installation
conda install mamba -n base -c conda-forge

# Optional: When gsplat compilation fails due to g++ version or CUDA toolkit issues.
# Install CUDA 12.1 in conda environment
mamba install nvidia/label/cuda-12.1.1::cuda-toolkit -c nvidia/label/cuda-12.1.1
# export CUDA_HOME=$CONDA_PREFIX

# Install PyTorch with CUDA 12.1
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121

# Optional: Differentiable Voxelization
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.3.1+cu121.html

# Install gsplat, specific versions supported by torch 2.3.1 and cuda 12.1 (用conda环境下的 cuda 12.1 编译)
pip install git+https://github.com/nerfstudio-project/gsplat.git@937e29912570c372bed6747a5c9bf85fed877bae --no-build-isolation

# Install python dependencies
pip install -r requirements.txt

# Install CLIP for semantic alignment
pip install git+https://github.com/openai/CLIP.git

```

## 7. 仿真数据采集、转换与审计

### 7.1 采集链路和环境

完整链路不是直接生成训练 `.pt`，而是：

```text
Isaac Sim 渲染 30 帧原始场景
  → 完成门控写入 .complete
  → convert_to_slarm 转换 RGB/depth/semantic/相机/轨迹
  → 发布 annotation 和 scene list
  → 审计三目、25 帧和六个 context 的可见性
  → SLARM Dataset 按 step 采样 7 个监督 target
```

采集需要两个 Python 入口：

- `ISAAC_PYTHON`：Isaac Sim 自带的 `python.sh`，只用于启动 Replicator；
- 当前终端的 `python`：已激活的 SLARM Conda 环境，用于转换、审计和训练。

精简后的 POC 保存生成配置和转换入口；Isaac 生成器位于完整 SLARM 源码树。先设置：

```bash
export POC_ROOT=/path/to/SLARM4POC
export GENERATOR_ROOT=/path/to/full-SLARM-source
export ISAAC_PYTHON=/path/to/isaac-sim/python.sh
cd "$POC_ROOT"
```

`GENERATOR_ROOT/data_gen/` 中必须包含 `isaac_generation_supervisor.py`、`isaac_replicator.py` 和 `convert_to_slarm.py`。不要用普通 Conda Python 启动`isaac_replicator`，否则无法导入 `isaacsim`。

### 7.2 冻结采集配置

| 配置段 | 作用 |
|---|---|
| `scene` | 场景数、30 个物理帧、基础随机种子和默认重试次数 |
| `camera` | 960×720 渲染分辨率、320×240 SLARM 分辨率、FOV、三目外参和 30 FPS |
| `launcher` | 4–6 m 发射距离、目标落点范围、飞行时间和 context 可见性门控 |
| `ball`、`physics` | 24 cm 直径球、质量、材质、重力和 240 Hz 仿真步长 |
| `domain_randomization` | 墙面、地面、光照和障碍物随机化 |
| `semantics` | `{background, ball, floor, obstacle}` 四类映射 |
| `output` | raw 输出、SLARM 数据根目录、scene 前缀和数据集名 |

换机器后先修改 `output.raw_data_dir` 和 `output.slarm_data_dir`。

### 7.3 单场景冒烟采集

全量运行前先生成一个未占用的 scene。`start` 包含、`end` 不包含：

```bash
cd "$GENERATOR_ROOT"
CUDA_VISIBLE_DEVICES=0 "$ISAAC_PYTHON" \
  -m data_gen.isaac_generation_supervisor \
  --config "$POC_ROOT/data_gen/configs/*.yaml" \
  --start 2000 --end 2001 \
  --attempt-start 0 --max-attempts 10 \
  --scenes-per-process 1
```

只有出现以下文件才表示该场景通过完整门控：

```text
raw_data/.../scene_2000/.complete
raw_data/.../scene_2000/generation_record.json
raw_data/.../scene_2000/ball_trajectory.json
raw_data/.../scene_2000/rgb_<camera>/00000.png ... 00029.png
raw_data/.../scene_2000/depth_<camera>/00000.png ... 00029.png
raw_data/.../scene_2000/seg_<camera>/00000.png ... 00029.png
```

`.complete` 是唯一完成凭证。目录存在但没有该文件时，可能是被拒绝的轨迹或中断的半成品，不能参与转换和训练。

### 7.4 转换、发布清单和硬审计

确认 `.complete` 数量为 2000 后，回到 POC，并使用当前激活的 SLARM 环境：

```bash
cd "$POC_ROOT"
python tools/prepare_reader_ablation_data.py \
  --config data_gen/configs/*.yaml \
  --generator-root "$GENERATOR_ROOT" \
  --workers 8
```

该命令会跳过已经转换的场景，其余场景通过 `convert_to_slarm.py` 写入：

```text
data/SLARM_data/
├── annotations/ball_catch_24cm_triview/training/scene_XXXX.json
├── datasets/ball_catch_24cm_triview/training/scene_XXXX/
│   ├── front_left/vis/{color,depth,semantic}/
│   ├── front_right/vis/{color,depth,semantic}/
│   ├── lower_front/vis/{color,depth,semantic}/
│   └── ball_gt/trajectory.json
└── scene_list/
```

原始 30 帧用于接触和轨迹完整性检查；转换后的训练 annotation 固定为 25 帧`[0,25)`。工具只有在以下条件全部满足时才发布 scene list：

- scene ID 与 annotation 一致；
- 相机顺序严格为 `front_left, front_right, lower_front`；
- `num_timesteps == 25`；
- 球在 `[0,3,6,9,12,15]` 六次观测中三目均可见；
- 2000 个新增场景全部存在并通过审计。

### 7.5 常见问题

| 现象 | 原因和处理 |
|---|---|
| `No module named isaacsim` | 使用了普通 Python；改用 `ISAAC_PYTHON` |
| `No module named data_gen` | 当前目录或 `GENERATOR_ROOT` 错误；从完整源码根目录启动 |
| 有场景目录但无 `.complete` | 生成中断或门控拒绝；用新的 attempt 区间续采 |
| Isaac 运行越久越慢或 OOM | 降低 `--scenes-per-process`，由 Supervisor 更频繁重启 |
| 转换时报 annotation/camera/visibility 错误 | 不要手工补文件；回到 raw 场景重新生成并重新审计 |
| 训练找不到场景 | 检查配置中的 `output.slarm_data_dir` 与训练的 `data_root` 是否指向同一目录 |

## 8. SLARM base 训练

### 8.1 配置

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

### 8.2 启动

```bash
bash run_sh/train_stream25_base.sh triview
```

`--resume_from`

若要从原始 Waymo SLARM 权重重新初始化当前三目 Stream25，先运行：

```bash
python tools/migrate_original_slarm_to_triview.py \
  --source /path/to/original/ckpt_003999.pth \
  --target_config configs/slarm_stream25_24cm_triview_from_original_window6.yaml \
  --output ckpts/slarm_original_003999_to_triview_init.pth
```

迁移会逐值继承共享权重，重新生成 320×240 Plücker 网格，以原始前视
camera affine token 初始化三个当前相机，并按固定 seed 新建四类语义头。
输出不含 optimizer、step 或 scaler，必须作为 `load_from` 从 step 0 训练：

```bash
bash run_sh/train_stream25_base.sh triview \
  --config configs/slarm_stream25_24cm_triview_from_original_window6.yaml
```

### 8.3 loss

[`src/utils/stream25_losses.py`](src/utils/stream25_losses.py) 隔离计算十项 loss：

| Loss                   |   权重 |
| ---------------------- | -----: |
| full RGB               | `1.00` |
| LPIPS                  | `0.05` |
| ball RGB               | `0.50` |
| full depth relative    | `1.00` |
| ball metric depth      | `0.02` |
| four-class semantic    | `1.00` |
| LSeg feature           | `1.00` |
| ball MS3               | `1.00` |
| static MS3             | `0.25` |
| opacity regularization | `0.10` |

球在 frame 16–24 某一目离屏时，只把该 frame-eye 的 ball-region loss/metric记为 N/A；全图 loss 仍有效，也不会冻结该视角后续全部帧。

## 9. CatchStateReader 训练

### 9.1 生成固定 prompt

本步骤读取本地 OpenAI CLIP ViT-B/32，将 token IDs、归一化 512 维 embedding 和来源哈希写入一个 `.pt` 文件。训练和推理都读同一个 artifact，不会在运行时下载或重新编码文本。

```bash
python tools/build_catch_prompt_artifact.py \
  --output ckpts/tennis_ball_clip_vit_b32.pt
```

`--clip-path` 指定本地 CLIP 文件；也可设置 `SLARM_CLIP_CHECKPOINT`。未指定时按`$XDG_CACHE_HOME/clip/ViT-B-32.pt`（若`XDG_CACHE_HOME` 未设置则`~/.cache/clip/ViT-B-32.pt`）查找；`--force` 才允许原子替换已有 artifact。

### 9.2 建立正式 token cache

分别建立 train 和 validation cache，保存每个场景第15帧聚合后的 token 进行复用，避免每次训练和测试都要重跑一次SLARM主网络。

`--checkpoint` 必须显式指定已选定的三目SLARM checkpoint；其路径和 SHA256 会写入 manifest。

`--generation-config` 用来读取每个场景实际记录的物理帧数，从而判断 1 m 下降交点是否发生在有效、未接触区间。

```bash
python tools/build_catch_state_cache.py --split train \
  --checkpoint <triview-stage-a.pth> --generation-config <generation.yaml> \
  --output-dir work_dirs/slarm/catch_state_reader/cache_train_formal

python tools/build_catch_state_cache.py --split validation \
  --checkpoint <triview-stage-a.pth> --generation-config <generation.yaml> \
  --output-dir work_dirs/slarm/catch_state_reader/cache_validation_formal
```

重要参数：

| 参数 | 含义 |
|---|---|
| `--config` | 三目 Stream25 YAML，默认正式三目配置 |
| `--prompt-artifact` | 上一步生成的固定文本 artifact |
| `--device` | cache forward 使用的设备 |
| `--shard-size` | 每个 `.pt` shard 的场景数，默认 16 |
| `--code-sha256` | dirty worktree 时显式记录代码身份 |
| `--resume` | 校验 build identity 后继续缺失 shard |

### 9.3 启动

```bash
bash scripts/train_catch_state_reader.sh
```

要运行 `[9,12,15]` perception-token 实验，使用：

```bash
bash scripts/run_reader_temporal_9_12_15_pipeline.sh
```

脚本将 3000 个训练场景等分到两张 GPU 并行建 cache，再在 GPU 0 建立
200-scene validation cache。随后它把四场景 `.pt` shard 转成逐场景可直接
切片的连续 mmap 文件，最后双卡训练 30000 步。该转换不重跑 SLARM，且
源 cache 保留；转换意外中断时重新运行脚本会从已完成 shard 继续。单场景
temporal token 约 33.2 MB，正式 train/validation packed cache 分别约
100 GB/6.6 GB，转换期间还需为源 cache 预留同等空间。

已有 temporal cache 也可单独转换：

```bash
python tools/pack_temporal_catch_state_cache.py \
  --source work_dirs/slarm/catch_state_reader/all3000/cache_temporal_9_12_15_validation \
  --output work_dirs/slarm/catch_state_reader/all3000/cache_temporal_9_12_15_validation_packed
```

训练配置必须指向 packed 目录。训练时两个 rank 共享操作系统 page cache，
不再在随机采样的每次 shard miss 上重复计算 SHA256 和反序列化整块 `.pt`。

配置字段：

| 字段 | 含义 |
|---|---|
| `world_size`, `batch_size_per_rank` | 2 卡，每卡 batch 8，全局 batch 16 |
| `max_steps` | 30000 |
| `learning_rate`, `min_learning_rate` | cosine 的起止学习率 |
| `warmup_steps` | 线性 warmup 步数 |
| `weight_decay` | AdamW 权重衰减 |
| `gradient_clip_norm` | 全局梯度裁剪上限 |
| `validation_interval` | 每 5000 步在完整 validation cache 上评估并保存 |
| `precision` | CUDA 训练为 BF16 |
| `dropout` | 四层 reader 的 attention/FFN dropout |
| `perception_frames` | Reader token 时刻；baseline 为 `[15]`，本实验为 `[9,12,15]` |
| `additional_train_cache_dirs` | 双 GPU 建出的其他训练 cache 分区，不复制 shard |

### 9.4 loss

Reader 只监督接球面 `x/y` 和第一层 attention，不再预测速度。原始轨迹速度仍作为 SLARM MS3 与解析接球交点的数据元信息保留。

```text
L_xy = mean(SmoothL1((pred_xy - target_xy) / 0.1 m, 0, beta=1))
L = L_xy
  + 0.1 × attention_cross_entropy
```

旧 Reader checkpoint 中的 `velocity_head.*` 会在加载时显式丢弃，其余权重可继续作为位置模型初始化。

## 10. 两阶段评估

### 10.1 Stage A：SLARM 重建评估

该命令是独立评估，不会继续训练。它按 `[0,3,6,9,12,15]` 真流式输入，
重建 frame 0–24，并输出 RGB、depth、四类语义/ball IoU、MS3
速度/加速度/jerk、frame-24 位置，以及 anchor/interpolation/extrapolation
分组指标：

```bash
mkdir -p work_dirs/slarm/stream25_eval/ckpt_034999

bash run_sh/eval_stream25_base.sh \
  --config configs/slarm_stream25_24cm_triview_window6.yaml \
  --checkpoint ckpts/ckpt_034999.pth \
  --split validation \
  --output work_dirs/slarm/stream25_eval/ckpt_034999/evaluation.json \
  --output-markdown work_dirs/slarm/stream25_eval/ckpt_034999/evaluation.md
```

更换 checkpoint 时，应同时更换输出目录，避免覆盖不同权重的结果。

### 10.2 Stage B：CatchStateReader 评估

Reader 在每个 `validation_interval` 对完整 validation cache 自动评估，结果已保存在相应 checkpoint 的 `validation_summary` 中。
正式独立复评使用下面的命令：

```bash
mkdir -p work_dirs/slarm/catch_state_reader/all3000/temporal_all6_full_30k/evaluation

bash scripts/eval_catch_state_reader.sh \
  --config configs/slarm_catch_state_reader_all3000_temporal_all6_full.yaml \
  --checkpoint work_dirs/slarm/catch_state_reader/all3000/temporal_all6_full_30k/checkpoint_step_005000.pt \
  --device cuda:0 \
  --batch_size 1 \
  --output work_dirs/slarm/catch_state_reader/all3000/temporal_all6_full_30k/evaluation/step_005000.json \
  --output_markdown work_dirs/slarm/catch_state_reader/all3000/temporal_all6_full_30k/evaluation/step_005000.md
```

默认使用配置中的 `validation_cache_dir` 和 `validation_cache_split`。需要评估其他兼容 cache 时可显式添加 `--cache_dir <目录> --split validation`。`--batch_size 1` 与训练时自动 validation 的逐场景执行方式一致，也最节省显存。


## 11. 推理

### 11.1 生成指定帧数的重建视频

```bash
bash run_sh/run_streaming_reconstruction.sh \
  --config configs/slarm_stream25_24cm_triview_window6.yaml \
  --checkpoint ckpts/triview_stage_a.pth \
  --data_root data/SLARM_data \
  --scene_ids 0,1,2 \
  --num_frames 40 \
  --output_dir output/stream25_triview \
  --lseg_model_scratch_path ckpts/lseg_model_scratch.pth \
  --lseg_model_pretrained_path ckpts/lseg_model_pretrained_replace_1x1conv_with_linear.pth
```

参数含义：

| 参数 | 含义 |
|---|---|
| `--config` | 与 checkpoint 完全匹配的双目或三目 Stream25 配置 |
| `--checkpoint` | 已训练的 Stage-A checkpoint，不是 Reader checkpoint |
| `--data_root` | 含 `scene_list/`、`annotations/` 和各模态数据的根目录 |
| `--scene_ids` | validation manifest 内的局部下标，不是 annotation 中的全局 scene 编号 |
| `--num_frames` | 从 frame 0 开始重建的总帧数，必须大于 0；默认 25 |
| `--output_dir` | 每个局部下标生成一个 `scene_XXXX.mp4` |
| `--lseg_*_path` | 配置启用 `online_feat` 时所需的本地 LSeg 权重 |

### 11.2 推理最终接球位置

`scripts/run_streaming_catch_inference.sh` 会加载三目 Stage-A、固定 prompt 和 Reader checkpoint，从指定 split 读取一个 episode，逐次输入六个观测，并在 frame 15 输出 JSON：

```bash
  bash scripts/run_streaming_catch_inference.sh \
  --config configs/slarm_stream25_24cm_triview_window6.yaml \
  --data_root data/SLARM_data \
  --slarm_ckpt ckpts/triview_stage_a.pth \
  --reader_ckpt ckpts/checkpoint_step_XXXXXX.pt \
  --prompt_artifact ckpts/tennis_ball_clip_vit_b32.pt \
  --split validation \
  --scene_index 0 \
  --output_json output/catch_state/scene_1000.json
```

| 参数 | 含义 |
|---|---|
| `--slarm_ckpt` | 与三目配置配套使用的冻结 SLARM checkpoint |
| `--reader_ckpt` | `train_catch_state_reader.py` 保存的 Reader checkpoint |
| `--prompt_artifact` | 固定 `"the tennis ball"` CLIP artifact；有正式默认路径 |
| `--split` | `train` 或 `validation`，默认 validation |
| `--scene_index` | split manifest 内的局部下标，默认 0 |
| `--device` | CUDA 设备，默认 `cuda:0` |
| `--output_json` | 可选；原子写入结果，同时仍在终端打印 |
| `--dry_run` | 只检查路径并打印解析后的请求，不加载模型或占用 GPU |

输出含义：

- `position_rig=[x,y,z]`：球下降穿过世界坐标 `z_world=1 m` 接球面时的位置，单位米；
  当前固定几何下 `z=-0.5 m`，Reader 实际学习 `x/y`后与`z=1`拼接。
- `first_layer_attention=[B,8,3600]`：第一层八个 head 对三目 patch 的注意力，仅用于诊断注意力是否在球上，不应直接作为动作指令。
