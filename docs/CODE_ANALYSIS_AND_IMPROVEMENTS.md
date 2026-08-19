# SLARM Tennis 代码结构分析与改进建议

> 本文档系统梳理 `slarm_tennis` 代码库的整体结构、数据生成链路、模型前向传播、损失计算、评估指标,并给出面向 **Stage1(流式重建 base)未来轨迹预测准确度提升** 的可改进点清单。
>
> ⚠️ 形状说明:代码中若干 head 的默认值(如 `embed_dim=1024`、DINOv2 patch14)与网球任务实际配置(`patch_size=8`、terminal token `[B,3,1200,1536]`)不同。本文 **以 `CONTEXT.md` / `README.md` 的任务配置为准**,标 `*` 处随 config 变化。

---

## 目录

1. [项目目标与端到端数据流](#1-项目目标与端到端数据流)
2. [目录结构地图](#2-目录结构地图)
3. [核心模块总览](#3-核心模块总览)
4. [数据生成链路](#4-数据生成链路)
5. [模型 Forward](#5-模型-forward)
6. [Loss 计算](#6-loss-计算)
7. [评估指标(Metric)与验收表](#7-评估指标metric与验收表)
8. [Stage1 未来轨迹预测改进建议](#8-stage1-未来轨迹预测改进建议)
9. [关键文件索引](#9-关键文件索引)

---

## 1. 项目目标与端到端数据流

### 1.1 目标(一句话)

流式动态场景重建 + 接球状态预测系统。给定网球飞行场景的 **6 次稀疏三目观测**(frame `0,3,6,9,12,15`),模型需要:

1. **重建** 25 帧的 RGB、metric depth、四类语义、运动场(MS3);
2. **预测** 网球在 frame 15 之后首次下降穿过 `z_world=1.0m` 平面时的**接球位置与速度**,供外部动作模块使用。

技术底座:**3D Gaussian Splatting(高斯泼溅)+ ViT/Aggregator 因果时空聚合 + MS3 连续时间运动模型**。

### 1.2 端到端数据流

```text
Isaac Sim 仿真采集(30帧原始)
   → convert_to_slarm 转换 + 硬审计 → 25帧训练数据
   → SLARM base 训练(重建 RGB/深度/语义/MS3)         ← Stage 1
   → 冻结 base,建 terminal token cache
   → 训练 CatchStateReader(接球状态回归)               ← Stage 2
   → 流式推理输出接球位置 JSON

运行时:
6次三目观测 [B,1,3,C,H,W]
  → Aggregator 因果时空聚合(维护 KV cache)
  → frame-15 terminal perception tokens [B,3,1200,1536]*
  → 固定 CLIP("the tennis ball") 条件的 4 层 CatchStateReader
  → catch_position_rig [B,3](z_rig 固定 -0.5m)
```

### 1.3 冻结的三目与时序契约

| 顺序 | 名称 | rig/FLU offset (m) | Pitch |
|:--:|:--:|:--:|:--:|
| 0 | `front_left`(canonical) | `[0.00,+0.20,0.00]` | 0° |
| 1 | `front_right` | `[0.00,-0.20,0.00]` | 0° |
| 2 | `lower_front` | `[+0.30,0.00,-1.00]` | +27° |

- 输入 320×240,patch_size=8,每目 40×30=1200 patch;terminal token dim 1536。
- 仿真 30 FPS 记录 30 帧;训练/评估用半开区间 `[0,25)`。
- 只有 `[0,3,6,9,12,15]` 进入因果上下文;每训练 step 采样 7 个 target(1 anchor + 2 interpolation + 4 extrapolation)。
- frame 16–24 由 frame-15 terminal context 独占外推。

---

## 2. 目录结构地图

| 目录 | 职责 |
|---|---|
| **根 `*.py`** | 训练/推理入口:`main_slarm.py`、`main_storm.py`、`inference.py`、`inference_stream.py`、`preprocess.py`、`engine_tools.py` |
| **`src/models/`** | 核心网络:SLARM、STORM、CatchStateReader、Aggregator、各类 head |
| **`src/dataset/`** | 数据加载:`datasets.py`、`stream25.py`、`catch_state_cache.py`、`data_utils.py`、`constants.py` |
| **`src/utils/`** | 损失、指标、渲染器、几何投影、分布式工具 |
| **`src/visualization/`** | `video_maker.py` 等可视化 |
| **`data_gen/`** | Isaac Sim 数据生成、物理、转换、审计 |
| **`preproc/`** | Waymo/ArgoVerse/nuScenes 预处理(继承自原始 SLARM) |
| **`tools/`** | 各阶段 CLI:建 cache、训练 reader、评估、流式推理 |
| **`scripts/`** | 一键 `.sh` 流水线 + 训练 launcher |
| **`configs/`** | 实验 YAML(stereo/triview base、reader) |
| **`third_party/`** | 外部代码:`depth_anything_v2`、`lang_seg`(LSeg) |

---

## 3. 核心模块总览

### 3.1 模型层 `src/models/`

- **`slarm.py`(2266 行)** — 主重建模型 `class SLARM(nn.Module, PyTorchModelHubMixin)` `slarm.py:79`
  - 子模块:`aggregator`、`gs_predictor`、`motion_predictor`、`task_semantic_predictor`、`renderer`
  - 三大新特性:`terminal_context_extrapolation`(frame15 独占外推)、`enable_task_semantic_head`(四类语义)、`emit_terminal_perception_tokens`(输出 `[B,3,1200,1536]*`)
- **`storm.py`(1208 行)** — 轻量 ViT 替代模型 `class STORM(ViT)` `storm.py:35`,注册表 `STORM_models`
- **`catch_state_reader.py`** — `class CatchStateReader(nn.Module)` `:98`,4 层 pre-norm cross-attention + FFN,CLIP query 融合
- 其他:`fixed_clip_prompt.py`、`stream_session.py`、`streaming_catch_model.py`、`temporal_ownership.py`

### 3.2 组件 `src/models/components/`

- `aggregator/aggregator.py` `class Aggregator :29` — 两级注意力(帧内 self-attn + 全局 self-attn)、Plücker/时间编码、KV cache
- `heads/` — `CameraHead`、`DPTHead`、`ScaleHead`、`TrackHead`
- `utils/geometry.py` — 几何工具(反投影、四元数、法向/尺度)

### 3.3 数据 `src/dataset/`

- `PerceptualModelDataset` `datasets.py:122`(基础加载)
- `Stream25Dataset(PerceptualModelDataset)` `:870`(Stream25 规格)
- `Stream25TargetScheduler` `stream25.py:171`(7-target 采样)
- `catch_state_cache.py`(`CacheRecord`,Reader 训练用)

### 3.4 损失与渲染 `src/utils/`

- `losses.py` — `compute_loss():889`(base SLARM 通用损失)
- `stream25_losses.py` — `compute_stream25_loss():235`(Stream25 十项损失)
- `stream25_metrics.py` — 评估指标 + `ACCEPTANCE_TABLE`
- 渲染:`rasterizer_1112.py`、`projection_three_dims_gaussian_fused_1112.py`(NPU),GPU 走 `gsplat`
- `misc.py` — `load_model()`、`stream25_checkpoint_contract()`

---

## 4. 数据生成链路

整条链路是 **采集 → 物理 → 渲染 → 硬审计 → 转换 → 发布** 六段,每段都有"硬门控",不通过不进入下一步。

### 4.1 采集监督 `data_gen/isaac_generation_supervisor.py`

- `main()` `:147` 监督循环:每批取最多 `--scenes-per-process` 个未完成场景 → 启动 Isaac worker → 检查结果。
- `_partition_missing_sentinels()` 区分两类失败:
  - **确定性拒绝**(`accepted=False` 且 `config_hash` 匹配 且 attempt 达上限)→ 场景固有问题,跳过不阻塞。
  - **基础设施错误**(超时/OOM/崩溃)→ 返回码非 {0,1},整批失败重来。
- `.complete` 文件是**唯一完成凭证**。

### 4.2 物理仿真 + 三模态渲染 `data_gen/isaac_replicator.py`(1316 行)

- `build_scene_runtime()` `:75` 冻结房间/相机/发射参数;`generate_scene()` `:234` 主循环。
- **渲染循环** `:305`:跑 `30 + DISCARD_LEADING(3)` 步,**前 3 步丢弃**(RTX/相机注释器相位未稳),保留重编号为 frame 0–29。每步 `world.step()` = 1/240s 物理。
- 每帧每相机导出:RGB(PNG)、Depth(mm uint16 PNG)、Semantic(4 类任务 id)、相机位姿(4×4)。
- **接触检测**:`BallContactReporter` 订阅 PhysX 接触事件。
- **硬完成门** `validate_scene_completion()` `:775`:文件集完整、形状一致、语义类合法、**球在六帧三目均可见**、必须出现 floor、fps=30±1e-9、相机外参每帧偏差 <1mm、接球 crossing 期间无接触——全过才写 `.complete`。

### 4.3 球物理 `data_gen/ball_physics.py`

- `randomize_launch_params()` 弹道解:`v₀ = (target − launch − ½·g·T²) / T`。
- `randomize_until_visible()` 循环重采样(≤50 次),直到轨迹对三目在 6 个 context 帧 **100% 可见**,且不出画面顶部。
- 重力 `g=[0,0,-9.81]`。

### 4.4 场景构建 `data_gen/scene_builder.py`

- `build_room()` 造地板/天花/4 墙 + 随机障碍物;**中央走廊 y∈[-1,1] 保持无障碍**(球沿 y≈0 飞)。
- 语义词表(Stream25 四类):`background/wall/ceiling→0, ball→1, floor→2, obstacle/object→3`。

### 4.5 轨迹与接球状态 `data_gen/trajectory_metadata.py`

- `FirstContactRecorder` / `PhysicsContactTraceRecorder` 记录首个接触帧与全部接触事件。
- `world_to_rig_state()` 世界系→rig/FLU 系:`pos_rig=(rig_to_world⁻¹·[pos,1])`,`vel_rig=Rᵀ·vel`。这是"一米接球状态"的坐标基础。

### 4.6 转换 + 审计 `data_gen/convert_to_slarm.py`(837 行)

`convert_scene()` `:188` 三阶段:
1. **预检** `_preflight_strict_scene()` `:389`:立体基线 0.40±1mm、rig 中点偏差 <1mm、六帧可见性。
2. **帧转换**:30 帧→25 帧,720×960→240×320(RGB CUBIC→jpg,Depth mm→m NEAREST→tif,Semantic NEAREST→png)。
3. **注释生成** `:327`:写 `scene_XXXX.json`(归一化内参、25 组 4×4 外参、rig 系球轨迹、`ball_visible_frames_by_camera`、`first_contact_frame`、`source_provenance` SHA256)。

产出目录:
```text
datasets/<dataset>/training/scene_0000/{front_left,front_right,lower_front}/vis/{color,depth,semantic}/
annotations/<dataset>/training/scene_0000.json
scene_list/<dataset>_{train,validation,final_test}.txt
```

### 4.7 切分 + 发布 `make_ball_state_splits.py` / `tools/prepare_reader_ablation_data.py`

- 切分是**确定性索引区间**(tri-view:train `[0,1000)`、val `[1000,1200)`、test `[1200,1400)`)。
- 发布工具用 `ProcessPoolExecutor` 并行转换,`_audit_annotation()` 逐场景校验(scene_id/相机序/`num_timesteps==25`/六帧可见性),全过才原子写 scene list。

### 4.8 数据生成端到端框图

```text
supervisor(批调度)
  → isaac_replicator(物理+渲染+硬门→.complete)
  → convert_to_slarm(30→25帧+审计+annotation)
  → prepare_reader_ablation_data(批审计+发布清单)
  → Stream25Dataset 按 step 采样 7 个 target 监督
```

---

## 5. 模型 Forward

主线:**Aggregator 时空聚合 → GS/Motion/Feat 头 → MS3 连续时间推进 → gsplat 光栅化**。入口 `SLARM.forward()` `slarm.py:1582`。

### 5.1 输入

```text
context_image       [B,T,V,C,H,W] = [1,6,3,3,320,240]  (6帧×3目 RGB)
context_camtoworlds [B,T,V,4,4]   context_intrinsics [B,T,V,3,3]
context_time        [B,T,3]
target_camtoworlds  [B,20,3,4,4]  target_time [B,20,3]   (20 个待渲染 target)
```

### 5.2 Aggregator 两级注意力 `components/aggregator/aggregator.py:204`

- **Patch Embed**:图像 → DINOv2 patch tokens,加 **Plücker 射线嵌入** + **时间嵌入**。
- **Token 组装**:`[camera(1) + register(4) + motion(4) + sky(1) + patch(1200*)]` 拼成序列。
- **两级注意力循环(24 层)**:
  - **Frame attention**(帧内):`[B*T*V, P, C]` 每帧独立 self-attn;
  - **Global attention**(跨帧跨视):`[B, T*V*P, C]` 全局 self-attn,**支持 KV cache**(流式时 seq_len 递增)。
- 输出各中间层 `output_list`,末层 tokens 供各 head。

### 5.3 各预测头

| 头 | 位置 | 输出(形状) |
|---|---|---|
| **GS 头** `forward_gs_predictor` | `:877` | means/scales/quats/colors/opacities/depths `[B,T,V,H,W,·]`;means = `origins + directions·depth` |
| **Motion 头** `forward_motion_predictor` | `:822` | `forward_ms3 [B,T,V,H,W,9]` = (v₀,a₀,j₀);motion token 与像素特征点积→softmax 权重→加权基向量;`decode_flow` Sigmoid 激活 |
| **Feat 头** | `:1730` | `pred_feat [...,64]`(LSeg 特征) |
| **Task Semantic 头** | 可选 | `[...,4]` 四类 logits,与高斯同几何渲染 |
| **Camera / Depth 头** | `:1655/:1673` | 位姿编码、context 深度 |

### 5.4 渲染 `forward_renderer` `:1059`(核心创新)

- **展平+扩展**:context 高斯展平 `[B, T*V*H*W, ·]`,repeat 到 20 个 target → batched。
- **MS3 连续时间位置更新**(Taylor 展开):
  ```text
  Δpos(t) = Σ ms3[i]·Δt^(i+1)/(i+1)!   →  means += Δpos
  ```
  用速度/加速度/jerk 把高斯"推进"到任意 target 时刻(可外推超过 frame 24)。
- **Terminal Context Extrapolation** `:1183`:`terminal_context_extrapolation=True` 时,**frame 15 的高斯独占** target 后期时刻的 opacity——保证 16–24 帧外推完全由最后一次观测负责,anchor/内插仍由各自表征负责。
- **光栅化**:颜色通道 concat `[RGB, flow, feat]`,调 `gsplat.rasterization(render_mode="RGB+ED")`,输出 `rendered_image [1,20,3,H,W,3]` + depth + alpha,再按通道 split 回各模态。
- `render_target_chunk_size` 按 target 分块渲染,降峰值显存。

### 5.5 流式会话 `stream_session.py`

- 帧序强制 `[0,3,6,9,12,15]`,6 步因果会话:
  - **Step 1–5**:`render_targets=False`,只累积 gs_params 与 Aggregator KV cache(seq 1→5);
  - **Step 6(terminal)**:`render_targets=True`,一次性渲染全部 20 target。
- 之后**滑窗**:丢最早 1 帧的 cache,保留最近 5 帧,准备下一窗口。
- `emit_terminal_perception_tokens=True` 时输出 `latest_perception_tokens [B,3,1200,1536]*`,喂给下游 CatchStateReader。

### 5.6 forward 数据流框图

```text
输入 → Aggregator(Patch+Plücker+Time → Frame/Global 24层)
     → GS头/Motion头/Feat头/Semantic头
     → forward_renderer(MS3 Taylor 推进 + Terminal 独占 + gsplat)
     → render_results{image,depth,flow,feat,semantic}
     → (可选) terminal perception tokens → CatchStateReader
```

---

## 6. Loss 计算

三套**互相独立**的损失体系,对应三种训练:base SLARM、Stream25、CatchStateReader。

### 6.1 Base SLARM 通用损失 `losses.py:889` `compute_loss()`

多损失加权组合,**context 损失有独立 warmup 期**,之后才加 target 帧损失。

| 子损失 | 位置 | 计算 |
|---|---|---|
| RGB(context/target) | — | MSE 或 LPIPS |
| Depth | `:186` | 置信度加权回归 + 多尺度梯度 |
| 3D Point | `:207` | 置信度加权 + 法向约束 |
| Normal | `:73` | `1 − cos(n_pred, n_gt)` |
| Gradient | `:131` | 相邻像素差,4 尺度 |
| Camera | `:588` | 多阶段 L1(平移/四元数/焦距),后期阶段权重更高 γ=0.6 |
| Sky depth/opacity | `:853` | 天空深度→远处、流→0 |

置信度加权通式:`L = γ·L_reg·conf − α·log(conf)`(γ=1.0, α=0.2);`filter_by_quantile(0.98)` 去异常值。

### 6.2 Stream25 十项损失 `stream25_losses.py:235` `compute_stream25_loss()`

**完全独立**,专为网球任务。权重从 `stream25_weights_from_args()` `:38` 读取:

| 损失 | 默认权重 | 计算 |
|---|---|---|
| RGB MSE | 1.00 | 全图 `chunked_mse_loss` `:63`(分块省显存) |
| LPIPS | 0.05 | `chunked_lpips_loss` `:127` |
| Ball RGB | 0.50 | 0.5×(core + 5px 膨胀) |
| Depth relative | 1.00 | SmoothL1 `(pred−gt)/clamp(gt,0.1)` |
| Ball depth metric | 0.02 | 均值 + top-10% 尾部惩罚 |
| **Semantic** | 1.00 | `CE(weight=cw.clamp(max=10)) + (1 − macroDice)`,只算 ball/floor/obstacle |
| LSeg feature | 1.00 | chunked MSE |
| **MS3 ball** | 1.00 | 球区域监督 v/a/j |
| **MS3 static** | 0.25 | 静态区域监督零运动 |
| Opacity | 0.10 | `L1(alpha, valid_depth_mask)` |

**MS3 物理监督** `_compute_ms3_loss` `:410`(结合 `temporal_ownership.py:16`):
- 误差按物理尺度归一化 `scales=[v:5.0, a:9.81, j:1.0]`;损失 = 均值 + `tail_weight`·top-10% 均值。
- **球区域**:监督仿真速度 + 重力 `a=−9.81` + `j=0`;**静态区域**:`v=a=j=0`。
- 类权重来自训练集频率的 inverse-sqrt,cap 到 10。

### 6.3 CatchStateReader 三项损失 `catch_state_losses.py:31`

```text
L_xy        = SmoothL1(pred_xy/0.1, gt_xy/0.1)        # 接球水平位置
L_velocity  = SmoothL1(pred_v/1.0,  gt_v/1.0)         # 接球速度(3D)
L_attention = CE(first_layer_attention, ball_target)  # 注意力对齐到球 patch
L_total     = L_xy + L_velocity + 0.1·L_attention
```
尺度常量:`XY_SCALE=0.1m`、`VELOCITY_SCALE=1.0m/s`、`ATTENTION_WEIGHT=0.1`。

### 6.4 三套损失对比

| 维度 | Base SLARM | Stream25 | Reader |
|---|---|---|---|
| 子损失数 | 15+(可选) | 10(固定) | 3(固定) |
| Warmup | 有(context 预热) | 无 | 无 |
| 物理约束 | 无硬物理 | 重力 9.81 / jerk 0 | 无(假设硬约束) |
| MS3 | 无 | 球/静态区分 + 尾部惩罚 | 无(但回归速度) |
| 是否走渲染管线 | 是 | 是 | **否**(只吃冻结 token) |

---

## 7. 评估指标(Metric)与验收表

评估在 `scripts/eval_stream25_base.py`,指标定义在 `src/utils/stream25_metrics.py`。核心是**三层结构 + 六时间桶 + 硬验收门**。

### 7.1 六个时间桶

| 桶 | 帧 | 物理时间 | 性质 |
|---|---|---|---|
| anchor | 0,3,6,9,12,15 | — | 观测帧 |
| interpolation | 1,2,4,5,7,8,10,11,13,14 | 0.1–0.4s | 内插 |
| **near** | 16,17 | ~0.5–0.6s | 外推 |
| **mid** | 18,19 | ~0.7–0.8s | 外推 |
| **far** | 20,21 | ~0.85–0.9s | 外推 |
| **farthest** | 22,23,24 | ~0.95–1.0s | 外推 |

### 7.2 各指标公式与验收门

| 指标 | 公式 | anchor→farthest 门 |
|---|---|---|
| **RGB PSNR** | `10·log10(1/MSE)`,封顶 120 | 25→20 dB |
| RGB PSNR p10 | 场景分布第 10 百分位(worst-case) | 23→18 dB |
| RGB SSIM | 标准 SSIM,C1=0.01², C2=0.03² | 0.90→0.80 |
| **ball_rgb_psnr** | 11×11 膨胀球区内 PSNR;球不可见→N/A | interp 22,其余 18 |
| depth AbsRel | `mean(|p−g|/clamp(g,0.1))` | 0.08→0.18 |
| ball_depth_error | 球 mask 内 \|p−g\|,取 median/p95 | median≤0.10m,p95≤0.25m(全桶统一) |
| semantic mIoU | 4 类 IoU 均值,N/A 类跳过 | 0.80→0.55 |
| ball_iou | class=1 的 IoU;球不可见→N/A | 0.75→0.45 |

### 7.3 轨迹动力学指标(未来预测重点)

- **MS3 向量误差** `compute_ms3_vector_error`:对 (v,a,j) 各算 L2,球 mask 内取 median/p95,**全桶统一阈值**:
  - 球速度 median≤0.25 m/s / p95≤0.75;加速度 median≤0.50 / p95≤1.50;jerk median≤1.00 / p95≤3.00
  - 静态区速度 median≤0.05 等(强制静止背景)
- **frame24_position**(最直接的"未来落点精度"):
  从 frame15 渲染深度 + semantic 恢复球心 3D → 转 rig 系 → 用 (v,a,j) **Taylor 积分 8 帧到 frame24** → 与 GT 距离(`stream25_metrics.py:168`,`eval:242`)。
  **门:median ≤ 0.15m / p95 ≤ 0.30m**。
- **context_ms3_***:观测帧上的运动学质量,与 target 版共享阈值。

### 7.4 验收逻辑

- `apply_acceptance_gates()` `eval:66`:每桶算 `worst_normalized_ratio`(越界比率),`worst_ratio=max(所有门)`,≤1 才 PASS。
- `scope_reports()`:**aggregate + 三目各自** 四个 scope 都要过。
- `check_final_test_sentinel()`:排他创建 `experiment_hash.json`(含 ckpt/config/evaluator/表的 hash),防重跑。

---

## 8. Stage1 未来轨迹预测改进建议

目标 metric 明确:**`frame24_position` ↓、外推桶 `ms3_ball_*` ↓、near/mid/far/farthest 的 PSNR·ball_iou·depth ↑**。

> 背景实现(用于定位改进):
> - 训练 target 采样(`stream25.py:171` `Stream25TargetScheduler`):每 step 固定 7 个 target = **1 anchor + 2 interp + 4 extrap**,4 个外推帧分别从 band `[16,17] [18,19] [20,21] [22,24]` 各采 1;评估用全 25 帧。
> - MS3 阶数固定 `ms3_deg=3`(v/a/j,`storm.py:231`),高阶通道按 `8^i` 衰减;**loss 对各 target 帧等权,无 horizon 时间加权**。

### 🔴 高优先级(直接打轨迹外推)

#### A1. 给外推帧加 horizon-aware 时间权重
- **现状**:`compute_stream25_loss` 对所有 target 帧**等权**;远端帧误差随 Taylor 阶数放大却不加权。
- **方案**:对 target 帧 loss 乘 `w(t)=1+λ·(Δt/Δt_max)`(或分桶权重 near<mid<far<farthest),尤其加在 MS3 ball、RGB、depth 上。
- **影响 metric**:farthest 桶 PSNR/depth、frame24_position。
- **成本**:低(改 loss 聚合,加几个权重参数)。

#### A2. 直接监督"球质心轨迹",而不只监督逐像素 MS3 系数
- **现状**:MS3 loss 监督每像素 (v,a,j) 与解析 GT;但验收看的是**积分后的 frame24 球心位置**。系数准 ≠ 积分轨迹准(误差沿时间累积)。
- **方案**:训练时加 **rollout 位置损失**——用预测 (v,a,j) 从 frame15 积分到各外推帧,和 GT 球心 3D/rig 位置算 L2(复用 `integrate_frame24_position` 的积分逻辑,`stream25_metrics.py:168`)。等于把评估指标搬进训练。
- **影响 metric**:frame24_position(最直接)、ms3_ball_*。
- **成本**:中(需在 loss 里取球心、做可微积分)。

#### A3. 球的物体级(instance)运动一致性约束
- **现状**:每高斯/像素独立预测 MS3(`forward_motion_predictor`),球作为刚体,所有球像素理应共享一致平动;像素间不一致会在长时外推放大成"球散开/形变"。
- **方案**:对球 mask 内像素的 (v,a,j) 加**一致性正则**(向 mask 均值靠拢),或显式聚合成单一球速度再广播。
- **影响 metric**:ball_iou(外推桶)、ball_rgb_psnr、ms3_ball_* 的 p95。
- **成本**:中。

### 🟠 中优先级(结构/物理先验)

#### A4. 用已知物理约束替代自由多项式外推
- **现状**:`ms3_deg=3` 自由 Taylor 外推(`storm.py:231`),球区 GT 本就是"仿真速度+重力 g+零 jerk"。模型却在自由回归 a、j,长时外推时 j 的噪声被 `t³/6` 放大。
- **方案**:对球区把加速度**软约束到 g=[0,0,−9.81]**、jerk 向 0 收敛(加先验正则,或直接用解析弹道 `p=p0+v·t+½g·t²`,只回归 v0)。与数据生成物理完全一致,是"免费"强先验。
- **影响 metric**:ms3_ball_acceleration/jerk、frame24_position。
- **成本**:低–中。
- ⚠️ 注意:若未来要支持反弹/空气阻力,此约束需放宽——当前 catch 定义在"首次下降穿越、之前无接触",纯弹道成立。

#### A5. 缓解 terminal 单帧信息瓶颈
- **现状**:`terminal_context_extrapolation=True` 时 frame16–24 完全由 **frame15 的高斯**推进(`slarm.py:1183`);虽然 terminal token 因果汇总了 6 次观测,但**位置外推只用 terminal 高斯的几何**。
- **方案**:让外推的运动状态融合多帧速度估计(如用 frame12→15 的观测速度做一致性锚定/校正),降低对单帧深度噪声的敏感。
- **影响 metric**:frame24_position 的 p95(worst-case)、far/farthest depth。
- **成本**:中–高。

#### A6. 提升外推帧采样密度 / 课程学习
- **现状**:每 step 仅 4 个外推帧,`[22,24]` band 3 帧只采 1(`stream25.py:177`);远端监督相对稀。
- **方案**:(a) 提高外推帧占比(如 1 anchor+1 interp+5 extrap);(b) **课程**:训练前期多采 near,后期逐步加 far/farthest。
- **影响 metric**:farthest 全指标。
- **成本**:低(改 `Stream25TargetScheduler.targets`)。

### 🟡 低优先级 / 诊断增强

#### A7. 增加"球心轨迹误差随帧"诊断曲线
- **现状**:评估分桶但主要是整帧 PSNR/depth;缺一条"预测球心 vs GT 随 frame 16→24 的 3D 距离"曲线。
- **方案**:eval 输出 per-frame 球心误差曲线,定位是"整体偏"还是"末端发散"。
- **成本**:低。

#### A8. depth 头对球区的针对性提升
- **现状**:frame24_position 依赖 **frame15 渲染深度**恢复球心;若 anchor 帧球深度偏,外推起点就错。
- **方案**:加大 anchor 帧 ball_depth 权重 / tail 惩罚,保证外推起点准。
- **影响 metric**:frame24_position(根因之一)。
- **成本**:低。

#### A9. 数据层面扩样
- **方案**:`ball_physics.py` 扩 `flight_time_range`、增加更陡/更平轨迹与不同落点,提升外推泛化;若要预测 **frame 25 以后**,需改 `convert_to_slarm` 的 25 帧上限扩展 GT(当前 `[0,25)` 硬编码,25+ 无监督无法评估)。
- **成本**:中–高(需重新跑仿真采集)。

### 8.1 建议落地顺序

1. **先做 A7 诊断**(看清末端发散形态)+ **A8 起点深度**(保证积分起点准);
2. **A1 时间加权 + A2 rollout 位置损失**(把评估指标搬进训练,最直接);
3. **A4 物理先验 + A3 球一致性**(治长时发散的根因);
4. 视效果再上 **A6 采样课程 / A5 多帧融合**。

> A1/A2/A4 三条改动小、见效快,且都指向 `frame24_position` 这个核心验收门,建议优先。

---

## 9. 关键文件索引

| 功能 | 文件 | 核心符号 / 行号 |
|---|---|---|
| SLARM 训练入口 | `main_slarm.py` | `main():414`、`get_args_parser():107` |
| 流式推理 | `inference_stream.py` | `main():31`(`StreamSession`) |
| 建模型 | `engine_tools.py` | `build_model():25` |
| 主重建模型 | `src/models/slarm.py` | `SLARM:79`、`forward:1582`、`forward_gs_predictor:877`、`forward_motion_predictor:822`、`forward_renderer:1059` |
| 轻量模型 | `src/models/storm.py` | `STORM:35`、`ms3_deg:231` |
| 接球读取器 | `src/models/catch_state_reader.py` | `CatchStateReader:98` |
| 时空聚合 | `src/models/components/aggregator/aggregator.py` | `Aggregator:29`、`forward:204` |
| 流式会话 | `src/models/stream_session.py` | 6 步因果会话 + KV cache |
| Stream25 数据 | `src/dataset/datasets.py` | `Stream25Dataset:870` |
| target 采样 | `src/dataset/stream25.py` | `Stream25TargetScheduler:171`、`build_dense_ms3_gt:237` |
| 通用损失 | `src/utils/losses.py` | `compute_loss:889` |
| Stream25 损失 | `src/utils/stream25_losses.py` | `compute_stream25_loss:235`、`_compute_ms3_loss:410` |
| 接球损失 | `src/utils/catch_state_losses.py` | `compute_catch_state_losses:31` |
| MS3 物理 | `src/models/temporal_ownership.py` | 运动方程 `:16` |
| 评估指标 | `src/utils/stream25_metrics.py` | `ACCEPTANCE_TABLE:30`、`compute_ms3_vector_error:163`、`integrate_frame24_position:168` |
| base 评估 | `scripts/eval_stream25_base.py` | `apply_acceptance_gates:66`、`scope_reports:144` |
| 接球指标 | `src/utils/catch_state_metrics.py` | `compute_per_sample_catch_metrics:27` |
| 数据采集监督 | `data_gen/isaac_generation_supervisor.py` | `main:147` |
| 物理渲染 | `data_gen/isaac_replicator.py` | `generate_scene:234`、`validate_scene_completion:775` |
| 格式转换 | `data_gen/convert_to_slarm.py` | `convert_scene:188`、`_preflight_strict_scene:389` |

---

*本文档由代码库静态分析生成,行号基于分析时的代码快照,后续改动请以实际代码为准。*
