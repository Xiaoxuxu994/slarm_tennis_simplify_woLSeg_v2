# Ball-token 可视化

入口：[scripts/visualize_ball_tokens.py](../scripts/visualize_ball_tokens.py)。脚本只做推理、读取特征和绘图，不改模型参数、不回写 patches，也不改变训练配置。默认单卡、validation 前三个场景，输出 PNG、MP4 和可离线重画的 NPZ。

## 1. 004 运行命令

在已配置 SLARM/CUDA、数据和 checkpoint 的训练机，进入仓库根目录：

```bash
CUDA_VISIBLE_DEVICES=0 bash run_sh/visualize_ball_tokens.sh \
  --config configs/exp0910_004_balltoken_temporal_joint.yml \
  --checkpoint work_dirs/slarm/exp0910_004_balltoken_temporal_joint/checkpoints/ckpt_003999.pth \
  --scene-indices 0,1,2 \
  --output-dir output/ball_token_004_ckpt003999 \
  --video mp4
```

已有 1k checkpoint 就能先看：将上面的 `ckpt_003999.pth` 替换为 `ckpt_000999.pth`，同时换一个输出目录。不要占用正在训练的 GPU。数据不在 config 默认位置时，加 `--data-root /absolute/path/to/data/slarm_data`。

**可视化必须使用与 checkpoint 配套的 config。** 004 在训练初始化时可以 `load_from` 旧 007999，但不能把旧 007999 当成训练后的 004 来画 attention。此脚本会拒绝缺失参数或多余参数，并核对 checkpoint 保存的 in-trunk、position supervision、prefix supervision、temporal refine 开关，避免错标监督模式。缺少保存的 `args` 的裸权重也会被拒绝；请使用训练脚本保存的完整 checkpoint。它不替代完整的数据/配置契约检查。

输出目录必须是新目录，已有目录会报错，不覆盖历史实验。`--scene-indices` 是所选 manifest 的局部下标，不是 JSON 标注中的 scene ID；真实 scene name 也会记录到输出。

## 2. 输出内容

```text
output/ball_token_004_ckpt003999/
  run.json
  scene_0000/
    overview.png
    attention_query_view0.png
    attention_query_view1.png
    attention_query_view2.png
    trajectory.mp4
    tokens.npz
    metrics.json
  scene_0001/...
  scene_0002/...
```

| 文件 | 展示内容 | 正确解读 |
| --- | --- | --- |
| `overview.png` | 末帧预测轨迹、frame45 端点、各 prefix 的接球误差、末帧三目 cosine、latent PCA、增强量 | 状态和特征的联合诊断，不是 latent 高质量的证明 |
| `attention_query_view*.png` | 指定 query view 的球 token，读取历史六帧三目 patches 的权重 | 新 temporal refiner 的 attention，不是原 aggregator 的完整 attention，更不是球定位概率 |
| `trajectory.mp4` | 随 frame0/3/6/9/12/15 更新预测；之后固定末帧状态做物理延续，旁边显示最近的真实观测 | frame15 后没有新 latent，也没有新图像输入 |
| `tokens.npz` | `[6,3,1536]` raw/refined tokens、状态、GT、context RGB 和可选 attention | 无 pickle，可供离线绘图和进一步诊断 |
| `metrics.json` | 每个 prefix 的位置/速度/接球点误差，单位 m、m/s | 单场景诊断，不替代 100 场景 acceptance 评测 |
| `run.json` | config/checkpoint 路径、参数、场景索引和摘要 | 可追溯本次可视化所用来源 |

- frame0..24 的灰线是标注轨迹，frame25 以后明确使用 frame15 GT 状态的解析重力延续。固定 frame45 端点不是地面相交事件，也不是机器人实际接球成功率。
- 动画默认播放 8 fps，独立于场景物理 fps；30 fps 数据相当于约 3.75 倍慢放，画面上会标明。全轨迹 GT 是离线诊断参考，不送入预测模块。
- 004 的 frame6/9/12 是直接监督的早期状态；frame0/3 读出仍属诊断。视频可能暴露早期预测不稳定，这是模型输出，不能作为已校准的在线跟踪结果。
- PCA 在每个场景内对 raw/refined 一起拟合一个投影，不使用 GT 标签。PCA 坐标不是物理坐标，不同场景或独立运行的 PCA 轴也不能直接比较。
- attention 在全部可读历史、视图和 patches 上 softmax，再对 heads 平均。每张图的所有子图共用色标，子图标题的 mass 是该时刻/相机的权重和，没有对子图单独归一化。
- 静态 overview 重点呈现末帧预测；动画固定坐标范围包含全部 prefix 预测，早期离群预测可能使坐标范围变大，不会通过移动视窗伪装收敛。

## 3. 验证早期 token 的因果读取

```bash
CUDA_VISIBLE_DEVICES=0 bash run_sh/visualize_ball_tokens.sh \
  --config configs/exp0910_004_balltoken_temporal_joint.yml \
  --checkpoint work_dirs/slarm/exp0910_004_balltoken_temporal_joint/checkpoints/ckpt_003999.pth \
  --scene-indices 0 \
  --attention-frame 6 \
  --output-dir output/ball_token_004_query6 \
  --video none
```

frame6 query 只能读取 0/3/6，后续列显示 `Future: masked`，不展示未来 RGB。脚本仅重算选定球 query 行的 QK，不物化整张 patch-to-patch attention 矩阵。GT 球心、mask、速度不参与 attention 计算。

## 4. 旧 baseline 也能可视化

```bash
CUDA_VISIBLE_DEVICES=0 bash run_sh/visualize_ball_tokens.sh \
  --config configs/exp0908_003_slarm_stream25_0903_2k_balltoken_intrunk_landing.yml \
  --checkpoint work_dirs/slarm/exp0908_003_slarm_stream25_0903_2k_balltoken_intrunk_landing/checkpoints/ckpt_007999.pth \
  --scene-indices 0,1,2 \
  --output-dir output/ball_token_baseline_ckpt007999 \
  --video mp4
```

旧 baseline、002 B、005 prefix-only 没有新 temporal refiner，因此不生成这组三目 attention 图，并明确输出 unavailable。旧 baseline 的早期状态是同一个末帧训练头在早期 latent 上的诊断读出，未做直接 prefix 监督。003 C 的图展示主输出使用的原 token，不把仅辅助位置分支中的增强当成实际导出的 latent。

## 5. 离线重画

提取 NPZ 后，不需要 CUDA、SLARM 模型或 checkpoint 即可重画：

```bash
python scripts/visualize_ball_tokens.py \
  --from-npz output/ball_token_004_ckpt003999/scene_0000/tokens.npz \
  --output-dir output/ball_token_004_replot \
  --video gif
```

离线模式只需要 NumPy、Matplotlib、ImageIO/Pillow；MP4 另外需要 `imageio-ffmpeg`。这些绘图/视频依赖已在仓库训练依赖中。`--video both` 同时生成 GIF/MP4，`--video none` 只画静态图。离线模式使用 NPZ 中已存的 attention query 时刻；更换 query 时刻必须重新提取。

不需要 attention 时加 `--no-attention`。推理默认 BF16，可用 `--dtype float32` 做数值检查。脚本使用完整六帧因果前向并关闭目标图像渲染，仍会运行原 GS/MS3 heads，因此提取端仍需完整 SLARM 运行依赖。

## 6. 验证边界

```bash
python -m pytest tests/scripts/test_visualize_ball_tokens.py \
  tests/scripts/test_ball_token_viz_attention.py \
  tests/scripts/test_ball_token_viz_plot.py -q
```

CPU 测试覆盖真实 temporal refiner 的 QK 与 SDPA 对照、因果 mask、BF16、hook 清理、状态与导出一致、checkpoint key 不匹配拒绝、NPZ 离线重画、物理时间换算及图像/视频输出。合成夹具只用于检查脚本和排版，不代表模型结果。本机无真实训练 checkpoint、数据和 CUDA renderer；真实场景的提取、耗时、显存和最终图像内容需训练机验证。
