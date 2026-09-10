# 网球落点精度与物理 latent：代码审计和实验路线

日期：2026-09-10。依据当前工作区代码、已有实验文档、用户提供的 validation 100 场景结果，以及下文链接的一手论文和官方实现。

后续实现更新：B + 轻量因果历史 patch 读取 + prefix 状态监督已新增为联合微调实验，见 [代码与运行说明](BALL_TEMPORAL_JOINT_FINETUNE.md)。下文其余方向仍是研究建议，不能视为已实现。

**结论：优先改进已有观测的几何读出、时间状态学习和背景鲁棒性，而不是先堆 token 或扩大模型。当前数据不支持“已经到达几何理论极限”。**

边界：输入固定为三目 `(0,3,6,9,12,15)`，不增加晚观测，也不通过下游视觉分支间接使用晚于 frame15 的球图像。训练标签和训练时教师可以使用已有目标数据，但不能成为推理输入。frame45 当前指固定时刻的解析弹道位置，不等于真实机器人接球成功或真实地面落点。

本文保留最初的分析与实验路线；实际已实现范围以开头链接的联合微调说明为准。

## 1. 当前算法究竟在做什么

```text
6 个时刻 x 3 个视角 RGB + 时间/相机几何
                     |
          patch embedding + ball query
                     |
       帧内 attention / 跨视角因果 attention
              /                         \
     patch 特征                           ball 特征
        |                              末帧 3 x 1536
  GS + MS3 + 渲染                         /        \
        |                           原样导出       mean -> MLP
  mask/depth/velocity                   latent       p15, v15
        |                                             |
   像素状态读出                                  解析弹道 -> p45

B: 每目原 ball latent -> 共享 MLP -> 每目 p15 -> 同一 rig 球心 GT
C: 每目原 ball latent -> cross 本目末帧 patches -> 共享 MLP -> 每目 p15
```

- 每张图 `320 x 240`、patch size 8，对应 1200 个 patch。单个可学习 ball query 在各帧各目复制，不是三个独立学习的参数；经过不同图像和 attention 后形成不同输出。配置见 [C config](../configs/exp0910_003_balltoken_pos_c.yml)，交互见 [aggregator](../src/models/components/aggregator/aggregator.py#L195)。
- **不做末端平均，不等于不做三视角融合。** 当前各目 tokens 和 patches 在主干里已经跨视角交换信息；B/C 也不是独立单目估计器。
- A 导出平均前的 `ball_latents`；B 用逐目位置损失替换 pooled 位置损失；C 在 B 的读出前增加本目 terminal patches 的 cross-attention。B/C 中 velocity、trajectory、landing 仍监督原 pooled 状态。见 [读出](../src/models/slarm.py#L1492)、[池化](../src/models/slarm.py#L1555)、[主状态](../src/models/slarm.py#L1688)、[导出](../src/models/slarm.py#L1769)。
- **C refined token 只用于辅助位置头，不进入主状态和导出的 latent。** 这是当前消融边界，不是实现忘接线。C loss 降低，可能仅说明新读出更强，不能直接宣布部署路径变好。
- C 是 1536 维完整 cross-attention + FFN，新增约 2833 万参数，不是小型定位模块。比较 C 与 B 时应报告参数量、训练时间和推理支路是否实际使用。见 [Block](../src/models/layers.py#L107)。
- 只有末帧 ball token 接收直接状态监督。已有 `ball_traj` 是同一个末帧 `(p15,v15)` 的多时刻展开，不是逐时刻 token 监督。已有 landing loss，不应再把“增加落点 loss”作为新方案。见 [轨迹与落点损失](../src/utils/stream25_losses.py#L340)。

另一个需要纠正的解释：外挂 cross-attention 在 forward 中不改写主干 token，不代表其 loss 不能训练主干。代码没有 detach，未冻结时仍能反传。in-trunk 的差异是 forward 中逐层双向交互，不是独占 backbone 梯度通路。见 [外挂读出](../src/models/slarm.py#L1671)。

## 2. 实验支持什么，不支持什么

位置单位 m，速度单位 m/s。下面两组日志没有完整的共同模型/数据版本记录，不能随意拼成同一 checkpoint 的对照。

| 最新 Stream25，exp0908_003 ckpt007999 | median | p95 |
| --- | ---: | ---: |
| Pixel frame24 position，历史未补偿口径 | 0.04738 | 0.15318 |
| Balltoken frame24 position | 0.06132 | 0.15561 |
| Balltoken pos15 | 0.03468 | 0.08262 |
| Balltoken vel15 | 0.10216 | 0.26935 |
| 渲染 MS3 ball velocity | 0.09914 | 0.21734 |
| Balltoken frame45 | 未提供实测 | 未提供实测 |

| 另一份 fusion_100 日志，pred mask + gravity | frame45 median / p95 | hit/all |
| --- | ---: | ---: |
| mean | 0.1017 / 0.2358 | 62% |
| ray_weighted | 0.1018 / 0.2348 | 63% |
| view_1 | 0.0992 / 0.2440 | 64% |

可以据此判断：

1. 同一最新模型，balltoken frame24 中位误差比像素路径高约 29%，尾部接近；尚未实现“更准的球级输出”。但两者接口与读出口径不同，这不是 latent 质量的完整比较。
2. 相对旧 GS 基线 `0.0523 / 0.1390`，最新像素路径 median 改善约 9%，p95 增加约 10%。不能只报 median 而宣称全面改善，也不能在缺少同预算续训 control 时全归因于 token 设计。
3. 当前 ray fusion 没有明显收益：一百场景多一个命中，逐场景配对平均只改善 0.6 mm。它是对现有末帧三维状态的后处理，不能否定“从六帧二维观测联合估计三维轨迹”。
4. GT mask 替换没有明显改善，不代表二维球心定位已完美：这些实验仍读取预测深度/MS3，没有用 GT 三维状态替换预测状态。
5. MS3 velocity 是逐像素、多目标时刻指标；balltoken vel15 是末时刻球级状态指标，不能用两者 median 相近来认定同样准确。
6. `lower_front` 不足样本的门和真实失败的门应分开。最新结果仍有 depth absrel anchor/interpolation 和 ball IoU near/mid 的真实失败；修复它们未必等价于落点改善。fusion 的第三目在 frame15 也只有 27/29 个有效场景，应区分可见性、mask、非有限预测等原因，不要统称“后期遮挡”。

### 旧误差预算中的推导不能充当理论下界

[原误差预算](EXPERIMENTS_AND_ERROR_BUDGET.md) 保留了历史判断，但其“0.163 px 极限”“速度已达到 LS 下界”“算法没有余量”需要撤回为未验证假设：

- `along_med=0.0292` 来自每场景最差有效相机的绝对方向误差，再跨场景取 median；不是每帧独立测量噪声的标准差。见 [实际统计](../tools/verify_physics_extrapolation.py#L168)。不能直接代入独立同方差 LS 方差公式。
- 两个误差中位数的平方比不是总体 MSE 的方向能量分解；几组终点 median 的直线拟合也不能证明“速度贡献 75%”。
- 正确的逐场景关系是 `e_T = e_p + T e_v`。同一 T 下，`E||e_T||^2 = E||e_p||^2 + T^2 E||e_v||^2 + 2T E[e_p dot e_v]`。最后一项可能抵消，也可能放大；应直接统计。
- 用平均距离、焦距、基线换算出的 0.163 px 是等效视差误差，不是传感器或当前学习器的已证实极限。
- GT 深度是球表面，GS 渲染期望深度并非精确球表面。补半径只有毫米级改善的实验，不应被重新包装成厘米级收益来源。
- 文档中 patch8 改 patch4 的成本倍率写错了：同分辨率 patch 数是 4 倍，稠密 attention 的二次项约 16 倍，而非 9/81 倍。这仍不便宜，但不能据错误倍率排除局部高分辨率读出。

## 3. 优先方向一：让已有时间 token 学成持续可读的状态

**这是最适合先做的小改动，不需要更多输入或更多 query。**

当前六个时刻的 tokens 都参与主干计算，但直接物理监督仅落在最后一步。新增共享状态头读出 frame6/9/12/15 的因果 prefix tokens：

```text
z6  -> p6,  v6  -> 预测同一绝对 catch 时刻
z9  -> p9,  v9  -> 预测同一绝对 catch 时刻
z12 -> p12, v12 -> 预测同一绝对 catch 时刻
z15 -> p15, v15 -> 主输出
```

建议损失：`L_prefix = mean_t w_t [L_pos(p_t,GT_t) + L_vel(v_t,GTv_t)]`。使用各时刻真实时间戳。第一组先只增加这项，不同时加新网络。frame0 不要求单次观测准确恢复速度；较早 prefix 低权重，保证末帧损失不被新项数量淹没。

之后再单独测试跨 prefix 的终点一致性：各自按 `t_catch - t` 外推，与 GT 或 stop-gradient 的末帧预测比较。对 GT 的状态监督不可移除，否则几个错误预测也能彼此一致。先核对批量训练的 attention 确实因果，并检查 cache 流式/批量输出一致，防止历史 token 含有未来信息。

**与现有 trajectory loss 的区别：约束不同观测前缀产生的不同 latent，而不是重复展开同一个末帧状态。** 预期价值是状态在时间上易读、可更新；是否降低 frame45 p95 仍需实验。

### 对 C 的后续扩展：专门读取运动证据

保留现有 B/C 作为干净对照，不静默改其含义。新增方案另命名：

- `C-small`：先将同样的本目 terminal cross-attention 压到 256/384 维，再 residual 投回 1536；只改变容量，检验定位机制是否依赖大新头。
- `C-temporal`：位置 query 偏向末帧球区域；运动 query 读取已有六帧、三目的球相关 memory，携带时间戳、ray/camera 标识。当前 terminal patches 已含历史，因此新增的是显式访问历史证据，不是第一次引入时间信息。
- `C-export`：让 refined token 真正进入主状态和导出接口，另做部署读出消融。注意 `MLP(mean(z))` 不等于 `mean(MLP(z))`，不能混用名称。

正式实验的 ROI 必须由预测位置/热图选择，漏检保留全局后备或明确报失败；GT ROI 只作诊断或训练辅助。流式版本还需保存带时间戳的历史 ROI memory，不能把已有 attention KV cache 当作已提供了所需的历史末层 patch 特征。

局部读取可借鉴 TAPIR 的“粗匹配 + 轨迹邻域相关性细化”，但球心不是可持续跟踪的纹理表面点，不能假定通用点跟踪权重直接适配 2~3 px 网球。只在允许的历史窗口内细化。[TAPIR 原文](https://arxiv.org/abs/2306.08637)

## 4. 优先方向二：二维球心证据 + 联合弹道拟合

**这是我认为最值得作为主要算法创新路线尝试的方向。** 思路不是多加一次对三维预测的平均，而是改变产生三维状态的方式：

```text
已有六帧三目图像/patches
      -> 球心 heatmap + 亚像素 offset + 可见性/不确定度
      -> 时间、相机标定、重力约束下联合拟合 p15/v15
      -> 投影回历史各帧，局部读取并修正观测（后续消融）
      -> 物理状态 + 几何证据 latent + p45
```

TrackNet 对高速小球使用多帧热图定位，TrackNetV3 进一步利用背景信息和轨迹修复。可借鉴的是“小目标定位专用分支”和背景处理，不能照搬整段视频背景估计、前后帧补洞：后者可能访问 frame15 后的信息。其像素级命中率也不能直接换算成本任务的三维落点误差。[TrackNet](https://arxiv.org/abs/1907.03698)，[TrackNetV3 原文](https://people.cs.nycu.edu.tw/~yushuen/data/TrackNetV3.pdf)

### 具体估计器

设 `u_it` 是相机 i、时刻 t 的二维球心，`Pi_it` 为标定投影，`dt=t-t15`：

```text
s_hat = argmin_(p,v) sum_(i,t) w_it * robust(
    Pi_it(p + v*dt + 0.5*g*dt^2) - u_it
)
```

只有六个状态未知数，不需要重建整个三维体素空间。可先用射线线性最小二乘初始化：令单位射线 d、相机中心 c、`Q=I-dd^T`，堆叠

```text
[Q, dt*Q] [p; v] = Q (c - 0.5*g*dt^2)
```

使用带阻尼的稳定求解，再按像素重投影误差细化。射线距离与像素距离权重并不等价，不能把初始线性解当作已校准最大似然解。所有相机和时刻须位于同一参考系；未来若 rig 运动，要显式处理逐时刻外参。

可微二维检测加三角化已有成熟先例；对本任务的适配点是**把三目六帧一起放进六维弹道状态求解器**，避免先给每个极小球区域读一个不稳定的独立深度，再拟合速度。[Learnable Triangulation of Human Pose](https://arxiv.org/abs/1905.05754)

### 监督与风险

- 二维监督目标用 GT 三维球心的标定投影，不把可见 mask 质心当精确球心。可见性来自真实遮挡/可见标注，不能只用是否投影落在画内判断。
- GT 可见性只用于训练监督和明确标注的诊断。正式推理的观测选择用预测可见性，不能让 GT 遮挡标签替代模型的判断。比较求解器与直接回归时，尽量固定输入的预测二维测量。
- 可见球监督 heatmap/offset；不可见帧可以监督多视角上下文推断的三维状态，但不能把猜出的二维点当独立实测证据高权重喂给拟合器。
- `L = L_current + lambda_2d L_center + lambda_state L_fit_state + lambda_end L_fit_endpoint + lambda_vis L_visibility`。先只验证观测头和拟合器，再加迭代 refinement，避免一次变更多个因素。
- 置信度必须有监督或校准，不能只乘在残差上让网络学成全零。处理无观测、病态矩阵、解在相机后方的失败情况；失败计入 hit/all。
- 同一主干生成的多目观测可能相关。拟合器的形式协方差不自动等于真实预测不确定度，要在 heldout 场景检查覆盖率。
- 球只有 2~3 px，优先增加保留原有像素细节的浅层 CNN/局部读出，不先扩大整图 Transformer。把低分辨率 crop 插值放大不会创造新的光学信息。

**先做离线可行性实验：** 相同可见观测集合下，依次输入精确 GT 球心投影、GT-mask 质心、预测球心。第一组是坐标/时间/求解器 sanity check，不是模型精度上限；后两组定位读出问题。若预测观测的轨迹拟合不比现有路径好，先改观测，不急于端到端接回大模型。

## 5. 低成本方向：端点参数化，而非再添加一份端点 loss

现有头为 `z -> (p15,v15)`。另做等维度对照：

```text
z -> (p15,p_catch)
v15 = (p_catch - p15 - 0.5*g*T^2) / T
T = t_catch - t15
```

保留位置、速度、轨迹和已有 landing 监督。它与原参数化物理信息等价，实验检验的是优化条件和梯度耦合，不是增加信息或突破测量下界。

若 T 固定、最后一层为线性层，可用原 p/v 输出层的线性组合初始化端点输出，使改动前后的 p/v/p_catch 初始预测一致；这样不把重新初始化的损失误认为方案差异。T 可变时，需时间条件化设计，不能用一套固定转换权重冒充通用模型。

同时评估 frame24/30/36/45，防止只在单一时刻利用 `e_p + T e_v` 的抵消降低误差。只在相同 checkpoint、相同续训预算下，与“原 p/v + 已有 landing loss”比较。

## 6. latent 质量：从辅助头能预测，变成输出本身可用

**无法靠某个 loss 或增加 token 数量保证 latent 高质量。** 应先定义可检验的要求：物理状态易读、背景变化时稳定、观测不足时有可解释的不确定性，同时保留有用的局部证据。

### 最值得增加的训练目标

**保轨迹、换背景的状态一致性。** 对相同球轨迹、相机、时间戳，换背景或重渲染外观；约束 `q_state(z_a)` 与 `q_state(z_b)`，同时保留 GT 状态监督。第一阶段固定遮挡条件，只改变外观，避免把“没有新证据”和“有证据”混成同置信度样本。背景几何改变造成真实遮挡时，应单独评价可见性和置信度。

不强迫整个场景 latent 或不同时间的整个 ball latent 相同。可以先保留现有三个 tokens，只在其投影的状态子空间做一致性；避免新增 slots 与新 loss 同时上线。使用同背景不同轨迹的样本以及 GT probe，排除常量特征也能降低一致性 loss 的退化解。

**物体局部的预测特征监督，作为第二阶段。** 用末帧球 latent 预测已有目标时刻的局部几何特征或可靠教师特征，教师冻结或 stop-gradient，并有防塌缩机制。先验证教师能表达小球中心/运动，不能默认全图 DINO patch 对 2 px 球足够敏感。训练可使用 frame16~24 的目标信息，推理不得使用这些图像，且 frame45 没有真实目标图像可做教师。

DINO-WM 表明未来 patch 特征预测可以作为控制相关表征目标，不必完全依赖 RGB 重建；这里借鉴目标形式，不移植其规划成功率，也不建议为网球重新训练整套通用 world model。[DINO-WM](https://arxiv.org/html/2411.04983v2)

### token 扩容的合理方式

先验证三个现有 tokens 的状态信息是否互补。确需扩容时，从少量具备不同读取任务的 query 开始，例如位置/运动 query 加三目证据 tokens；状态监督作用于相应状态读出，证据 tokens 接局部预测/可见性任务。

不要给所有 K 个 tokens 复制同一六维监督后期待自动分工，也不要仅靠正交损失制造表面差异。仅给 query 起不同名字不会产生解耦，仍需 slot 屏蔽、相同容量 probe 等消融。

### 必须补的 latent 验收

| 测试 | 做法 | 回答的问题 |
| --- | --- | --- |
| 冻结 probe | encoder 固定，在 train split 训练小型 pos/vel/p45 头，在 heldout 评估 | 物理信息是否容易读取 |
| 原始/增强对照 | mean、三目集合、C refined 使用相同输出瓶颈和受控容量读出 | 新支路改善了 latent 还是只改善读出容量 |
| 背景配对 | 同轨迹换背景，报状态/端点预测变化及真实误差 | 是否依赖背景捷径 |
| 视角缺失 | 有效输入 mask 与缺失训练，报 error、hit/all、coverage | 不完整输入时是否稳健 |
| token 消融 | 屏蔽单个 token/角色，报退化；有效秩仅辅助 | 是否真正互补 |
| 不确定性校准 | NLL、置信区域覆盖率、risk-coverage | 置信度能否解释失败 |

这些是感知代理指标，不替代真实 action 成功率。Probe 数据必须按场景/轨迹分开，避免同一弹道换背景后泄漏到两侧。

## 7. 更大改动：球级持久 GS 与状态头共享物理状态

当前球级读出与 dense GS/MS3 可以通过不同路径拟合监督。更强的表示约束是给球分配一小组持久 GS，共享一个球心轨迹：

```text
mu_j(t) = p15 + v15*dt + 0.5*g*dt^2 + R(t)*delta_j
ball latent -> p15/v15 + 球的局部形状/外观
同一组状态 -> p45
同一组状态 -> 球 GS 位置 -> 多时刻 ROI 渲染监督
```

最小版本先固定球形/半径和局部偏移结构，不从几像素球强行估计不可辨识的旋转。背景仍由原 GS 分支处理。要处理球/背景遮挡合成与重复球表示，不能只叠加第二个球而让旧 dense 分支继续解释同一物体。

这样 RGB、语义、几何监督需要经过同一物理状态，减少“球级状态错但像素重建仍可由另一条路拟合”的自由度。已知球半径约束的是三维表示，不是给预测深度统一加半径。

Dynamic 3D Gaussians 的可借鉴点是持久属性和运动一致性。原方法逐时刻进行测试场景优化，并不是从六帧一次前向预测未来，也不能把其渲染 FPS 当作状态估计延迟。本建议是把对象持久性先验移植到前馈 SLARM，而非照搬整套优化管线。[Dynamic 3D Gaussians 原文](https://dynamic3dgaussians.github.io/paper.pdf)

这是中长期方向，先做球 ROI-only 对照；若不改善落点/latent probes，不应仅因为重建更好就扩大实施范围。

以上构件在文献中都有相关先例，研究价值在于固定观测预算下的组合、共享状态约束及可验证收益；本文不宣称其首创性。

## 8. DynamicVLA：保留结构化状态和证据 latent 两条信息

建议接口内容为：`ball_latents`、预测 p/v、可选联合状态协方差、视角有效性、对应观测时间、到 catch 的时间，以及坐标系/标定标识。六维状态给明确几何，latent 保留状态以外的观测证据；是否比 state-only 更有用必须验证。

若输出联合协方差 `Sigma_s`，固定重力下 `Sigma_p(T) = [I, T*I] Sigma_s [I, T*I]^T`，包含 p/v 交叉相关。只预测六个独立方差会丢失这部分；增加协方差头本身不保证降低点误差。

核对 DynamicVLA 官方实现后，其原路径为：

```text
图像/FastViT + 语言 + robot state
              -> SmolLM prefix -> 每层条件 KV
              -> action expert 多次去噪 -> action chunk
```

`sample_vlm_embedding()` 生成条件 cache，`denoise_step()` 读取。SLARM tokens 不是现成的逐层 KV，不能只对齐向量维度就拼进去。原 expert cross-attention 复用 VLM 的 keys，并有特定投影及位置编码逻辑。[官方主模型](https://github.com/hzxie/DynamicVLA/blob/master/policies/dynamicvla/modeling_dynamicvla.py)，[VLM/expert 层实现](https://github.com/hzxie/DynamicVLA/blob/master/policies/dynamicvla/modeling_vlm_with_expert.py)

建议顺序：

1. **先做 prefix 融合。** SLARM 少量 tokens 经 LN/MLP adapter，连同时间/结构化状态作为新模态进入原 SmolLM prefix；贯通训练、推理和 masks。原视觉/语言路径保留。先冻结 SLARM，用预测而非仅 GT 状态训练下游。
2. **再做 expert 独立 cross-attention 的位置消融。** 保留原 SmolLM 条件路径，另加 `h += gate * CrossAttn(h, SLARM_memory)`，独立 Q/K/V。SLARM 这一路可绕过 SmolLM，但不是移除整个原 VLM。额外 cross-attention 每次动作去噪仍有成本，是否更快需要测量。
3. 下游至少比较原 DynamicVLA、state-only、latent-only、state+latent，并控制 adapter 容量和总时延，证明 latent 在明确状态之外有增量价值。

以上是新增融合方案，不是官方现有 SLARM 接口。论文的 Latent-aware Action Streaming 是动作时间对齐机制，不是外部 latent 融合方法。[DynamicVLA](https://arxiv.org/abs/2601.22153)

**时间与坐标不能省略：** frame15 后 latent 是旧观测的编码，不能改时间戳伪装成新观测。显式 p/v 按实际 elapsed time 推进，另输入 observation age 和 time-to-catch。静态 rig 到 robot-base 的变换为 `p_B=R*p_R+t`、`v_B=R*v_R`、`g_B=R*g_R`；latent 本身不能像三维向量一样旋转。真实运动 rig 还需处理参考系运动。

测量曝光、SLARM、传输、prefix、expert 到实际执行的端到端 p50/p95。现有 `render_targets=False` 仍生成 GS 等头，并非专用 latent-only 快速通路；真正部署应另测跳过无关头/渲染的收益，不能把“导出三个 token”当作已经完成加速。见 [当前 forward](../src/models/slarm.py#L1699)。

## 9. 最小实验顺序和停止条件

| 优先级 | 实验 | 第一轮只改变什么 | 继续投入的条件 |
| --- | --- | --- | --- |
| P0 | 统一逐场景评测 | 导出 pooled/raw/refined、p/v 残差、frame24/45 | 口径与 checkpoint 完整可追溯 |
| P0 | 冻结 latent probe | 原始和 refined 的受控读出 | 明确表征瓶颈还是读出瓶颈 |
| P1 | 现有 control/B/C | 按已有 configs 跑，不改范围 | 主 frame45 和 latent probe，而非辅助 loss 改善 |
| P1 | 因果 prefix 状态监督 | 复用已有历史 token 和状态头 | 末帧/尾部改善，流式一致 |
| P1 | 端点参数化 | 6 维输出参数化 | 同预算优于原 p/v+landing，多 horizon 不退化 |
| P1 | 离线球心联合弹道拟合 | 不重训 SLARM，先检查读出可行性 | 预测观测路径优于现有部署读出 |
| P2 | C-small -> C-temporal | 先改容量，再改历史读取 | 收益不能由头容量或成本差异解释 |
| P2 | 配对背景一致性 | 先固定遮挡，只变背景外观 | heldout 背景鲁棒性和端点 p95 改善 |
| P3 | 球级 GS 共用状态 | 球 ROI 表示和监督通路 | 状态、重建、latent 同时受益 |

现有可运行命令与 control/B/C config 对应关系在 [BALL_POSITION_ABLATIONS.md](BALL_POSITION_ABLATIONS.md#training-commands)。上述新方案尚无代码/config，不应使用臆造的开关启动。

评测规则：同场景、同 checkpoint 起点、同数据顺序/训练预算，先单 seed 筛选，再对候选做多 seed 和按场景配对 bootstrap。训练/验证按底层场景与轨迹去重。保留全部样本分母，报告失败原因；frame24 历史 worst-view/free 指标与可部署 mean/physics 指标分栏，不能互相替换。

结果至少包含 frame24/45 median、p95、hit/all、pos15、vel15、p/v 交叉项、球可见尺寸/距离/遮挡分层、latent probe 和延迟。100 场景的 p95 只由很少尾部样本决定，一场命中变化不应直接宣称显著提升。

如果最终任务是球穿过接球平面，还要评估平面交点二维误差、到达时刻误差、无交点/不可达比例和实际接球成功率。frame45 解析 GT 只验证当前弹道设定；当前仿真里 a 已接近重力、j 接近零，暂不优先扩大自由加速度/jerk 模型。真实阻力、旋转和碰撞需要真实残差证据后再建模。

**优先推荐组合：因果状态监督 + 几何约束的球心轨迹估计 + 背景稳健的物理子空间。** 这三个方向分别改善时间信息可读性、状态估计结构和 latent 泛化；成立与否可逐项消融，不需要增加晚观测，也不依赖“更多 token 自然更好”的假设。
