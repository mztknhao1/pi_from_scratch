# 第 3 讲：动作表征与归一化——模型究竟在预测什么数字？

> 第二讲保证了图像、状态和未来动作在时间上对齐。但“对齐”只说明这些数字来自正确的时刻，还没有说明这些数字是什么意思。

CVer熟悉的图像输入使用像素值矩阵表示，如果用RGB表示则是3通道 (H, W, 3)，存储时可能使用unit8存储，像素值离散取0～255；但是训练时每个像素值还可能归一化到0～1，并且可能输入mean, std这个作为超参。
动作表征也要考虑存储和训练两种情况，并且数字到底表示了什么含义也要约定清楚，这样才能真正的沟通机器人

## 模型的输出数字含义到底是什么？

假设机械臂当前关节角是 `1.0 rad`，数据集中的下一个 action 是 `1.1`。这个 `1.1` 可能表示：

- 目标关节角是 `1.1 rad`；
- 在当前位置上增加 `1.1 rad`；
- 关节速度是 `1.1 rad/s`；
- 一个已经被缩放过、没有物理单位的 normalized value。

它们的 tensor shape 都可以是 `[1]`，但发给机器人后会产生完全不同的运动。模型无法从字段名 `action` 自动推断这些语义；语义必须由数据适配器、训练配置和 runtime 共同遵守。

> 怎样把机器人原始动作明确转换成模型学习的动作表征，怎样只用训练集统计量调整数值尺度，并保证模型输出能够无歧义地变回环境动作？-> 约定表征含义、实现正逆变换处理

可以带着这几个问题进入接下来的阅读
1. 区分 absolute、current-state delta 和 velocity；
2. 解释归一化统计量究竟是什么、为什么只能来自 train episodes；
3. 写出训练和推理两条严格互逆的数据路径；
4. 用 round-trip test 证明变换没有丢失动作语义。

## 一、动作空间的约定

```text
space           控制的是关节、末端位姿，还是 simulator 特有坐标？
representation  数值是绝对目标、相对量，还是单位时间内的变化率？
frame           世界坐标系、机器人基座坐标系，还是末端局部坐标系？
units           rad、m、rad/s、pixel，还是无量纲开合量？
frequency       一秒执行多少个命令？
bounds          合法范围和安全范围分别是什么？
```

这就是第一讲中 `ActionSpec` 存在的原因。例如 PushT 的二维 action 是工作区中的绝对目标位置，不能因为 shape 恰好是 `[H, 2]`，就把它当作二维速度。

真实机械臂还可能在同一个 action 中混合多种语义：前六维是关节 delta，最后一维夹爪仍是绝对开合位置。此时“一整个 tensor 都是 delta”也不够准确，必须额外保存 per-dimension mask 或字段布局。

## 二、absolute、delta 和 velocity 回答的是不同问题

设观测时刻的机器人状态为 $s_t$，未来绝对动作目标为：

$$
A_t^{abs}=[a_t^{abs},a_{t+1}^{abs},\ldots,a_{t+H-1}^{abs}]
$$

### 2.1 Absolute：要到哪里？

absolute action 直接表示目标位置：

$$
a_{t+h}^{model}=a_{t+h}^{abs}
$$

优点是目标不随参考状态改变，也不会在长序列积分中累计误差；缺点是不同机器人零位、安装方式和工作空间会直接进入数值分布。

PushT 原始 action 就属于这一类：两个维度表示二维工作区内的目标坐标。

### 2.2 Current-state delta：相对现在要移动多少？

π 风格数据处理中常见的 delta 让每个未来目标都减去同一个当前状态：

$$
a_{t+h}^{delta}=a_{t+h}^{abs}-s_t
$$

例如：

```text
current state:           10, 20
absolute future targets: [11, 22], [13, 25], [16, 29]
delta from state:        [ 1,  2], [ 3,  5], [ 6,  9]
```

请留意，这和下面这种相邻差分含义不同：

```text
[11, 22] - [10, 20]
[13, 25] - [11, 22]
[16, 29] - [13, 25]
```

两种算法的第一步相同，后面却完全不同。若训练使用第一种、推理 inverse 却使用第二种，第一步之后的 action 全会错。

在 openpi 的 [`DeltaActions`](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/transforms.py) 中，每个未来 action 也是相对当前 state 转换，并通过 dimension mask 允许关节使用 delta、夹爪保持 absolute。本讲的 `CurrentStateDeltaTransform` 保留了这个关键语义。

### 2.3 Velocity：每秒变化多少？

若离散控制频率为 $f$，控制周期为 $\Delta t=1/f$，本讲用有限差分定义速度：

$$
v_t = \frac{a_t^{abs}-s_t}{\Delta t}, \qquad
v_{t+h}=\frac{a_{t+h}^{abs}-a_{t+h-1}^{abs}}{\Delta t}
$$

逆变换是从当前状态开始积分：

$$
a_{t+h}^{abs}=s_t+\sum_{i=0}^{h}v_{t+i}\Delta t
$$

因此 velocity 的数值离不开时间单位。相同的位置差，在 10 Hz 下和 20 Hz 下代表的速度相差一倍。改变 fps 后只重采样数组、却不重新解释速度单位，是一种训推不一致。

### 2.4 什么是“最佳表征”？根据业务需要制定

选择取决于环境原生接口、预训练策略使用的约定、不同机器人的可对齐程度以及控制稳定性。这里遵循三个判断顺序：

1. 先确认数据集 action 的原生语义；
2. 再确认要兼容的 pretrained policy 期待什么语义；
3. 最后用离线 round trip 和闭环指标比较候选变换。

不要仅凭“delta 通常更容易学”就修改数据，更不能对本来已经是 delta 或 velocity 的动作再次求差。

## 三、归一化统计量是什么？

动作语义确定后，不同维度的数值尺度仍可能相差很大。例如：

```text
joint angle:       大约 -3 到 3 rad
gripper position:  大约  0 到 1
base velocity:     大约 -0.2 到 0.2 m/s
```

如果直接计算统一的回归损失，大尺度维度更容易主导梯度。归一化的目的，是给模型建立一个更均衡的数值坐标系；它不会改变动作的物理语义。

本讲先实现最容易看懂的逐维 z-score：

$$
\mu_j=\frac{1}{N}\sum_{i=1}^{N}a_{i,j},\qquad
\sigma_j=\sqrt{\frac{1}{N}\sum_{i=1}^{N}(a_{i,j}-\mu_j)^2}
$$

$$
\tilde a_{i,j}=\frac{a_{i,j}-\mu_j}{\max(\sigma_j,\epsilon)}
$$

这里的归一化统计量包括每个 action 维度的 `mean`、`std`、参与计算的有效 action 数量，以及这些 action 来自哪些 train episodes。它们在训练前扫描数据得到，作为 dataset artifact 保存，不参与神经网络的反向传播。

推理时必须使用训练时保存的同一份统计量：

$$
a_{i,j}=\tilde a_{i,j}\max(\sigma_j,\epsilon)+\mu_j
$$

openpi 同样把 state/action 的 normalization statistics 随 checkpoint 资产保存，并支持 z-score 和 1%–99% quantile 两种缩放方式，可参考其[归一化说明](https://github.com/Physical-Intelligence/openpi/blob/main/docs/norm_stats.md)与 [`Normalize/Unnormalize`](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/transforms.py)。本项目这一讲只实现 z-score；quantile normalization 会在遇到明显离群值、并能做公平对照实验时再加入。

## 四、为什么 validation 不能参与统计量计算？

考虑一个极端例子：

```text
train actions:       0, 2, 4, 6
validation actions:  100, 102
```

只看 train，均值是 `3`；把 validation 偷偷加入后，均值变成 `35.7`。训练样本会因此被映射到完全不同的位置。

即使 validation action 从未作为 loss target 输入模型，它的分布范围已经参与定义训练坐标系。这仍然是数据泄漏。评估过程回答的应该是“只使用 train 得到的模型和资产，在未见 episode 上表现如何”，所以正确顺序是：

```text
完整 episodes
  -> episode-level split
  -> 对 train episodes 做语义变换
  -> 只扫描 train 的有效 action，拟合 statistics
  -> 保存 statistics + train episode ids
  -> train 和 validation 都使用这份 statistics
```

episode 尾部的 padding 也不能参与统计。否则被重复多次的最后一个动作会被过度计数。本讲的 `RunningActionStats.update(values, valid_mask)` 会先应用第二讲生成的 mask。

## 五、训练与推理必须是镜像路径

把物理表征和数值归一化分开后，完整路径是：

```text
训练：
raw environment action
  -> representation.forward(current_state)
  -> fit/apply train-only normalization
  -> model target

推理：
model prediction
  -> denormalize with the checkpoint artifact
  -> representation.inverse(current_state)
  -> safety check / environment action
```

注意逆变换的顺序必须与训练相反。如果训练时先做 delta、再 normalization，推理时就必须先 denormalize、再把 delta 加回当前 state。

本项目的 `ActionChunk` 用两个字段阻止 normalized action 静默越过 policy 边界：

```text
normalized:       bool
normalization_id: 对应 statistics artifact 的稳定标识
```

`PolicyOutput` 会拒绝仍标记为 normalized 的 chunk。`ActionNormalizer` 还会检查 inverse 时的 artifact id，防止拿机器人 A 的统计量还原机器人 B 的输出。

## 六、跨 embodiment 还要处理补零之外的语义

不同机器人可能有 7、8 或 14 个 action 维度。为了组成 batch，可以映射到一个 canonical layout 并补齐到固定宽度，但必须同时保留：

- `action_dim_mask`：哪些维度对当前 embodiment 有意义；
- 每一维的物理含义、单位和 representation；
- `embodiment_id` 或 layout id；
- 与该 layout 匹配的 normalization artifact。

```text
canonical layout: [arm_0 ... arm_6, gripper, base_x, base_y]
arm-only robot:    [  valid 7 dims , valid,  pad  ,  pad  ]
mobile robot:      [  valid 7 dims , valid, valid, valid ]
```

补零只解决 shape，不解决语义。没有 dimension mask，模型会把不存在的维度当成“目标恰好为零”；共用一份统计量时，如果两个 layout 的同一列含义不同，归一化反而会掩盖 schema 错误。

首个 PushT demo 只有一种二维 action layout，因此本讲先冻结接口原则，不提前实现多机器人 batching。以后在 π₀.₅ 异构联合训练中真正出现多 embodiment 样本时，再加入 canonical layout 和 `action_dim_mask`，届时 flow loss 与 attention mask 都必须支持它。

## 七、配套实验：让每个变换自己证明可逆

安装项目后运行：

```bash
source .venv/bin/activate
pip install -e '.[dev]'
pi-action-transform-demo
pytest -q tests/test_action_transforms.py
```

终端首先会用同一段运动打印 absolute、current-state delta 和 velocity 三组数字，然后分别执行 inverse。两项 `round-trip max error` 都应接近零。

接着实验会比较两份统计量：

```text
train-only statistics
train + validation statistics（故意泄漏的错误示例）
```

两者均值和标准差会明显不同。最后，程序用 train-only statistics 执行：

```text
raw -> normalize -> denormalize -> restored raw
```

误差也应接近零，并打印类似 `action-zscore-xxxxxxxxxxxx` 的 artifact id。

阅读代码时建议按下面顺序：

1. [`actions.py`](../../src/pi_from_scratch/representations/actions.py)：三个可逆变换与 masked statistics；
2. [`lesson03.py`](../../src/pi_from_scratch/cli/lesson03.py)：看 toy 数据怎样暴露语义差异和 validation 泄漏；
3. [`contracts.py`](../../src/pi_from_scratch/contracts.py)：看 normalized chunk 为什么不能直接进入 runtime；
4. [`test_action_transforms.py`](../../tests/test_action_transforms.py)：看 round trip、fps、mask 和 artifact provenance 如何被锁定。

## 八、这一讲刻意没有做什么？

- 没有断言 delta 一定优于 absolute；
- 没有给 PushT 强行应用 delta；
- 没有实现 quantile normalization 或裁剪预测值；
- 没有把 normalization 塞进模型内部；
- 没有加入多 embodiment padding，因为当前数据还不需要它。

这些边界很重要。教学代码会逐步增加真正需要的选项，同时让每一次变换都有明确输入语义、逆变换和测试。

## 九、回到开头：模型预测的数字现在可以被解释了

这一讲建立了两层互不混淆的坐标系：

```text
物理坐标系：absolute / delta / velocity + space + frame + units + fps
模型坐标系：由 train-only mean/std 定义的 normalized values
```

训练 checkpoint 不应只有 model weights，还必须带上动作说明书和 normalization artifact。只有这样，一串模型输出才能沿着完全相反的路径还原成环境能够执行、工程上能够检查的命令。

下一讲我们会把单个 action 扩展为 action chunk，并进一步区分三个经常被混用的量：prediction horizon、execution horizon 和 replanning interval。那时就能严谨回答“模型预测了 16 步”究竟意味着机器人会执行多久、执行几步，以及何时重新观察环境。

## 自检问题

1. current-state delta 为什么始终使用观测时刻的 state 作为参考？
2. 为什么 velocity transform 必须知道 fps？
3. validation 只参与 mean/std、不参与 loss，为什么仍属于泄漏？
4. 为什么训练时的变换顺序与推理时的逆变换顺序相反？
5. 把 7 维 action 补零到 14 维后，至少还必须保存哪些信息？

## 扩展阅读：动作还可以怎样表示？

这一讲先处理了动作的物理坐标和数值尺度。研究工作中的“动作表征”还覆盖序列、离散 token、概率分布和 latent action。下面五组阅读分别沿着这些方向继续展开，均为选读。

### 1. Position 和 velocity 的选择会与生成模型相互影响

[Diffusion Policy](https://arxiv.org/abs/2303.04137) 对 position control 和 velocity control 做了直接比较。论文发现，Diffusion Policy 在其实验中能够更好地利用 position control，并讨论了多模态动作分布、误差累积和 action-sequence prediction 之间的关系。

建议重点阅读第 4.2 节 *Synergy with Position Control*、第 4.3 节 *Benefits of Action-Sequence Prediction* 和对应消融。阅读时可以继续追问：动作坐标系的效果是否独立于 policy head，还是会随着高斯回归、扩散模型和 flow model 改变？

### 2. 一个 action 也可以扩展成一段有结构的序列

[ACT](https://arxiv.org/abs/2304.13705) 预测未来一段 target joint positions，并用 CVAE 表达演示中的行为差异。它把 action chunk 看作一个联合生成对象，时间维度因此也进入了动作表征。

建议先读第 IV-A 节 *Action Chunking and Temporal Ensemble*，观察 chunk size 如何影响有效任务长度、反应速度和动作平滑性。第四讲会详细处理 prediction horizon、execution horizon 和 replanning interval，这里只需建立“动作表征可以包含时间结构”的印象。

### 3. 连续动作可以量化为离散 token

[RT-1](https://arxiv.org/abs/2212.06817) 第 5.1 节把每个 action 维度均匀离散成 256 个 bin，然后用分类目标预测 action token。这种逐维、逐时刻量化容易实现，也会引入量化误差和较长的 token 序列。

[FAST](https://arxiv.org/abs/2501.09747) 进一步利用动作轨迹的频域结构：先通过 DCT 把时间序列变换到频率空间，再量化和压缩。它关心的不只是一维数值落在哪个 bin，还关心整段轨迹怎样用更少 token 表示。FAST 是第 11 讲的主线，当前建议阅读摘要、方法总览和 tokenizer 的 encode/decode 图。

### 4. 跨机器人时，模型接口也可以吸收 action-space 差异

[Octo](https://arxiv.org/abs/2405.12213) 使用独立的 readout token 和轻量 action head，让 backbone 能够在微调时接入新的 action space。训练数据仍需要对齐 end-effector delta control 和 gripper 约定，论文也展示了对新 joint-position action space 的适配。

建议阅读第 III-A 节的 *Transformer backbone and readout heads* 以及第 III-B 节的数据处理。它提供了另一种思路：跨 embodiment 不一定要求所有机器人从头到尾共享完全相同的输出头，共享 backbone 与适配 action head 也能形成清晰边界。

### 5. 没有机器人 action 标签时，可以从视频学习 latent action

[LAPA](https://arxiv.org/abs/2410.11758) 用当前帧和未来帧学习离散 latent action，再用少量带机器人 action 标签的数据把 latent action 映射到可执行的 end-effector delta action。[UniVLA](https://arxiv.org/abs/2505.06111) 继续探索面向任务的 latent action，希望从不同视角和 embodiment 的视频中提取更可迁移的动作表征。

建议先读 LAPA 第 3.1 节及其 latent-action 可视化，再读 UniVLA 的方法总览。这里最值得保留的问题是：latent action 可能同时吸收手部运动、物体变化和相机运动，怎样判断它真正编码了可控制、可迁移的行为？
