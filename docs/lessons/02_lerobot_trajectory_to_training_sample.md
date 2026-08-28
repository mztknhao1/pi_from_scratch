# 第 2 讲：从 LeRobot trajectory 得到无泄漏的训练样本

> 第一讲画出了 VLA 系统的全景。这一讲开始拧紧第一颗螺丝：模型在训练时看到的一个 sample，究竟是怎样从机器人 trajectory 中切出来的？

## 从一个不会报错的数据 bug 说起

假设数据集中有两个 episode：

```text
episode 0: frame 0, 1, 2, ..., 99
episode 1: frame 0, 1, 2, ..., 84
```

我们希望在 episode 0 的 frame 98 构造一个长度为 `H=4` 的未来动作窗口。如果直接把所有 frame 拼成一个大数组，再写：

```python
actions[98 : 98 + 4]
```

得到的可能是：

```text
episode 0 的 action 98
episode 0 的 action 99
episode 1 的 action 0
episode 1 的 action 1
```

代码没有越界，tensor shape 也完全正确，训练甚至可以正常下降。但对模型来说，这段监督表示“机器人执行到 episode 0 末尾后，瞬间跳到 episode 1 的起点”。

这类错误比程序崩溃更危险，因为它会悄悄改变模型学习到的动作分布。

所以第二讲只解决一个问题：

> 如何从按 episode 存储的多模态轨迹中，按时间对齐 observation 与未来 action window，同时不跨越 episode 边界，也不让相邻帧泄漏到 validation？

完成这一讲后，我们应该能够解释并验证四件事：

1. 一个 VLA 训练 sample 的时间语义是什么；
2. `fps`、timestamp、frame index 和 action horizon 如何共同确定窗口；
3. episode 末尾为什么需要 padding 和 mask；
4. 为什么 train/validation 必须按 episode 划分。

## 一、一个训练 sample 到底在问模型什么？

最简单的 VLA 行为克隆样本可以写成：

$$
x_t = (I_t, s_t, l), \qquad
A_t = [a_t, a_{t+1}, \ldots, a_{t+H-1}]
$$

其中：

- $I_t$ 是时刻 `t` 的图像；
- $s_t$ 是同一时刻的机器人状态；
- $l$ 是任务语言；
- $A_t$ 是从当前时刻开始的未来动作窗口。

它表达的问题不是“这张图对应哪个动作标签”，而是：

> 机器人在这个时刻看到 `I_t`、处于状态 `s_t`、需要完成任务 `l`，接下来一小段时间应该怎样行动？

因此，样本是否正确不仅取决于 tensor shape，还取决于图像、state、action 是否来自同一个 episode，以及**它们的时间关系是否符合约定**。

## 二、LeRobot 保存的是 trajectory，不是现成的 action chunk

LeRobot 用 episode 组织机器人时序数据。表格数据通常保存在 Parquet 中，图像序列可以保存在 MP4 中，`meta/info.json` 则描述数据频率、特征名称、shape 和数据版本。读取时，`LeRobotDataset` 把这些存储细节重新组合成按 frame 索引的字典。

以本项目使用的 [`lerobot/pusht`](https://huggingface.co/datasets/lerobot/pusht) 为例，Hub 页面当前列出的主要信息包括：

| 字段 | 含义 |
|---|---|
| `fps = 10` | 相邻 frame 的标称间隔是 0.1 秒 |
| `observation.image` | `96 × 96 × 3` 的图像 |
| `observation.state` | 二维状态 |
| `action` | 二维连续动作 |
| `episode_index` | 当前 frame 属于哪个 episode |
| `frame_index` | 当前 frame 在 episode 内的位置 |
| `timestamp` | 当前 frame 在 episode 内的时间 |

具体数字可能随着数据 revision 更新，因此代码应当读取 metadata，而不是把 `10 Hz`、episode 数量或 feature shape 永久写死。LeRobot 的[官方数据接口](https://huggingface.co/docs/lerobot/main/api/datasets)也把 `fps` 和 `delta_timestamps` 作为时间窗口查询的核心字段。

这里有一个重要边界：LeRobot 中的一行通常描述一个时间 frame，而 `action chunk` 是训练数据适配器根据 horizon 临时构造出来的视图。我们不需要把每个重叠的 chunk 再存一遍。

## 三、从时间定义窗口，而不是从数组长度猜窗口

假设数据频率是 10 Hz，动作 horizon 是 `H=5`。从当前 frame 开始，我们需要的 action 相对时间为：

```text
0.0 s, 0.1 s, 0.2 s, 0.3 s, 0.4 s
```

对应的 LeRobot 查询写法是：

```python
delta_timestamps = {
    "action": [step / metadata.fps for step in range(H)]
}
```

LeRobot 会检查这些时间差是否与数据集 `fps` 对齐，并返回 shape 为 `[H, action_dim]` 的动作窗口。官方实现也采用“离散 index 除以数据集 fps”的方式构造 action `delta_timestamps`。

需要区分两个看起来相近的时间量：

- 从第一个动作时间戳到最后一个动作时间戳，相差 `(H - 1) / fps`；
- 如果每个命令保持一个控制周期，那么 `H` 个动作覆盖约 `H / fps` 的执行时间。

例如 `H=5, fps=10` 时，时间戳从 `0.0 s` 到 `0.4 s`，但五个控制周期一共覆盖约 `0.5 s`。后续讨论 execution horizon 和重采样时，我们会继续使用这个约定。

## 四、episode 末尾缺少未来动作时怎么办？

越接近 episode 末尾，可用的未来动作越少。但神经网络训练通常希望一个 batch 中的 action tensor 具有相同 shape，因此需要 padding。

常见策略有三种：

1. 丢弃所有不够一个完整 horizon 的 anchor；
2. 用零补齐；
3. 重复 episode 最后一个动作，并提供 padding mask。

本仓选择第三种：

```text
episode actions:       [a0, a1, a2]
anchor = 1, H = 4

window values:         [a1, a2, a2, a2]
valid mask:            [ T,  T,  F,  F]
```

重复末尾动作只是为了让 tensor 保持固定 shape，并不意味着模型应该学习这些补齐值。因此 loss 必须只统计 `valid_mask=True` 的位置：

$$
\mathcal{L}
=
\frac{\sum_{b,h}m_{b,h}\,\ell(\hat a_{b,h},a_{b,h})}
{\sum_{b,h}m_{b,h}}
$$

只有 padding、没有 mask，是不完整的实现。模型可能学到“越接近 episode 末尾，越应该重复某个默认动作”；用零补齐时，还可能把零误解成真实控制命令。对 Transformer action expert，mask 不仅要作用于 loss，还应作为 attention padding mask，防止有效动作 token 读取补齐位置。

当前代码在 [`data_windows.py`](../../src/pi_from_scratch/data_windows.py) 中显式返回：

```text
values
valid_mask
timestamps_s
source_indices
```

`source_indices` 主要用于教学和调试：它能让我们直接看到 padding 是否仍然引用当前 episode 的最后一个 frame，而不是下一个 episode 的开头。

## 五、为什么必须先按 episode 划分，再生成窗口？

机器人轨迹中的相邻 frame 高度相似。如果随机把所有 frame 分到 train 和 validation，很可能出现：

```text
train:      episode 12, frame 40
validation: episode 12, frame 41
```

这两个样本的图像几乎相同，未来 action window 也大量重叠。validation loss 看起来很好，并不代表模型能泛化到一条没有见过的 rollout。

正确顺序是：

```text
读取 episode id
    ↓
划分 train episodes / validation episodes
    ↓
分别加载两个 episode 子集
    ↓
在各自子集内部构造时间窗口
```

本讲的 `split_episode_ids(...)` 保证两个集合互斥，并使用固定 seed 复现划分。PushT 只有一个任务，简单的 episode-level random split 足够用于入门实验；未来处理多任务、多机器人数据时，还需要按 task、embodiment 或采集场景做分层划分。

还要注意：归一化统计量只能由 train episodes 计算。否则即使窗口没有交叉，validation action 的分布信息仍然会泄漏到训练过程。归一化会在下一讲单独实现。

## 六、把一个 batch 展开检查

在真正训练前，我们至少应该打印或可视化一次 batch，而不是只看 DataLoader 有没有报错。

本讲希望看到的核心 shape 是：

```text
image:          [B, C, height, width]
state:          [B, state_dim]
actions:        [B, H, action_dim]
action_mask:    [B, H]
```

对每个 sample 还应检查：

- observation 和 action window 的 `episode_index` 相同；
- action timestamp 严格递增；
- 有效 mask 是连续前缀，即不出现 `[True, False, True]`；
- padding source index 没有越过 episode 末尾；
- train 和 validation 的 episode id 没有交集。

图像可视化很有用，但时间窗口最好先画成表格，因为表格能同时显示 slot、source frame、timestamp、mask 和 action value。本讲的终端实验就是这样做的。

## 七、配套实验：故意站在 episode 边界旁边

安装项目后运行：

```bash
source .venv/bin/activate
pip install -e '.[dev]'
pi-data-window-demo
pytest -q tests/test_data_windows.py
```

实验构造两个数值范围明显不同的 toy episode：episode 0 的动作接近 `0`，episode 1 的动作接近 `100`。然后从 episode 0 倒数第二个位置切窗口。

你会看到类似结果：

```text
slot | source frame | timestamp | valid | action
-----+--------------+-----------+-------+----------------
   0 |            2 |     0.20s |  True | [2.00, 1.00]
   1 |            3 |     0.30s |  True | [3.00, 1.50]
   2 |            3 |     0.40s | False | [3.00, 1.50]
   3 |            3 |     0.50s | False | [3.00, 1.50]
```

最重要的不是重复值本身，而是窗口中永远不会出现 episode 1 的 `100.xx`，并且补齐位置全部被标记为无效。

准备好 LeRobot 依赖后，可以检查真实 PushT 数据：

```bash
pip install -e '.[dev,lerobot]'
pi-data-window-demo --dataset lerobot/pusht --horizon 16 --index -2
```

这里安装的不只是基础 `lerobot`，还包括官方 `dataset` extra 提供的
`datasets`、视频解码等依赖；当前版本要求 Python 3.12 或更高版本。
本讲显式使用 PyAV 解码视频，以避免教学入口依赖系统安装的 FFmpeg 动态库；这不会改变 action window 的时间语义。
`--index -2` 会选择数据集倒数第二个 frame，便于观察 episode 尾部 padding。第一次运行可能需要从 Hugging Face 下载 metadata、表格和视频数据。

阅读代码时建议按下面顺序：

1. [`data_windows.py`](../../src/pi_from_scratch/data_windows.py)：看 episode split 和窗口构造；
2. [`lesson02.py`](../../src/pi_from_scratch/lesson02.py)：看 toy/LeRobot 检查器怎样把字段画成表格；
3. [`data.py`](../../src/pi_from_scratch/data.py)：看 LeRobot 字段怎样适配到当前训练接口；
4. [`model.py`](../../src/pi_from_scratch/model.py)：看 `action_mask` 怎样进入 attention 和 loss；
5. [`test_data_windows.py`](../../tests/test_data_windows.py)：看哪些时间边界被自动测试锁住。

## 八、把一个 action chunk 画到图像上

表格适合检查边界，但它不能让我们直观看到一段 action chunk 在空间中怎样运动。根据 [gym-pusht 环境定义](https://github.com/huggingface/gym-pusht)，PushT 的 action 是工作区 `[0, 512] × [0, 512]` 内的二维绝对目标位置，而数据图像是 `96 × 96`，因此可以按宽高比例把 action 投影到图像像素坐标。

运行：

```bash
pi-visualize-chunk \
  --dataset lerobot/pusht \
  --index 100 \
  --horizon 16 \
  --output outputs/lesson02/pusht_chunk.png
```

生成的图片包含两部分：

- 左侧在 chunk 起点图像上绘制完整未来轨迹，白色圆环表示当前 state，青色到橙色表示从早到晚的 action target；
- 右侧从 chunk 中等间隔抽取六张未来图像，并分别标出同一时刻的 action target。

如果投影约定正确，右侧 action marker 应大致落在每张图的蓝色圆形执行器上。这个检查非常有价值：shape、mask 和 loss 都可能正常，但错误的 x/y 顺序、图像缩放或坐标方向会让投影点系统性偏离机器人。

可视化代码在 [`visualize_chunk.py`](../../src/pi_from_scratch/visualize_chunk.py)。这里的线性投影是 PushT 特有的，因为它的 action 与俯视图共享同一个二维工作区。真实机械臂若要把末端三维轨迹画到相机图像上，需要相机内参、相机外参以及坐标系变换，不能直接照搬这个比例缩放。

## 九、这一讲刻意没有做什么？

为了让一个问题保持清楚，本讲暂时不处理：

- action 是绝对位置、增量还是速度；
- state 和 action 是否需要不同的归一化策略；
- 不同 embodiment 的 action 维度如何对齐；
- 多帧 observation history；
- 图像增强和大规模视频解码性能。

其中前三项是下一讲的主线。当前窗口中的 action 仍保留数据集原始数值，mask 只回答“这个位置是否存在真实监督”，不回答“这个数值在物理上代表什么”。

## 十、回到开头：一个可靠 sample 是怎样形成的？

现在可以把本讲压缩成一条数据路径：

```text
LeRobot metadata
  -> 读取 fps、feature schema 和 episode 边界
  -> 按 episode 划分 train / validation
  -> 选取时刻 t 的 observation
  -> 按 delta_timestamps 查询未来 H 步 action
  -> 在 episode 末尾 padding，并生成 valid mask
  -> 组成固定 shape 的 batch
  -> loss 只使用有效动作
```

这一讲真正建立的不是一个切片函数，而是一条原则：

> VLA 数据中的每个 tensor 都必须保留它来自哪条 episode、对应哪个机器人时刻，以及哪些位置是真实观测或监督。

如果这些信息在进入模型前丢失，后面再复杂的 VLM、flow matching 或 RTC 都只能在错误的时间关系上学习。

## 自检问题

1. 为什么随机划分 frame 会让 validation loss 过于乐观？
2. `H=16, fps=10` 时，动作时间戳跨度和控制覆盖时长分别是多少？
3. 为什么重复最后一个动作仍然必须提供 mask？
4. `delta_timestamps` 为什么使用秒，而不是直接只传 frame index？
5. 如果 validation episode 参与了 normalization statistics 计算，还算不算无泄漏？

下一讲我们会继续追问：窗口已经对齐了，但其中的二维 action 究竟是位置、位移还是速度？如果不先回答这个问题，归一化只是把含义不清楚的数字缩放到了另一个范围。
