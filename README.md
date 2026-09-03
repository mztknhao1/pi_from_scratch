# pi_from_scratch

![pi_from_scratch 项目封面](assets/pi-from-scratch-cover.png)

`pi_from_scratch` 是一个面向具备大模型/VLM 和基础具身知识读者的 π 系列学习项目。

我们将从一个可以在普通开发机上运行的小型模型出发，逐步理解并实现 π₀、FAST、RTC、π₀.₅、π*₀.₆、MEM 和 π₀.₇ 中的重要思想，最终把数据、训练、推理与仿真环境连接成一个完整的小型 VLA 系统。

本项目参考 [openpi](https://github.com/Physical-Intelligence/openpi)，重点是用较小、易读、可实验的代码解释核心机制；原论文的模型规模和性能不属于项目目标。

## 你会学到什么

- 一个 VLA 系统如何连接视觉、语言、机器人状态和连续动作；
- LeRobot trajectory 如何变成带 action chunk 的训练样本；
- π₀ 如何使用 action expert 和 flow matching 生成连续动作；
- FAST 如何压缩高频动作并进行自回归预测；
- RTC 如何在存在推理延迟时连续执行 action chunk；
- π₀.₅、π*₀.₆、MEM 和 π₀.₇ 如何分别引入异构训练、经验学习、长短期记忆与 context steering；
- 如何在仿真环境中评估 success、reward、延迟、轨迹连续性与失败案例。

完整学习路线见 [课程大纲](docs/00_learning_path.md)，论文之间的关系见 [π 系列知识地图](docs/01_pi_family.md)。

## 从第一讲开始

第一讲先建立完整系统观，再进入公式：

> [第 1 讲：认识 VLA 系统——机器人如何从“看见、听懂”走到“行动”](docs/lessons/01_vla_system_overview.md)

这一讲会介绍数据、VLA policy、动作生成、训练目标、runtime、仿真环境和评估之间的关系，并通过一个不学习的 random policy 跑通最小闭环。

安装并运行第一讲实验：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'

pi-contract-demo
pytest -q tests/test_contracts.py
```

第二讲开始处理真实机器人时序数据：

> [第 2 讲：从 LeRobot trajectory 得到无泄漏的训练样本](docs/lessons/02_lerobot_trajectory_to_training_sample.md)

先用无需下载的 toy episode 看清窗口边界、padding、mask 和 episode split：

```bash
pi-data-window-demo
pytest -q tests/test_data_windows.py
```

把一个真实 PushT action chunk 投影到图像上：

```bash
pip install -e '.[lerobot]'
pi-visualize-chunk --index 100 --horizon 16
```

图片默认保存到 `outputs/lesson02/pusht_chunk.png`。

第三讲继续确认 action 数字的物理语义和模型尺度：

> [第 3 讲：动作表征与归一化——模型究竟在预测什么数字？](docs/lessons/03_action_representation_and_normalization.md)

运行可逆 action transform 与 train-only normalization 实验：

```bash
pi-action-transform-demo
pytest -q tests/test_action_transforms.py
```

第四讲把 action chunk 从训练 target 接到同步执行时间线：

> [第 4 讲：Action chunk——一次预测多少步，又执行多少步？](docs/lessons/04_action_chunk_prediction_and_execution.md)

```bash
pi-chunk-execution-demo
pytest -q tests/test_chunk_execution.py
```

第五讲固定 π₀ flow matching 的 path、target velocity 和时间方向：

> [第 5 讲：让模型学会把噪声推回动作——Conditional Flow Matching](docs/lessons/05_conditional_flow_matching.md)

```bash
pi-flow-matching-demo
pytest -q tests/test_flow_matching.py
```

第六讲把图像与语言 prefix、state/action suffix 和双专家 attention 接到 flow objective：

> [第 6 讲：让 VLM 看懂场景，让 Action Expert 生成动作](docs/lessons/06_vlm_backbone_and_action_expert.md)

```bash
pi-prefix-suffix-demo
pytest -q tests/test_prefix_suffix.py tests/test_model.py
```

第七讲把前面的模块组装成可诊断训练，并用固定 flow bank 区分“能运行、能过拟合、能泛化”：

> [第 7 讲：让 TinyPi0 真正学起来——从反向传播到可诊断训练](docs/lessons/07_training_a_tiny_pi0_policy.md)

```bash
pi-training-demo
pytest -q tests/test_training.py tests/test_model.py
```

第八讲固定 checkpoint、observation 和 initial noise，比较 Euler/Heun 的 steps、NFE、误差与延迟：

> [第 8 讲：从一团噪声到 Action Chunk——把 Flow 推理看成数值积分](docs/lessons/08_flow_sampling_as_ode_integration.md)

```bash
pi-sampling-demo
pytest -q tests/test_flow_sampling.py tests/test_flow_matching.py
```

第九讲把 observation、policy、action chunk 和 environment 接成同步闭环：

> [第 9 讲：让 Policy 真正闭环——观察、规划、执行与再次观察](docs/lessons/09_closing_the_policy_environment_loop.md)

```bash
pi-closed-loop-demo --env point --policy goal --execution-horizon 3
pytest -q tests/test_closed_loop.py
```

命令会生成带 timing provenance 的 `summary.json` 和二维轨迹图。官方 PushT adapter 也已接入；建议在 Python 3.11/3.12 环境安装 `.[sim]` 后运行。

第十讲加入延迟时间线、普通异步与 RTC prefix guidance：

> [第 10 讲：RTC——模型思考时，机器人怎样继续平稳运动？](docs/lessons/10_real_time_chunking.md)

```bash
pi-rtc-demo --horizon 16 --execution-horizon 6 --delay-steps 3
pytest -q tests/test_rtc.py
```

第十一讲把连续 action chunk 变成 FAST-like 离散 token：

> [第 11 讲：FAST——怎样把高频连续动作压缩成短 token 序列？](docs/lessons/11_fast_action_tokenizer.md)

```bash
pi-fast-tokenizer-demo
pytest -q tests/test_fast.py
```

## 代码目录

代码按职责分层，课程命令只负责把这些模块串起来：

```text
src/pi_from_scratch/
├── data/             # LeRobot adapter、episode split、时间窗口与 mask
├── representations/  # 文本与动作表示；后续加入 action transform、FAST tokenizer
├── models/           # tiny π₀ 等模型结构
├── objectives/       # flow matching 等训练目标
├── training/         # split、normalization、训练循环、固定评估与 checkpoint
├── evaluation/       # loss curve 与离线 action trajectory 可视化
├── inference/        # Euler/Heun、flow sampling；不管理环境时钟
├── envs/             # dependency-free 闭环环境与可选 PushT adapter
├── policies/         # 面向 runtime 的统一 policy 接口
├── runtime/          # 同步 chunk 执行、延迟模拟与 RTC runtime
├── cli/              # 各讲实验、训练和可视化命令入口
├── config.py         # 当前 Python 配置对象
└── contracts.py      # 跨模块共享的 observation/action 契约
```

异步 runtime、memory 和更多 environment adapter 会在对应课程真正需要时加入。

## 运行最小训练

仓库目前提供一个小型 PyTorch π₀ 风格模型，包括：

- 图像、文本和低维 state 条件编码；
- action chunk；
- conditional flow-matching loss；
- π₀ 论文时间约定：`τ=0` noise，`τ=1` action；
- Euler/Heun action sampling 与固定 noise sweep；
- synthetic dataset 和自动测试。

无需下载机器人数据即可运行：

```bash
pi-train --dataset synthetic --steps 20 --device cpu
pytest -q
```

训练完成后，checkpoint 会保存在 `outputs/debug/`。

## LeRobot PushT 数据

项目选择 [`lerobot/pusht`](https://huggingface.co/datasets/lerobot/pusht) 作为第一套真实数据。它体积较小、使用连续二维动作，适合检查数据窗口、action chunk、训练和闭环控制。

安装 LeRobot 并校验数据接口：

```bash
pip install -e '.[dev,lerobot]'
pi-train --dataset lerobot/pusht --steps 5 --device cuda
```

LeRobot 的数据读取能力属于额外依赖；本项目的 `lerobot` extra 已包含官方
`lerobot[dataset]`。当前版本要求 Python 3.12 或更高版本。

目前 PushT 接口用于管线 smoke test；episode split 与 train-only action normalization 已接入训练 CLI。state normalization、显式 action representation transform 和闭环环境评估仍未加入，因此不应使用上述 5-step 结果评价策略性能。

PushT 只有一个固定任务，适合验证连续控制机制，但不能证明语言泛化、开放世界能力或分钟级记忆。后续高级章节会使用受控实验，并预留一个小型 LIBERO 多任务扩展。

## 当前内容

```text
图像 + 语言 + 本体状态
          ↓
      条件编码
          ↓
     action expert
          ↓
   flow matching action chunk
```

当前仓库已经包含前十一讲正文，以及 observation/action 契约、episode-safe action window、可逆动作变换、TinyPi0 训练、Euler/Heun sampling、闭环 runner、RTC 和 FAST-like tokenizer。后续内容会按照 [课程大纲](docs/00_learning_path.md) 逐步加入。

## 项目边界

这是一个教学实现。小模型和小数据实验用于理解算法、检查接口和比较机制，不能代表 Physical Intelligence 原始模型在大规模跨机器人数据上的能力。
