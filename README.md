# pi_from_scratch

![pi_from_scratch 项目封面](assets/pi-from-scratch-cover.png)

一个面向具备大模型/VLM 与基础具身知识读者的迷你 π 系列训练框架。

它参考 [openpi](https://github.com/Physical-Intelligence/openpi) 的关键设计，但不会复刻工业级代码，也不会声称能用小数据、小模型复现论文结果。第一阶段只保留最重要的闭环：

```text
图像 + 语言 + 本体状态 -> 条件编码 -> action expert -> action chunk
                                            ^
                                      flow matching
```

当前里程碑 `M0` 已包含：

- 一个纯 PyTorch 的小型 π₀ 风格模型；
- action chunk 的 conditional flow-matching 训练和 Euler 采样；
- 无需下载数据的 synthetic smoke test；
- `lerobot/pusht` 适配器入口；
- 论文演进地图和后续实现边界。

## 先跑通，再理解

要求 Python 3.11+。在本目录执行：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pi-train --dataset synthetic --steps 20 --device cpu
pytest
```

校验 `lerobot/pusht` 的下载与 batch 管线（会从 Hugging Face 下载数据）：

```bash
pip install -e '.[dev,lerobot]'
pi-train --dataset lerobot/pusht --steps 5 --device cuda
```

PushT 适合验证数据、模型、loss、反向传播和保存 checkpoint 的完整链路，但它只有一个固定任务，不能检验真正的语言泛化。当前 M0 也还没有接入训练集统计量归一化，因此这条命令只是接口 smoke test，不应拿 loss 或策略效果做结论。归一化和 episode split 是 M1 的第一项任务。

## 课程设计

目前先冻结大纲、模块边界和统一验收方式，再逐讲编写正文与实现：

1. [`docs/00_learning_path.md`](docs/00_learning_path.md)：16 讲课程大纲，以及训练线和推理/runtime 线。
2. [`docs/01_pi_family.md`](docs/01_pi_family.md)：π₀ → FAST → π₀.₅ → π*₀.₆ → MEM → π₀.₇ 的核心变化。
3. [`docs/03_architecture_blueprint.md`](docs/03_architecture_blueprint.md)：目标代码架构、接口、PushT demo 与 RTC 验证蓝图。
4. [`docs/lessons/01_vla_system_overview.md`](docs/lessons/01_vla_system_overview.md)：第一讲正式内容，从完整 VLA 系统全景和 random policy probe 开始。

当前代码仍是 M0 可运行骨架，后续会按蓝图渐进迁移，不做一次性重写。

## 项目原则

- 每个里程碑必须有 CPU smoke test。
- 论文概念先写最小实现，再替换成预训练大模型。
- 数据格式、模型结构、训练目标分层，避免把机器人差异写死在模型里。
- 训练、采样、runtime 和 simulator 分层；RTC 是 runtime/inference 能力，不写死在模型版本中。
- 每个“简化”都要在文档里说清楚，避免教学代码被误认为论文复现。

## 状态

`M0` 是可审阅的第一版骨架。课程大纲和验收约束已先行定义；下一步从第 1–2 讲的接口与 LeRobot 数据检查工具开始，而不是立即扩写整套讲义。
