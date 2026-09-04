# pi_from_scratch 课程大纲

我也在学习 π 系列。我的起点是具身微调经验和大模型/VLM 基础，但对 π 系列论文中的不少实现细节还缺少系统理解。这个仓库希望通过“从零手写、逐讲验证、最终组装”的方式，把论文概念变成可以检查的代码和实验。

本文件定义课程结构和交付顺序，各讲正文放在 `docs/lessons/`，目标代码架构见 [`03_architecture_blueprint.md`](03_architecture_blueprint.md)。

## 课程定位

目标读者已经具备：

- Transformer、attention、tokenizer、VLM 和常规微调知识；
- PyTorch 训练、checkpoint、混合精度等基本工程经验；
- 机器人 observation/action、模仿学习和闭环控制的基础概念。

因此课程不重复讲解反向传播、AdamW、Transformer 入门或 Python 基础。篇幅集中在 VLA 特有的问题：机器人时序数据、动作表征、action chunk、flow matching、动作 tokenization、VLM/action expert 耦合、实时推理、异构联合训练、经验学习、具身记忆和 context steering。偶尔需要但不属于主线的知识，只给出可靠外部资料和“读到什么程度”的说明。

## 两条相互交叉的主线

### 模型与训练线

```text
LeRobot trajectory
  -> aligned training sample
  -> action representation / normalization
  -> action chunk
  -> π₀ conditional flow matching
  -> FAST autoregressive action tokens
  -> π₀.₅ heterogeneous co-training
  -> MEM short-term video + long-term text memory
  -> π*₀.₆ experience + advantage conditioning
  -> π₀.₇ diverse context conditioning
```

### 推理与执行线

```text
flow ODE sampling
  -> chunk execution
  -> receding-horizon closed loop
  -> latency and asynchronous inference
  -> RTC action-prefix inpainting
  -> training-time RTC
  -> simulator latency sweep
```

两条线共享 observation/action contract，但模型不负责控制时钟和 action buffer。RTC 属于 policy runtime，不应被实现为 π₀.₅ 或 π₀.₇ 内部的特殊分支。

### 论文改进快线

希望尽快进入 π 系列后续工作的读者，完成第 1–8 讲后按下面的顺序阅读：

```text
第 9 讲：用最小闭环冻结 runtime 接口
  -> 第 10 讲：RTC
  -> 第 11 讲：FAST
  -> 第 12 讲：π₀.₅
  -> 第 13 讲：MEM
```

第 9 讲只补齐这些工作共同依赖的 observation、timestamp、buffer 和 environment 边界。π*₀.₆、π₀.₇ 与最终组装随后继续，主线中不再插入通用大模型基础课。

## 课程大纲

采用“每讲解决一个主要问题”的组织方式。讲数服从内容边界，每个问题都要清楚且可验证。

### 第一部分：定义问题与数据契约

#### 第 1 讲：认识 VLA 系统——机器人如何从“看见、听懂”走到“行动”

唯一问题：一个完整 VLA 系统由哪些部分组成，它们为什么存在，又如何把观测、语言和动作连接成闭环？

交付：建立数据、policy、训练目标、runtime、环境和评估的全景；再用统一 contract 与不学习的 random policy 跑通最小系统。

正文：[`lessons/01_vla_system_overview.md`](lessons/01_vla_system_overview.md)

#### 第 2 讲：从 LeRobot trajectory 得到无泄漏的训练样本

唯一问题：如何从按 episode 存储的多模态时序数据，按 timestamp/fps 对齐 observation 与未来 action window？

交付：PushT 数据检查器、episode split、窗口边界 padding/mask、一个 batch 可视化。

正文：[`lessons/02_lerobot_trajectory_to_training_sample.md`](lessons/02_lerobot_trajectory_to_training_sample.md)

#### 第 3 讲：动作表征与归一化决定模型学的是什么

唯一问题：absolute、delta、velocity、joint/end-effector action 以及跨 embodiment 维度如何进入统一训练接口？

交付：可逆 action transform、仅由 train split 计算的统计量、round-trip 与分布检查。

正文：[`lessons/03_action_representation_and_normalization.md`](lessons/03_action_representation_and_normalization.md)

#### 第 4 讲：Action chunk 改变了训练目标，也改变了执行方式

唯一问题：为什么预测未来 `H` 步，以及 horizon、control frequency、execution horizon 之间是什么关系？

交付：单步与 chunk 数据样本对比；同步 chunk executor；边界连续性指标。

正文：[`lessons/04_action_chunk_prediction_and_execution.md`](lessons/04_action_chunk_prediction_and_execution.md)

### 第二部分：π₀ 的训练与推理

#### 第 5 讲：Conditional flow matching 如何学习连续动作分布

唯一问题：怎样从 action、noise、time 构造 vector-field regression，并保证训练和采样的时间方向一致？

交付：最小 flow objective、解析 toy test、tiny-set overfit。

正文：[`lessons/05_conditional_flow_matching.md`](lessons/05_conditional_flow_matching.md)

#### 第 6 讲：π₀ 为什么需要 VLM backbone 和 action expert

唯一问题：语义 observation prefix 如何条件化连续 action suffix，同时避免破坏预训练 VLM 的职责边界？

交付：小模型结构图、attention mask 检查、与 openpi 张量接口的逐项映射。

正文：[`lessons/06_vlm_backbone_and_action_expert.md`](lessons/06_vlm_backbone_and_action_expert.md)

#### 第 7 讲：训练一个小型 π₀ 风格 policy

唯一问题：把数据、条件编码、flow loss、优化器和 checkpoint 组装后，怎样判断模型确实学到了，同时排除管线只达到“能跑”的情况？

交付：固定配置训练、过拟合基准、validation loss、预测轨迹可视化和失败诊断。

正文：[`lessons/07_training_a_tiny_pi0_policy.md`](lessons/07_training_a_tiny_pi0_policy.md)

#### 第 8 讲：Flow policy 的推理是一个数值积分问题

唯一问题：如何从噪声生成 action chunk，以及 solver steps、速度、误差和随机性如何权衡？

交付：Euler sampler；固定 noise 的可复现实验；sampling steps/latency/误差曲线。

正文：[`lessons/08_flow_sampling_as_ode_integration.md`](lessons/08_flow_sampling_as_ode_integration.md)

### 第三部分：闭环控制与实时推理

#### 第 9 讲：从离线预测走到 simulator 闭环

唯一问题：policy inference、环境 step、控制频率、action buffer 和重规划应如何解耦？

交付：同步、阻塞式 action-chunk runner；dependency-free 二维闭环验收；复用同一接口的 PushT adapter；success/reward/timing/trajectory 记录。

正文：[`lessons/09_closing_the_policy_environment_loop.md`](lessons/09_closing_the_policy_environment_loop.md)

#### 第 10 讲：RTC 如何在推理延迟下保持动作连续

唯一问题：异步生成新 chunk 时，如何冻结已经确定会执行的 action prefix，并用 flow inpainting 补全未来动作？

交付：先实现可注入固定延迟的离散事件 runtime，对比 blocking、naive async、RTC 三种策略的吞吐、观测年龄、边界跳变和 jerk；再加入 training-time RTC 作为独立扩展。延迟抖动、deadline miss 分位数和任务成功率留到部署实验继续补全。

正文：[`lessons/10_real_time_chunking.md`](lessons/10_real_time_chunking.md)

### 第四部分：π 系列的表示与训练改进

#### 第 11 讲：FAST 如何把高频连续动作变成更短的 token 序列

唯一问题：DCT、量化和压缩式 tokenization 为什么比逐维分桶更适合 action chunk？

交付：FAST-like tokenizer 的 encode/decode；重建误差、token 数、训练吞吐对比；flow 与 autoregressive policy 使用同一数据和评估协议。

正文：[`lessons/11_fast_action_tokenizer.md`](lessons/11_fast_action_tokenizer.md)

#### 第 12 讲：π₀.₅ 如何联合机器人动作和高层语义数据

唯一问题：异构样本如何共享模型并采用离散预训练、flow 后训练的 hybrid recipe？

交付：typed mixture sample、objective routing、sampling ratio 记录；小规模 semantic subtask/context 实验；用可测的遗忘约束落实 knowledge insulation。

正文：[`lessons/12_pi05_heterogeneous_cotraining.md`](lessons/12_pi05_heterogeneous_cotraining.md)

专题附录：[`appendices/a_knowledge_insulation.md`](appendices/a_knowledge_insulation.md) 展开解释 Knowledge Insulation 的中文含义、研究动机、joint-training loss、attention 内的 stop-gradient，以及它与冻结 backbone 的区别。

#### 第 13 讲：MEM 如何让 VLA 同时拥有短期与长期记忆

唯一问题：长时程任务中，怎样既保留最近视觉细节以应对遮挡，又用紧凑语义记忆追踪分钟级任务进度？

交付：短期视频窗口与压缩编码、长期文本 memory state、memory update/read 接口；在受控 partial-observability 与多阶段任务中比较无记忆、单尺度记忆和多尺度记忆。

正文：[`lessons/13_multiscale_embodied_memory.md`](lessons/13_multiscale_embodied_memory.md)

#### 第 14 讲：π*₀.₆ / RECAP 如何从部署经验和纠正中学习

唯一问题：如何让 demonstration、autonomous rollout、failure 和 intervention 通过 value/advantage 信号共同改进 policy？

交付：带 reward/source/intervention 的 episode schema；offline value/advantage pipeline；advantage-conditioned policy 对照实验。真实机器人在线 RL 不属于首版范围。

正文：[`lessons/14_recap_learning_from_experience.md`](lessons/14_recap_learning_from_experience.md)

#### 第 15 讲：π₀.₇ 如何用多模态 context 控制策略

唯一问题：除 task command 外，怎样让 performance metadata、strategy、subgoal image 等上下文改变 policy 行为，并明确缺失 context 时的语义？

交付：统一 context schema、context dropout、subgoal-image condition、同任务不同策略的可控性对比。

### 第五部分：组装与复盘

#### 第 16 讲：组装可复现的仿真 demo

唯一问题：怎样用同一份配置和评估协议，把数据、训练、推理、runtime 与 simulator 组装成别人能复现的小系统？

交付：一条命令训练、一条命令评估、一份结果表和失败视频；至少比较 flow baseline、FAST-like decoder、blocking runtime 与 RTC。高级的 π₀.₅–π₀.₇ 概念以小型受控实验验证，不宣称复现论文规模能力。

## 工程里程碑

“讲”服务于理解，“M”表示代码仓的可运行状态，两套编号承担不同职责。

| 里程碑 | 覆盖讲次 | 仓库必须达到的状态 |
|---|---:|---|
| M0 | 1 | synthetic 数据能完成 loss、backward、checkpoint；接口初版冻结 |
| M1 | 2–4 | PushT batch、统计量、action transform 和 chunk mask 可检查 |
| M2 | 5–9 | 小型 π₀ 风格 policy 能训练、采样；同步闭环与 PushT adapter 可检查 |
| M3 | 10 | 延迟注入、异步 runtime、RTC 与实时性指标可复现实验 |
| M4 | 11 | FAST-like tokenizer 和 autoregressive baseline 能公平对比 |
| M5 | 12 | π₀.₅ 风格 mixed objective 与语义/遗忘检查 |
| M6 | 13 | MEM 短期视频/长期文本记忆与 partial-observability 实验 |
| M7 | 14 | π*₀.₆ 风格 offline experience/advantage 实验 |
| M8 | 15–16 | π₀.₇ context 实验和统一仿真 demo |

原先的 M0–M6 增加为 M0–M8：RTC 有独立的 runtime 与实时性验收，MEM 也有独立的 memory state、更新机制与长时程验证；二者都不应作为其他模型版本下的一条附注。

## 数据与仿真阶梯

### 必修快线

- 数据：`lerobot/pusht`。
- 环境：LeRobot PushT simulator。
- 验证：数据对齐、连续动作生成、action chunk、flow/FAST、闭环控制、延迟与 RTC。

### 高级概念的小型受控实验

PushT 不具备足够的语言和长时程任务多样性，不能证明 π₀.₅–π₀.₇ 的开放世界能力，也无法直接验证 MEM 的分钟级记忆。高级章节先用明确标注为 toy/controlled 的 context、reward、subgoal 和 partial-observability 变体验证机制是否接通。

### 可选扩展线

后续选择一个小型 LIBERO 子集验证真实的多任务语言条件。它是扩展实验，不作为所有读者跑通核心课程的前置条件。

## 不在首版范围内

- 从头预训练数十亿参数 VLM；
- 复现 Physical Intelligence 的私有数据混合与论文绝对指标；
- 完整 DROID 或跨机器人万小时预训练；
- 真实机器人在线 RL 和无安全约束的部署；
- 为了“覆盖大模型基础”而重写已有的通用教程。
