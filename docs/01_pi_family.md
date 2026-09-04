# π 系列核心知识地图

| 阶段 | 要解决的问题 | 核心方法 | 本仓对应模块 |
|---|---|---|---|
| π₀ | VLM 如何输出高频连续机器人动作 | 预训练 VLM + 独立 action expert；用 flow matching 生成 action chunk；跨机器人数据训练 | M2 |
| FAST | 自回归 VLA 的逐维离散 action token 太长且效果差 | action chunk 归一化后做 DCT，在频域量化并压缩成 token；FAST+ 是通用 tokenizer | M4 |
| Hi Robot | 机器人如何理解复杂、多阶段指令，并在执行中接收人的反馈和纠正 | 分层推理：高层 VLM 根据画面、复杂 prompt 和实时反馈生成低层语言指令，低层 π₀ VLA 执行动作 | M5（高低层接口） |
| π₀.₅ | 如何获得开放世界、长时程泛化 | 多机器人动作、web/VL、高层语义任务联合训练；混合离散预训练与 flow 后训练 | M5 |
| Knowledge Insulation | 随机初始化的 continuous action expert 如何避免干扰预训练 VLM | 联合训练 FAST action token、VLM 数据和 continuous action；在 action expert 读取 backbone 的 attention 路径上停止 flow gradient | M5（附录 A） |
| RTC | 高延迟 action-chunk policy 如何连续、异步执行 | 推理时冻结已提交 action prefix，并用 flow inpainting 补全未来；后续工作把 prefix conditioning 移到训练期 | M3 |
| π*₀.₆ | 如何从部署经验、失败和人工纠正中继续变强 | RECAP：value/advantage 学习与 advantage-conditioned policy，采用分批采集、离线更新、再次部署的迭代 | M7 |
| MEM | 长时程任务如何同时记住近期视觉细节和长期语义进度 | 视频编码器压缩短期视觉历史；文本记忆概括长期事件与已完成阶段 | M6 |
| π₀.₇ | 如何让通用策略可控，并利用质量不一的异构数据 | diverse context conditioning：除命令外再条件化策略、表现 metadata、subgoal images 等 | M8 |

## 1. π₀：连续动作的 flow matching

给定真实 action chunk `a`、高斯噪声 `ε` 和论文时间 `τ∈[0,1]`：

```text
x_τ = (1 - τ) ε + τ a
target velocity = a - ε
loss = ||v_θ(x_τ, τ, observation) - (a - ε)||²
```

全仓采用 π₀ 论文的时间约定：`τ=0` 是噪声，`τ=1` 是动作。推理从高斯噪声出发，用数值积分从 0 走到 1。openpi 源码使用互补变量 `t=1-τ`，对照源码时需要同时翻转 velocity 符号。

本仓的简化：论文使用预训练 VLM 和大 action expert；M0 使用小 CNN、哈希词表 embedding 和小 Transformer。保留算法骨架，不保留规模能力。

## 2. FAST：动作也可以是 token

朴素方法把每个时间步、每个 action 维度单独分桶，会产生很长的 token 序列，并忽视轨迹在时间上的平滑结构。FAST 对一段动作做 DCT，把能量集中的频率系数量化，再用压缩式序列表示减少 token 数。它把连续动作生成问题变成标准 next-token prediction。

必须分别测量 tokenizer 重建误差与 policy 闭环性能；重建好不等于控制好。

## 3. π₀.₅：异构联合训练

关键不只是“更大的 π₀”，而是训练样本类型发生了变化：机器人低层动作、高层子任务/语义预测、图文与 web 数据可以在同一训练体系中迁移知识。公开描述还强调 hybrid recipe：预训练阶段包含离散 token 目标，后训练阶段使用 flow matching action expert。

## 4. π*₀.₆：从经验和纠正中学习

behavior cloning 只模仿数据，无法天然区分成功、失败和更优策略。RECAP 将 reward feedback、autonomous rollouts 和 human interventions 纳入训练，通过 value/advantage 估计来条件化 policy。教学实现应先离线验证 advantage 标签和加权/条件化逻辑，再做在线机器人闭环。

## 5. MEM：记忆需要不同的时间尺度和模态

简单地把所有历史图像拼进 context，会随着任务增长变得昂贵，也难以同时保留局部细节和高层进度。MEM 将两种记忆结合：短期视频记忆保留最近的视觉与状态变化，用于遮挡和局部操作；长期文本记忆压缩已经发生的语义事件，用于追踪多阶段任务进度。两种记忆服务于同一个 VLA policy，但更新频率、信息密度和使用方式不同。

教学实现需要分别验证“近期细节是否还可访问”和“早期阶段是否被语义记忆保留”，不能只增加历史帧数量后把性能变化统称为 memory 能力。

## 6. π₀.₇：context 不只是 task command

同一句“完成任务”可以对应不同策略、速度、质量和子目标。π₀.₇ 的主要思想是用更丰富的多模态 context steering policy，使 demonstration、次优/失败 autonomous data 和非机器人数据能通过上下文被区分和利用。

## 原始资料

- [π₀ paper](https://www.physicalintelligence.company/download/pi0.pdf)
- [FAST paper](https://arxiv.org/abs/2501.09747)
- [Hi Robot paper](https://www.physicalintelligence.company/download/hirobot.pdf)
- [Hi Robot project page](https://www.pi.website/research/hirobot)
- [RTC paper](https://arxiv.org/abs/2506.07339)
- [Training-time RTC paper](https://arxiv.org/abs/2512.05964)
- [RTC Kinetix reference implementation](https://github.com/Physical-Intelligence/real-time-chunking-kinetix)
- [π₀.₅ paper](https://arxiv.org/abs/2504.16054)
- [Knowledge Insulation paper](https://www.physicalintelligence.company/download/pi05_KI.pdf)
- [Knowledge Insulation 附录](appendices/a_knowledge_insulation.md)
- [π*₀.₆ paper](https://www.physicalintelligence.company/download/pistar06.pdf)
- [MEM paper](https://arxiv.org/abs/2603.03596)
- [MEM project page](https://www.pi.website/research/memory)
- [π₀.₇ paper](https://arxiv.org/abs/2604.15483)
- [openpi source](https://github.com/Physical-Intelligence/openpi)
