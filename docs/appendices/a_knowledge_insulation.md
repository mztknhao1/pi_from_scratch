# 附录 A：Knowledge Insulation——让 VLM 学会控制，同时保护预训练知识

> 论文：[Knowledge Insulating Vision-Language-Action Models: Train Fast, Run Fast, Generalize Better](https://www.physicalintelligence.company/download/pi05_KI.pdf)

把一个预训练 VLM 改造成机器人策略时，我们希望同时得到三种能力：

- 继承 VLM 对物体、场景和语言的理解；
- 输出连续、精细的 action chunk；
- 在机器人需要的控制节拍内完成推理。

π₀ 的 continuous action expert 解决了后两项，却引入了新的训练问题：action expert 从随机参数开始学习，flow loss 可以穿过它与 VLM 的交互路径回传到 backbone。训练早期，这条梯度会扰动已经学好的视觉语言表征，模型可能收敛得更慢，也更容易忽略语言指令。

Knowledge Insulation 给出的方案可以概括成一句话：

> **让 action expert 在前向传播中读取 VLM 的知识，同时截断 flow loss 回写 VLM 的梯度；再用 FAST action token 和通用 VLM 数据继续训练 backbone。**

这篇附录只解决这套训练机制。异构数据的类型与采样比例见[第 12 讲](../lessons/12_pi05_heterogeneous_cotraining.md)，FAST 的编码过程见[第 11 讲](../lessons/11_fast_action_tokenizer.md)，flow matching 的定义见[第 5 讲](../lessons/05_conditional_flow_matching.md)。

## 一、Knowledge Insulation 中文怎么理解？

`insulation` 常见于电气与热学语境，含义是“绝缘”或“隔热”。本文语境中，我建议统一译作 **知识隔离**；“知识绝缘”更贴近论文使用的工程隐喻，也可以帮助理解梯度边界。

这里的“知识”主要指预训练 VLM backbone 中形成的视觉、语言与常识表征。“隔离”针对的是 continuous action expert 带来的反向梯度：

```text
前向：VLM backbone ──语义特征──> action expert       允许
反向：VLM backbone <──flow gradient── action expert  截断
```

因此，action expert 仍然知道画面里有什么、指令要求什么。受保护的是 VLM 参数，flow loss 无法沿这条路径直接修改它们。

可以把 VLM 想成一位已经具备丰富常识的老师，把随机初始化的 action expert 想成刚开始练习机械臂控制的学生。学生可以阅读老师给出的场景解释；学生早期笨拙的动作误差，不会反过来改写老师已有的全部知识。与此同时，老师仍通过 FAST action token 和图文数据学习机器人场景中的新概念。

## 二、论文从什么问题出发？

### 2.1 离散 action 路线：容易借用 VLM，在线解码偏慢

RT-2、π₀-FAST 一类方法把连续动作变成 token，再使用标准 next-token prediction：

$$
\mathcal L_{AR}
=-\sum_j M_j\log p_\theta(y_{j+1}\mid x_{1:j}).
$$

这条路线与 VLM 预训练接口一致，backbone 能直接收到清晰的 token-level 监督。FAST 还会压缩 action token 序列，训练效率较好。

部署时仍要逐 token 自回归解码。论文报告 π₀-FAST 在 RTX 4090 上生成 1 秒 action chunk 约需 750 ms，约 1.3 Hz；高频、灵巧的连续控制会受到延迟与量化误差影响。

### 2.2 连续 action 路线：运行快，新模块会干扰 backbone

π₀ 增加一个较小的 continuous action expert，通过 flow matching 一次并行处理整段 action chunk。论文中的 backbone 约 2B 参数，action expert 约 300M 参数；连续路径能达到约 10 Hz 的控制频率。

问题出现在训练起点：VLM 已经预训练完成，action expert 及其连续输入输出投影仍是随机参数。若 flow loss 直接回传到 backbone，backbone 会接收到一条经过随机模块形成的训练信号。论文把由此产生的影响描述为 gradient interference，并观察到：

- flow-only 的 π₀ 训练收敛较慢；
- 部分模型更容易依据画面中的行为先验行动，忽略语言指定的目标；
- web/VLM 预训练知识向机器人动作的迁移变弱。

这里需要保持严谨：论文用多组消融实验支持“随机初始化 adapter 的梯度干扰预训练表征”这一解释，但没有给出一套直接测量“知识存量”的单一标尺。这是一项由训练表现、语言跟随和 OOD 泛化共同支持的机制判断。

### 2.3 直接冻结 backbone 也走不通

完全冻结 VLM 看起来能保护预训练参数，却会阻止 backbone 适应机器人输入：多相机视角、proprioceptive state、机械臂自身外观和精细空间关系，都可能超出原始 VLM 的训练分布。

论文的冻结消融在部分真实机器人任务上接近 0% 表现。由此可以得到一条更精确的目标：

> backbone 需要继续学习机器人表征，同时要避开随机 continuous expert 带来的破坏性梯度。

## 三、完整方法由三个部件组成

Knowledge Insulation 经常被简化成“加 stop-gradient”。完整 recipe 还需要另外两条路径配合。

### 3.1 Joint-training：同一段动作生成两份监督

对一段连续动作 $A$，训练时同时构造：

```text
A ──FAST tokenizer──────────> action tokens ──> token CE
│
└──加噪并采样 flow time τ──> noisy action ───> flow MSE
```

联合目标可以写为：

$$
\mathcal L_{CO\text{-}VLA}
=\mathcal L_{token}
+\alpha M_{act}\mathcal L_{flow}.
$$

$\mathcal L_{token}$ 覆盖普通文本、high-level subtask 和 FAST action token；$M_{act}$ 表示当前样本是否带有连续 action chunk。论文加入 insulation 后令 $\alpha=1$，理由是 flow loss 已经主要作用于独立的 action-expert 参数。

两份 action representation 分工如下：

| 表示 | 训练时的职责 | 部署时的职责 |
|---|---|---|
| FAST action tokens | 为 backbone 提供机器人动作的 next-token 学习信号 | 实时低层控制中不解码 |
| continuous action chunk | 用 flow matching 训练 action expert | 由较小 expert 快速生成并执行 |

FAST 在这里更像一个 **representation-learning objective**。即便部署时舍弃这条解码路径，它在训练阶段仍能教 backbone：哪些视觉语言特征与后续机器人动作有关。

### 3.2 VLM data co-training：持续复习语义知识

训练 mixture 还包含 caption、VQA、object localization 和 robot planning 等非动作数据。它们只激活 token loss：

$$
M_{act}=0,
\qquad
\mathcal L=\mathcal L_{token}.
$$

这些样本持续约束 backbone 的视觉语言能力。论文实验显示，移除 VLM data 后，generalist policy 的语言跟随与新物体泛化会下降；没有 stop-gradient 的 joint-training 对这类数据尤其敏感。

### 3.3 Stop-gradient：只阻断 action expert 回写 backbone

设 VLM backbone 参数为 $\theta_b$，continuous action expert 参数为 $\theta_a$。直观写法是：

$$
h=f_{\theta_b}(I,q,\ell),
\qquad
\tilde h=\operatorname{sg}(h),
$$

$$
\hat u=f_{\theta_a}(A_\tau,\tau,\tilde h).
$$

$\operatorname{sg}(x)$ 的前向值等于 $x$，反向导数为 0：

$$
\operatorname{sg}(x)=x,
\qquad
\frac{\partial\operatorname{sg}(x)}{\partial x}=0.
$$

于是 flow loss 满足：

$$
\frac{\partial\mathcal L_{flow}}{\partial\theta_b}=0,
\qquad
\frac{\partial\mathcal L_{flow}}{\partial\theta_a}\ne0.
$$

backbone 仍会被 token objective 更新：

$$
\frac{\partial\mathcal L_{token}}{\partial\theta_b}\ne0.
$$

这三个式子共同定义了 KI 的核心边界。只写第一个零梯度条件会遗漏 backbone 如何适应机器人数据。

## 四、stop-gradient 在 Transformer 的什么位置？

`TinyPi05` 对最终 condition 做 `detach()`，便于检查宏观梯度边界。论文的实现更细：VLM token 与 continuous action token 在每一层使用两套 expert 权重，并在共享 attention 中交互。

把一层里的 token 分成两组：

- $X_b$：image、language、state、FAST action 等 backbone stream；
- $X_a$：noisy continuous action stream。

注意力概率可按 query 与 key 的来源分块：

$$
P=
\begin{bmatrix}
P_{bb} & 0\\
P_{ab} & P_{aa}
\end{bmatrix}.
$$

四个位置的含义是：

| 分块 | 信息流 | 是否允许 |
|---|---|---:|
| $P_{bb}$ | backbone query 读取 backbone key | ✓ |
| 右上角 | backbone query 读取 action-expert key | — |
| $P_{ab}$ | action-expert query 读取 backbone key | ✓ |
| $P_{aa}$ | action-expert query 读取 action-expert key | ✓ |

因此前向信息从 backbone 单向流入 action expert。实现 insulation 时，跨 expert 的 key/value 使用 stop-gradient：

$$
P_{ab}
=\operatorname{softmax}
\left(Q_a(X_a)\operatorname{sg}(K_b(X_b))^\top\right),
$$

$$
E_a
=P_{ab}\operatorname{sg}(V_b(X_b))
+P_{aa}V_a(X_a).
$$

为什么 key 和 value 都要截断？

- 若只截断 value，flow loss 仍可通过 attention score 回传到 backbone key；
- 若只截断 key，flow loss 仍可通过被加权的 backbone value 回传；
- 两者都使用 `sg`，cross-attention 的前向条件保持不变，flow gradient 才无法沿跨 expert 路径进入 backbone。

每一层都存在这种交互，所以论文在 attention 内逐层处理。课程中的一次 `condition.detach()` 保留了最终梯度关系，也省略了 layer-wise 双 expert 结构。

## 五、两种 action 表示为什么还要互相看不见？

robot sample 同时含有 FAST action target 和 continuous action target。如果 continuous action tokens 能读取 ground-truth FAST tokens，flow expert 在训练时可能通过答案预测答案；部署时 FAST 答案不存在，模型会遭遇训练—推理信息差。

论文的 attention mask 因此规定：

```text
image / language / text state：构成 prefix
FAST action tokens：读取 prefix 和先前 FAST token，保持 autoregressive
continuous action tokens：读取 prefix 和全部 continuous action tokens
FAST 与 continuous action token：彼此不可见
```

这里包含两种边界：

1. **attention mask** 管理前向传播时“谁能看见谁”；
2. **stop-gradient** 管理反向传播时“哪项 loss 能更新谁”。

二者解决不同问题。mask 防止答案泄漏与不合适的信息耦合，stop-gradient 保护预训练 backbone 的参数更新。

## 六、参数到底由哪项 loss 更新？

对一个带动作的 robot sample，可以用下表检查训练实现：

| 参数组 | token CE 更新 | flow loss 更新 |
|---|---:|---:|
| VLM/image backbone | ✓ | — |
| text / FAST token head | ✓ | — |
| continuous state 表征路径 | ✓，通过离散 action 预测获得监督 | —，跨 expert 梯度被截断 |
| continuous action expert | — | ✓ |
| noisy-action/time/output projection | — | ✓ |

对普通 VLM sample，只有第一、二行参与训练。对 action-only sample，FAST token CE 与 flow loss 同时存在。对附带 high-level description 的 robot sample，语言 token、FAST token 和 continuous action 可以在同一训练样本中各自获得监督。

这也回答了一个常见疑问：**有了 stop-gradient，action expert 怎样学会利用语言？**

action expert 前向时一直能读取由图像和语言形成的 backbone key/value。flow loss 更新 action expert 的 query、attention 和后续层，使它逐渐学会从这些特征中选择与动作相关的信息。梯度无需进入 backbone，expert 自身仍然能够学习“怎样读懂 backbone”。

## 七、它与 π₀、FAST、π₀.₅ 是什么关系？

| recipe | backbone 的机器人监督 | continuous expert | flow 是否回传 backbone | 低层部署输出 |
|---|---|---:|---:|---|
| π₀ | flow matching | 从头训练 | ✓ | continuous action |
| π₀-FAST | FAST token CE | 无 | — | autoregressive tokens |
| 原始 π₀.₅ | 阶段 1 用 FAST；阶段 2 token + flow | 阶段 2 加入 | ✓ | continuous action |
| joint-training 消融 | FAST/token + flow 同时训练 | 从训练开始存在 | ✓ | continuous action |
| Knowledge Insulation | FAST/token + VLM co-training | 从训练开始存在 | 截断 | continuous action |

原始 π₀.₅ 通过两阶段训练，先让 backbone 学会离散机器人动作，再加入 continuous expert。KI 将这条思路形式化为单阶段联合训练：action expert 可以从第一步起学习，FAST 和 VLM objective 同时保护、适配 backbone。

## 八、论文实验真正说明了什么？

论文在静态单臂、静态双臂、移动双臂、DROID 和 LIBERO 上比较多种 recipe。读图时建议关注四组结论。

### 8.1 训练速度

在 generalist table-bussing 实验中，KI 的收敛速度接近 π₀-FAST；flow-only π₀ 达到相近性能需要约 7.5 倍训练 step。这支持 FAST token 作为 representation-learning signal 的价值。

### 8.2 语言跟随

items-in-drawer 与 table-bussing 要求模型从多个合理动作中选择语言指定的目标。论文观察到 π₀ 和无 stop-gradient 的 joint-training 更容易忽略语言；KI 的表现更好。加入 VLM co-training 也能缓解 joint-training 的问题，说明数据约束与梯度隔离都在发挥作用。

### 8.3 连续控制与推理速度

π₀-FAST 保留了较好的语言能力，但自回归推理使真实任务耗时更长，在细致、动态动作上也受限。KI 部署时只运行较小的 continuous expert，保留了 flow action 的并行生成优势。

### 8.4 泛化

论文在未见环境、新物体、多 embodiment generalist 以及 LIBERO 上报告了收益。例如 DROID 评分为 $0.55\pm0.09$，论文重训的 π₀ 与 π₀-FAST 分别为 $0.49\pm0.09$ 和 $0.45\pm0.09$。LIBERO 结果并非所有 suite 都领先：从 generalist fine-tune 的模型在 Spatial 和 LIBERO-90 上取得表中最佳结果，在 LIBERO-10 上低于部分基线。

这些结果适合支持“该 recipe 在论文设置中改善训练、语言跟随和部分泛化”的结论。它们不能推出 stop-gradient 对所有 backbone、数据比例和机器人任务都必然最优。

## 九、对照本仓代码理解

最小对应实现位于 [`tiny_pi05.py`](../../src/pi_from_scratch/models/tiny_pi05.py)：

```python
condition = self.encode(observation)
if insulate_backbone:
    condition = condition.detach()
velocity = self.action_expert(condition, noisy_actions, time)
```

可以把这三行对应到论文结构：

| 教学代码 | 论文模型 |
|---|---|
| `encode(observation)` | 多层 VLM/image backbone |
| `condition.detach()` | 每层 cross-expert attention 中对 backbone K/V 使用 `sg` |
| `action_expert(...)` | 300M continuous action expert 与输入输出投影 |

loss 路由位于 [`mixed.py`](../../src/pi_from_scratch/objectives/mixed.py)：

```python
robot_loss = discrete_action_ce + continuous_flow_mse
semantic_loss = semantic_ce
```

运行：

```bash
pi-pi05-mixture-demo
pytest -q tests/test_pi05_mixture.py
```

其中最关键的 V1 检查是：

```text
semantic CE                  -> backbone grad > 0
discrete action CE           -> backbone grad > 0
flow without insulation      -> backbone grad > 0
flow with insulation         -> backbone grad = 0
flow with insulation         -> action expert grad > 0
```

课程实现用于验证梯度接线。它没有多层 dual-expert attention、真实 FAST 序列、PaliGemma 初始化和论文数据 mixture，实验数值也不能与论文成功率横向比较。

## 十、常见误解与实现检查

### 10.1 “KI 就是冻结 VLM”

冻结会让所有 loss 都无法更新 backbone。KI 允许 token CE 持续更新 backbone，只隔离 continuous expert 的 flow gradient。

### 10.2 “action expert 不需要 FAST，因为推理时不用 FAST”

FAST 的主要价值出现在训练期：它让 backbone 获得与机器人动作直接相关的 token-level 监督。论文消融中，朴素离散化也比仅用 continuous action 训练更好，但效果低于 FAST。

### 10.3 “对最终 flow output 调用 detach 就能保护 backbone”

此时梯度要么连 action expert 也无法训练，要么早已通过 attention 进入 backbone。边界应放在 action expert 读取 backbone 特征的位置；论文具体截断跨 expert 的 backbone key/value。

### 10.4 “stop-gradient 会让语言 condition 失效”

前向特征没有被删除。action expert 仍读取语言相关的 key/value，并通过自己的 flow loss 学习如何使用它们。

### 10.5 “同一份 action 的 FAST 与 continuous token 可以互相 attention”

这会产生 target leakage。联合 loss 共享 prefix 和底层语义表征，两种 action target 之间由 attention mask 隔开。

## 十一、代价与开放问题

论文报告，同时训练离散和连续输出会增加约 20% 的训练计算量；更快的收敛在其实验中抵消了这项成本。工程上仍要关注：

- token loss 与 flow loss 的 batch 比例会影响两条路径的学习速度；
- backbone 换成更强或已做机器人预训练的模型后，最佳梯度边界可能变化；
- 连续 state projector 如何获得充分监督，取决于它被接入哪条 token 预测路径；
- stop-gradient 保护参数更新，无法单独解决数据偏差、错误语言标注或 action space 冲突；
- 论文数据包含大量私有机器人轨迹，开源小数据集只能验证机制，无法复现其规模结论。

## 十二、读完后应该形成的整体图景

Knowledge Insulation 解决的是 VLA 训练中的一条结构性冲突：预训练 backbone 需要适应机器人，又容易被随机 continuous adapter 的梯度扰动。

它用三条互补机制化解这项冲突：

1. FAST action token 让 backbone 学习机器人相关表征；
2. 通用 VLM co-training 持续维持语义与泛化能力；
3. continuous action expert 读取 backbone 的前向特征，flow gradient 在 cross-expert attention 处停止。

最终，token 路径服务于“训练快、保留知识”，continuous expert 服务于“运行快、动作精细”。两条路径在训练时共存，低层部署只使用 continuous action 输出。

## 自测问题

1. 为什么只冻结 backbone 无法替代 Knowledge Insulation？
2. FAST action token 在部署时不解码，为什么仍能提升 continuous action policy？
3. attention mask 与 stop-gradient 分别约束什么？
4. 为什么跨 expert attention 的 key 和 value 都要使用 `sg`？
5. 只保留 flow loss 并加 stop-gradient，会出现什么问题？
6. `condition.detach()` 与论文逐层处理 K/V 有哪些相同点和简化？

## 建议阅读顺序

- 先读论文 Figure 1 和第 4 节，建立“随机 action expert 干扰 backbone”的问题意识；
- 再读第 5.1 节的 joint-training loss，确认 FAST 与 continuous action 在训练时同时存在；
- 精读第 5.2 节公式 (5)、(6)，沿着 K/V 路径手画一次前向信息流和反向梯度流；
- 最后读 Figure 4–9 与 Discussion，区分语言跟随、训练速度、在线速度和泛化这几组证据。
