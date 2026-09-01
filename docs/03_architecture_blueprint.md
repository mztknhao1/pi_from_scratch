# 目标代码架构与仿真验证蓝图

本文件定义目标边界。当前代码已按 `data`、`representations`、`models`、`objectives`、`inference`、`policies`、`runtime` 和 `cli` 分层；environment、evaluation 与 memory 等目录仍按讲次在真正需要时加入，不预先创建空壳。

## 1. 目标目录

```text
pi_from_scratch/
├── configs/
│   ├── data/                 # dataset revision、keys、fps、action space
│   ├── model/                # tiny_pi0、fast-like 等
│   └── experiment/           # 可复现的训练/评估组合
├── docs/
│   ├── 00_learning_path.md
│   ├── 01_pi_family.md
│   ├── 03_architecture_blueprint.md
│   └── lessons/              # 各讲正文，尚未开始编写
├── src/pi_from_scratch/
│   ├── data/                 # schema、LeRobot adapter、window、split、normalizer
│   ├── representations/      # action transform、FAST tokenizer
│   ├── models/               # encoder、action expert、value/context heads
│   ├── memory/               # short-term video、long-term text memory 与更新接口
│   ├── objectives/           # flow、autoregressive、mixed、advantage objectives
│   ├── policies/             # 训练模型到统一推理接口的适配
│   ├── inference/            # solver、sampling；不管理环境时钟
│   ├── runtime/              # buffer、scheduler、async worker、RTC
│   ├── envs/                 # PushT/LIBERO adapter、latency wrapper
│   ├── evaluation/           # rollout、metrics、video、result schema
│   └── cli/                  # inspect/train/eval/benchmark 命令
├── tests/
│   ├── unit/                 # V0/V1
│   ├── integration/          # V2/V3
│   └── e2e/                  # V4 smoke tests
└── outputs/                  # gitignored；每次运行包含 config 和代码版本
```

## 2. 核心数据契约

下面是语义约束，不是要求立即使用某种 dataclass 库。

### ObservationBatch

```text
images:        dict[camera_name, float[B,C,H,W]]
image_masks:   dict[camera_name, bool[B]]
state:         float[B,state_dim]
state_mask:    bool[B,state_dim]
prompts:       tuple[str]，公共边界保留 raw text
context:       optional typed fields，不使用无定义的 metadata dict
timestamp_s:   float[B]，simulator/robot time，单位秒
embodiment_id: optional string/int
```

### ActionChunk

```text
values:         float[B,H,A]
valid_mask:     bool[B,H]
timestamps:     float[B,H]
representation: absolute | delta | velocity
space:          joint | ee | simulator-specific
```

归一化后的 tensor 仍需携带可追溯到原 action space 的 transform artifact。policy 输出进入环境前必须显式 denormalize 和 inverse transform。

### PolicyOutput

```text
action_chunk: ActionChunk（越过 policy 边界前已 inverse transform）
generated_at_monotonic_s: wall-clock monotonic time
source_observation_timestamp_s: simulator/robot observation time
inference_latency_s: measured latency
debug: optional solver/token/context diagnostics
```

### EpisodeResult

```text
success / reward / episode length
policy latency samples / deadline misses
executed and predicted action traces
chunk boundary indices
seed / config id / checkpoint id
video and failure reason
```

## 3. 依赖方向

```text
data -> representations -> models/objectives -> policies -> inference
                                                    |
envs -> runtime ------------------------------------+
  \          |
   \------ evaluation
```

必须保持的边界：

- `data` 不导入模型；
- `objectives` 不导入 simulator/runtime；
- `inference` 只解决“如何产生 chunk”，不调用 `env.step()`；
- `runtime` 通过统一 policy protocol 工作，不判断模型是 flow 还是 FAST；
- `RTC` 只依赖支持条件化/inpainting 的 flow-policy adapter；普通 runtime 不依赖 RTC；
- `evaluation` 观察 runtime 和 environment 事件，不修改 policy 行为。

## 4. Policy 与 Runtime 接口

同步基线：

```text
observation -> policy.predict_chunk -> wait -> execute E actions -> observe again
```

异步基线：

```text
control loop: consume action buffer at fixed fps
worker:       read timestamped observation -> predict next chunk -> publish
scheduler:    decide how a published chunk replaces/merges with the buffer
```

RTC scheduler 还需要：

- 推理开始时已经 committed 的 action prefix；
- 新 chunk 对应的 observation timestamp；
- 预计 inference delay 或执行到达位置；
- flow sampler 的 per-step prefix constraint/inpainting hook。

控制循环必须使用 monotonic clock。数据 timestamp、环境 step index 和 wall-clock latency 分开记录。

## 5. 仿真策略

### 核心 demo：PushT

选择 `lerobot/pusht` + PushT environment，原因是数据小、连续 2D action、像素 observation、官方 LeRobot 训练和评估入口完整，适合快速闭环与延迟实验。

核心 demo 验证：

- trajectory 到 action chunk；
- flow/FAST-like 两种输出方式；
- synchronous/receding-horizon execution；
- latency injection 与 RTC；
- success、reward、throughput、jerk。

它不能验证开放世界语言泛化。固定 prompt 只是在验证 language 字段穿过管线，不能据此声称学会语言条件。

### 高级章节：controlled context experiments

π₀.₅–π₀.₇ 与 MEM 先用受控变体验证代码机制：task/source/reward/advantage/strategy/subgoal image 是否能正确影响目标和输出，以及短期视频/长期文本记忆能否分别处理遮挡与阶段追踪。所有结果标记为 mechanism test。

### 可选扩展：LIBERO 小子集

当核心 demo 稳定后，再用 LIBERO 的少量语言任务验证真实多任务条件。LIBERO 依赖和训练成本不进入必修 CPU smoke path。

## 6. 最终 demo 命令契约

具体 CLI 名可以在实现时微调，但最终必须保留四类稳定入口：

```bash
# 看懂一个样本及其时间窗口
pi-inspect --config configs/experiment/pusht_flow.yaml

# 训练并保存完整 provenance
pi-train --config configs/experiment/pusht_flow.yaml

# 无人为延迟的闭环评估
pi-eval --checkpoint <path> --runtime receding_horizon

# 对推理延迟和 runtime 方法做统一 sweep
pi-benchmark-rtc --checkpoint <path> --delays-ms 0,50,100,200 \
  --runtimes blocking,async_latest,rtc
```

每个 output 目录至少包含：

```text
resolved_config.yaml
run_metadata.json       # git commit、依赖版本、设备、seed
metrics.json
traces.npz
videos/
checkpoint/             # 仅训练任务
```

## 7. 近期实现顺序

1. 把已经独立验证的 episode split、窗口 mask、action transform 和 normalizer 接入同一份训练数据配置。
2. 已在锁定的 flow path、target velocity 和采样方向上实现 observation prefix、action suffix、双专家层与 attention mask。
3. 把数据配置、prefix/suffix model、flow objective 和 optimizer 组装成可诊断的小型训练任务。
4. 在已经明确的 prediction/execution horizon、replanning interval 和同步 chunk executor 上接入真实 policy 输出。
5. 增加统一 `Policy` adapter 与同步 PushT runner。
6. 加入 latency wrapper 和 async runtime，再单独实现 RTC。
7. 基线稳定后才进入 FAST、π₀.₅–π₀.₇ 和 MEM，避免高级目标掩盖数据/执行错误。
