from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelConfig:
    image_size: int = 96
    state_dim: int = 2
    action_dim: int = 2
    action_horizon: int = 16
    vocab_size: int = 2048
    max_text_tokens: int = 16
    width: int = 128
    num_layers: int = 3
    num_heads: int = 4
    dropout: float = 0.0


@dataclass(frozen=True)
class DataConfig:
    dataset: str = "synthetic"
    image_key: str = "observation.image"
    state_key: str = "observation.state"
    action_key: str = "action"
    default_prompt: str = "push the T-shaped block to the target"
    num_workers: int = 0


@dataclass(frozen=True)
class TrainConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    steps: int = 1_000
    log_every: int = 20
    save_every: int = 500
    seed: int = 7
    output_dir: str = "outputs/debug"
