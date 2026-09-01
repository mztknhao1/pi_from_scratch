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
    dataset_revision: str = "v3.0"
    image_key: str = "observation.image"
    state_key: str = "observation.state"
    action_key: str = "action"
    video_backend: str = "pyav"
    default_prompt: str = "push the T-shaped block to the target"
    num_workers: int = 0
    validation_fraction: float = 0.2
    synthetic_num_episodes: int = 6
    synthetic_episode_length: int = 8


@dataclass(frozen=True)
class TrainConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    steps: int = 1_000
    log_every: int = 20
    eval_every: int = 100
    save_every: int = 500
    max_eval_batches: int = 8
    sampling_steps: int = 10
    overfit_samples: int | None = None
    seed: int = 7
    output_dir: str = "outputs/debug"
