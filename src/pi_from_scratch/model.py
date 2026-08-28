import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from pi_from_scratch.config import ModelConfig
from pi_from_scratch.flow_matching import euler_sample, sample_flow_batch


def masked_action_mse(predicted: Tensor, target: Tensor, valid_mask: Tensor) -> Tensor:
    """Average action error over valid timesteps only."""
    if predicted.shape != target.shape or predicted.ndim != 3:
        raise ValueError("predicted and target must share shape [batch, horizon, action_dim]")
    if valid_mask.shape != predicted.shape[:2] or valid_mask.dtype != torch.bool:
        raise ValueError("valid_mask must be bool with shape [batch, horizon]")
    per_step = F.mse_loss(predicted, target, reduction="none").mean(dim=-1)
    weights = valid_mask.to(per_step.dtype)
    return (per_step * weights).sum() / weights.sum().clamp_min(1.0)


def sinusoidal_time_embedding(time: Tensor, width: int) -> Tensor:
    if width % 2:
        raise ValueError("width must be even")
    fraction = torch.linspace(0.0, 1.0, width // 2, device=time.device)
    period = 4e-3 * (4.0 / 4e-3) ** fraction
    angles = time[:, None] * (2.0 * math.pi / period[None, :])
    return torch.cat((angles.sin(), angles.cos()), dim=-1)


class ImageEncoder(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, width, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, image: Tensor) -> Tensor:
        return self.net(image).flatten(1)


class TinyPi0(nn.Module):
    """A small π0-shaped policy, not a reproduction of the pretrained π0 model."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        width = config.width
        self.image_encoder = ImageEncoder(width)
        self.text_embedding = nn.Embedding(config.vocab_size, width, padding_idx=0)
        self.state_encoder = nn.Sequential(nn.Linear(config.state_dim, width), nn.SiLU())
        self.condition = nn.Sequential(nn.Linear(3 * width, width), nn.SiLU())
        self.action_input = nn.Linear(config.action_dim, width)
        self.time_mlp = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width))
        self.position = nn.Parameter(torch.randn(config.action_horizon, width) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=config.num_heads,
            dim_feedforward=4 * width,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.action_expert = nn.TransformerEncoder(layer, num_layers=config.num_layers)
        self.action_output = nn.Linear(width, config.action_dim)

    def encode_condition(
        self, image: Tensor, state: Tensor, text_ids: Tensor, text_mask: Tensor
    ) -> Tensor:
        image = image.float()
        if image.max() > 1.0:
            image = image / 255.0
        image_feature = self.image_encoder(image)
        text_tokens = self.text_embedding(text_ids)
        weights = text_mask.unsqueeze(-1).to(text_tokens.dtype)
        text_feature = (text_tokens * weights).sum(1) / weights.sum(1).clamp_min(1.0)
        state_feature = self.state_encoder(state.float())
        return self.condition(torch.cat((image_feature, text_feature, state_feature), dim=-1))

    def predict_velocity(
        self,
        noisy_actions: Tensor,
        time: Tensor,
        condition: Tensor,
        valid_mask: Tensor | None = None,
    ) -> Tensor:
        horizon = noisy_actions.shape[1]
        if horizon > self.config.action_horizon:
            raise ValueError("action chunk is longer than configured action_horizon")
        if valid_mask is not None and (
            valid_mask.shape != noisy_actions.shape[:2] or valid_mask.dtype != torch.bool
        ):
            raise ValueError("valid_mask must be bool with shape [batch, horizon]")
        tokens = self.action_input(noisy_actions)
        tokens = tokens + self.position[:horizon]
        tokens = tokens + self.time_mlp(sinusoidal_time_embedding(time, self.config.width))[:, None]
        tokens = tokens + condition[:, None]
        padding_mask = None if valid_mask is None else ~valid_mask
        return self.action_output(
            self.action_expert(tokens, src_key_padding_mask=padding_mask)
        )

    def loss(self, batch: dict[str, Tensor]) -> Tensor:
        condition = self.encode_condition(
            batch["image"], batch["state"], batch["text_ids"], batch["text_mask"]
        )
        noisy_actions, time, target_velocity = sample_flow_batch(batch["actions"].float())
        action_mask = batch.get(
            "action_mask",
            torch.ones(noisy_actions.shape[:2], dtype=torch.bool, device=noisy_actions.device),
        )
        predicted_velocity = self.predict_velocity(noisy_actions, time, condition, action_mask)
        return masked_action_mse(predicted_velocity, target_velocity, action_mask)

    @torch.no_grad()
    def sample_actions(self, batch: dict[str, Tensor], num_steps: int = 10) -> Tensor:
        condition = self.encode_condition(
            batch["image"], batch["state"], batch["text_ids"], batch["text_mask"]
        )
        shape = (
            batch["state"].shape[0],
            self.config.action_horizon,
            self.config.action_dim,
        )
        return euler_sample(
            lambda actions, time: self.predict_velocity(actions, time, condition),
            shape,
            device=batch["state"].device,
            num_steps=num_steps,
        )
