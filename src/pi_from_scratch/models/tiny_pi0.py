import math

import torch
from torch import Tensor, nn

from pi_from_scratch.config import ModelConfig
from pi_from_scratch.inference import FlowSolver, flow_sample
from pi_from_scratch.models.prefix_suffix import (
    Pi0AttentionLayout,
    TwoExpertTransformer,
    make_pi0_attention_layout,
)
from pi_from_scratch.objectives import (
    masked_flow_matching_loss,
    sample_flow_batch,
    training_rtc_flow_batch,
)


def masked_action_mse(predicted: Tensor, target: Tensor, valid_mask: Tensor) -> Tensor:
    """Backward-compatible name for the masked per-step vector MSE."""
    return masked_flow_matching_loss(predicted, target, valid_mask)


def sinusoidal_time_embedding(time: Tensor, width: int) -> Tensor:
    if width % 2:
        raise ValueError("width must be even")
    fraction = torch.linspace(0.0, 1.0, width // 2, device=time.device)
    period = 4e-3 * (4.0 / 4e-3) ** fraction
    angles = time[..., None] * (2.0 * math.pi / period)
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
            nn.AdaptiveAvgPool2d((2, 2)),
        )

    def forward(self, image: Tensor) -> Tensor:
        return self.net(image).flatten(2).transpose(1, 2)


class TinyPi0(nn.Module):
    """A small π0-shaped policy, not a reproduction of the pretrained π0 model."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        width = config.width
        self.image_encoder = ImageEncoder(width)
        self.text_embedding = nn.Embedding(config.vocab_size, width, padding_idx=0)
        self.state_input = nn.Linear(config.state_dim, width)
        self.action_input = nn.Linear(config.action_dim, width)
        self.time_mlp = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width))
        self.position = nn.Parameter(torch.randn(config.action_horizon, width) * 0.02)
        self.transformer = TwoExpertTransformer(
            width=width,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            dropout=config.dropout,
        )
        self.action_output = nn.Linear(width, config.action_dim)

    def encode_prefix(
        self, image: Tensor, text_ids: Tensor, text_mask: Tensor
    ) -> tuple[Tensor, Tensor]:
        image = image.float()
        if image.max() > 1.0:
            image = image / 255.0
        image_tokens = self.image_encoder(image)
        text_tokens = self.text_embedding(text_ids)
        image_mask = torch.ones(image_tokens.shape[:2], dtype=torch.bool, device=image.device)
        prefix_tokens = torch.cat((image_tokens, text_tokens), dim=1)
        prefix_mask = torch.cat((image_mask, text_mask), dim=1)
        return prefix_tokens, prefix_mask

    def embed_suffix(
        self,
        state: Tensor,
        noisy_actions: Tensor,
        time: Tensor,
    ) -> Tensor:
        horizon = noisy_actions.shape[1]
        if horizon > self.config.action_horizon:
            raise ValueError("action chunk is longer than configured action_horizon")
        state_token = self.state_input(state.float())[:, None]
        action_tokens = self.action_input(noisy_actions) + self.position[:horizon]
        if time.ndim == 1:
            action_time = time[:, None].expand(-1, horizon)
        elif time.shape == noisy_actions.shape[:2]:
            action_time = time
        else:
            raise ValueError("time must have shape [batch] or [batch, horizon]")
        time_tokens = self.time_mlp(
            sinusoidal_time_embedding(action_time, self.config.width)
        )
        action_tokens = action_tokens + time_tokens
        return torch.cat((state_token, action_tokens), dim=1)

    def predict_velocity(
        self,
        noisy_actions: Tensor,
        time: Tensor,
        prefix_tokens: Tensor,
        prefix_mask: Tensor,
        state: Tensor,
        valid_mask: Tensor | None = None,
        *,
        return_layout: bool = False,
    ) -> Tensor | tuple[Tensor, Pi0AttentionLayout]:
        horizon = noisy_actions.shape[1]
        if valid_mask is not None and (
            valid_mask.shape != noisy_actions.shape[:2] or valid_mask.dtype != torch.bool
        ):
            raise ValueError("valid_mask must be bool with shape [batch, horizon]")
        if valid_mask is None:
            valid_mask = torch.ones(
                noisy_actions.shape[:2], dtype=torch.bool, device=noisy_actions.device
            )
        suffix_tokens = self.embed_suffix(state, noisy_actions, time)
        layout = make_pi0_attention_layout(prefix_mask, valid_mask)
        _, suffix_output = self.transformer(prefix_tokens, suffix_tokens, layout)
        velocity = self.action_output(suffix_output[:, 1 : horizon + 1])
        velocity = velocity * valid_mask.unsqueeze(-1).to(velocity.dtype)
        if return_layout:
            return velocity, layout
        return velocity

    def loss(
        self,
        batch: dict[str, Tensor],
        *,
        noise: Tensor | None = None,
        time: Tensor | None = None,
    ) -> Tensor:
        prefix_tokens, prefix_mask = self.encode_prefix(
            batch["image"], batch["text_ids"], batch["text_mask"]
        )
        flow_batch = sample_flow_batch(batch["actions"].float(), noise=noise, time=time)
        action_mask = batch.get(
            "action_mask",
            torch.ones(
                flow_batch.noisy_actions.shape[:2],
                dtype=torch.bool,
                device=flow_batch.noisy_actions.device,
            ),
        )
        predicted_velocity = self.predict_velocity(
            flow_batch.noisy_actions,
            flow_batch.time,
            prefix_tokens,
            prefix_mask,
            batch["state"],
            action_mask,
        )
        assert isinstance(predicted_velocity, Tensor)
        return masked_flow_matching_loss(
            predicted_velocity,
            flow_batch.target_velocity,
            action_mask,
        )

    def training_rtc_loss(
        self,
        batch: dict[str, Tensor],
        *,
        prefix_lengths: Tensor,
        noise: Tensor,
        time: Tensor,
    ) -> Tensor:
        """Flow loss for the training-time RTC extension with per-token flow times."""
        prefix_tokens, prefix_mask = self.encode_prefix(
            batch["image"], batch["text_ids"], batch["text_mask"]
        )
        rtc_batch = training_rtc_flow_batch(
            batch["actions"].float(),
            prefix_lengths,
            noise=noise,
            time=time,
        )
        action_mask = batch.get(
            "action_mask",
            torch.ones_like(rtc_batch.loss_mask),
        )
        loss_mask = action_mask & rtc_batch.loss_mask
        predicted_velocity = self.predict_velocity(
            rtc_batch.noisy_actions,
            rtc_batch.token_time,
            prefix_tokens,
            prefix_mask,
            batch["state"],
            action_mask,
        )
        assert isinstance(predicted_velocity, Tensor)
        return masked_flow_matching_loss(
            predicted_velocity,
            rtc_batch.target_velocity,
            loss_mask,
        )

    @torch.no_grad()
    def sample_actions(
        self,
        batch: dict[str, Tensor],
        num_steps: int = 10,
        *,
        noise: Tensor | None = None,
        solver: FlowSolver = "euler",
    ) -> Tensor:
        prefix_tokens, prefix_mask = self.encode_prefix(
            batch["image"], batch["text_ids"], batch["text_mask"]
        )
        shape = (
            batch["state"].shape[0],
            self.config.action_horizon,
            self.config.action_dim,
        )
        return flow_sample(
            lambda actions, time: self.predict_velocity(
                actions,
                time,
                prefix_tokens,
                prefix_mask,
                batch["state"],
            ),
            shape,
            device=batch["state"].device,
            num_steps=num_steps,
            noise=noise,
            solver=solver,
        )
