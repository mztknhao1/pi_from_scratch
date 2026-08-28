"""Visualize a PushT action chunk on its observation images."""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor


@dataclass(frozen=True)
class VisualChunk:
    images: Tensor
    actions: Tensor
    valid_mask: Tensor
    state: Tensor
    fps: float
    episode_index: int
    frame_index: int
    timestamp_s: float

    def __post_init__(self) -> None:
        if self.images.ndim != 4 or self.images.shape[1] not in (1, 3, 4):
            raise ValueError("images must have shape [horizon, channels, height, width]")
        if self.actions.ndim != 2 or self.actions.shape != (self.images.shape[0], 2):
            raise ValueError("PushT actions must have shape [horizon, 2]")
        if self.valid_mask.shape != (self.images.shape[0],) or self.valid_mask.dtype != torch.bool:
            raise ValueError("valid_mask must be bool with shape [horizon]")
        if self.state.shape != (2,):
            raise ValueError("PushT state must have shape [2]")
        if not self.valid_mask[0].item():
            raise ValueError("the first chunk step must be valid")


def project_workspace_xy(
    points: Tensor, *, image_width: int, image_height: int, workspace_size: float = 512.0
) -> Tensor:
    """Project PushT ``[x, y]`` coordinates onto image pixel coordinates."""
    if points.shape[-1] != 2:
        raise ValueError("points must end with an [x, y] dimension")
    if image_width < 2 or image_height < 2:
        raise ValueError("image dimensions must be at least two pixels")
    if workspace_size <= 0:
        raise ValueError("workspace_size must be positive")
    scale = torch.tensor(
        [(image_width - 1) / workspace_size, (image_height - 1) / workspace_size],
        dtype=torch.float32,
        device=points.device,
    )
    return points.to(torch.float32) * scale


def select_timeline_slots(valid_mask: Tensor, max_frames: int) -> tuple[int, ...]:
    """Select evenly spaced valid chunk steps, including both ends when possible."""
    if valid_mask.ndim != 1 or valid_mask.dtype != torch.bool:
        raise ValueError("valid_mask must be a one-dimensional bool tensor")
    if max_frames < 1:
        raise ValueError("max_frames must be positive")
    valid_slots = torch.nonzero(valid_mask, as_tuple=False).flatten()
    if valid_slots.numel() == 0:
        raise ValueError("at least one valid step is required")
    if valid_slots.numel() <= max_frames:
        return tuple(int(slot) for slot in valid_slots.tolist())
    positions = torch.linspace(0, valid_slots.numel() - 1, max_frames).round().long()
    return tuple(int(slot) for slot in valid_slots[positions].tolist())


def _import_lerobot() -> tuple[type[Any], type[Any]]:
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
    except ImportError as exc:
        raise ImportError(
            "LeRobot dataset dependencies are missing. Run: pip install -e '.[lerobot]'"
        ) from exc
    return LeRobotDataset, LeRobotDatasetMetadata


def _scalar(value: Any) -> int | float:
    return torch.as_tensor(value).item()


def load_pusht_chunk(repo_id: str, *, index: int, horizon: int) -> VisualChunk:
    """Load aligned future images and absolute target positions from LeRobot."""
    if horizon < 1:
        raise ValueError("horizon must be positive")
    LeRobotDataset, LeRobotDatasetMetadata = _import_lerobot()
    metadata = LeRobotDatasetMetadata(repo_id)
    if "observation.image" not in metadata.camera_keys:
        raise ValueError(f"{repo_id!r} does not contain observation.image")
    offsets = [step / metadata.fps for step in range(horizon)]
    dataset = LeRobotDataset(
        repo_id,
        delta_timestamps={"observation.image": offsets, "action": offsets},
        video_backend="pyav",
    )
    resolved_index = index if index >= 0 else len(dataset) + index
    if not 0 <= resolved_index < len(dataset):
        raise IndexError(f"index {index} is outside a dataset with {len(dataset)} frames")

    sample = dataset[resolved_index]
    images = torch.as_tensor(sample["observation.image"])
    actions = torch.as_tensor(sample["action"], dtype=torch.float32)
    action_padding = torch.as_tensor(sample["action_is_pad"], dtype=torch.bool)
    image_padding = torch.as_tensor(sample["observation.image_is_pad"], dtype=torch.bool)
    return VisualChunk(
        images=images,
        actions=actions,
        valid_mask=~(action_padding | image_padding),
        state=torch.as_tensor(sample["observation.state"], dtype=torch.float32),
        fps=float(metadata.fps),
        episode_index=int(_scalar(sample["episode_index"])),
        frame_index=int(_scalar(sample["frame_index"])),
        timestamp_s=float(_scalar(sample["timestamp"])),
    )


def _step_color(step: int, horizon: int) -> tuple[int, int, int]:
    fraction = 0.0 if horizon <= 1 else step / (horizon - 1)
    start = (0, 210, 255)
    end = (255, 92, 55)
    return tuple(round(left + fraction * (right - left)) for left, right in zip(start, end))


def _tensor_to_pil(image: Tensor) -> Any:
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError("Pillow is required; install the LeRobot extra") from exc
    image = image.detach().cpu()
    if image.shape[0] in (1, 3, 4):
        image = image.permute(1, 2, 0)
    if image.shape[-1] == 1:
        image = image.expand(-1, -1, 3)
    if image.is_floating_point():
        image = image.clamp(0.0, 1.0).mul(255.0)
    return Image.fromarray(image.to(torch.uint8).numpy())


def _draw_marker(draw: Any, point: tuple[float, float], color: tuple[int, int, int]) -> None:
    x, y = point
    radius = 7
    draw.ellipse(
        (x - radius, y - radius, x + radius, y + radius),
        fill=color,
        outline=(15, 18, 24),
        width=2,
    )


def render_chunk(
    chunk: VisualChunk,
    output_path: Path,
    *,
    workspace_size: float = 512.0,
    max_frames: int = 6,
) -> Path:
    """Render an overview trajectory and a sampled image timeline to a PNG."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as exc:
        raise ImportError("Pillow is required; install the LeRobot extra") from exc

    valid_slots = torch.nonzero(chunk.valid_mask, as_tuple=False).flatten()
    last_valid_slot = int(valid_slots[-1].item())
    timeline_slots = select_timeline_slots(chunk.valid_mask, max_frames)

    margin = 24
    header_height = 66
    overview_size = 480
    thumbnail_size = 180
    label_height = 46
    columns = 3
    rows = (len(timeline_slots) + columns - 1) // columns
    timeline_width = columns * thumbnail_size + (columns - 1) * 14
    timeline_height = rows * (thumbnail_size + label_height) + max(0, rows - 1) * 12
    canvas_width = margin * 3 + overview_size + timeline_width
    canvas_height = header_height + max(overview_size, timeline_height) + 64
    canvas = Image.new("RGB", (canvas_width, canvas_height), (245, 247, 250))
    draw = ImageDraw.Draw(canvas)
    title_font = ImageFont.load_default(size=22)
    body_font = ImageFont.load_default(size=16)
    small_font = ImageFont.load_default(size=13)

    draw.text((margin, 16), "PushT action chunk: images + projected target trajectory", fill=(20, 24, 32), font=title_font)
    subtitle = (
        f"episode {chunk.episode_index} · frame {chunk.frame_index} · "
        f"{chunk.fps:g} Hz · H={chunk.actions.shape[0]} · valid={last_valid_slot + 1}"
    )
    draw.text((margin, 43), subtitle, fill=(80, 86, 98), font=small_font)

    overview_origin = (margin, header_height)
    overview = _tensor_to_pil(chunk.images[0]).resize(
        (overview_size, overview_size), Image.Resampling.BILINEAR
    )
    canvas.paste(overview, overview_origin)
    overview_draw = ImageDraw.Draw(canvas)
    action_pixels = project_workspace_xy(
        chunk.actions[: last_valid_slot + 1],
        image_width=overview_size,
        image_height=overview_size,
        workspace_size=workspace_size,
    )
    state_pixel = project_workspace_xy(
        chunk.state[None],
        image_width=overview_size,
        image_height=overview_size,
        workspace_size=workspace_size,
    )[0]
    points = [tuple(state_pixel.tolist()), *(tuple(point) for point in action_pixels.tolist())]
    for step in range(len(points) - 1):
        color = _step_color(step, last_valid_slot + 1)
        overview_draw.line((points[step], points[step + 1]), fill=color, width=5)
    for step, point in enumerate(action_pixels.tolist()):
        _draw_marker(overview_draw, tuple(point), _step_color(step, last_valid_slot + 1))
    state_x, state_y = state_pixel.tolist()
    overview_draw.ellipse(
        (state_x - 10, state_y - 10, state_x + 10, state_y + 10),
        outline=(255, 255, 255),
        width=4,
    )
    overview_draw.text(
        (overview_origin[0] + 12, overview_origin[1] + 12),
        "anchor image + full future path",
        fill=(255, 255, 255),
        stroke_width=2,
        stroke_fill=(20, 24, 32),
        font=body_font,
    )

    timeline_x = margin * 2 + overview_size
    for panel_index, slot in enumerate(timeline_slots):
        row, column = divmod(panel_index, columns)
        x = timeline_x + column * (thumbnail_size + 14)
        y = header_height + row * (thumbnail_size + label_height + 12)
        thumbnail = _tensor_to_pil(chunk.images[slot]).resize(
            (thumbnail_size, thumbnail_size), Image.Resampling.BILINEAR
        )
        canvas.paste(thumbnail, (x, y))
        point = project_workspace_xy(
            chunk.actions[slot][None],
            image_width=thumbnail_size,
            image_height=thumbnail_size,
            workspace_size=workspace_size,
        )[0]
        panel_draw = ImageDraw.Draw(canvas)
        _draw_marker(panel_draw, (x + point[0].item(), y + point[1].item()), _step_color(slot, last_valid_slot + 1))
        relative_time = slot / chunk.fps
        panel_draw.text(
            (x, y + thumbnail_size + 6),
            f"step {slot:02d} | t+{relative_time:.1f}s\n"
            f"target [{chunk.actions[slot, 0]:.0f}, {chunk.actions[slot, 1]:.0f}]",
            fill=(38, 43, 52),
            font=small_font,
        )

    legend_y = canvas_height - 42
    draw.ellipse((margin, legend_y, margin + 14, legend_y + 14), outline=(255, 255, 255), width=3)
    draw.text((margin + 22, legend_y - 1), "current state", fill=(55, 60, 70), font=small_font)
    gradient_x = margin + 140
    for offset in range(120):
        draw.line(
            (gradient_x + offset, legend_y + 2, gradient_x + offset, legend_y + 12),
            fill=_step_color(offset, 120),
        )
    draw.text(
        (gradient_x + 130, legend_y - 1),
        "early -> late action",
        fill=(55, 60, 70),
        font=small_font,
    )
    draw.text(
        (timeline_x, legend_y - 1),
        f"projection: [0,{workspace_size:g}] x [0,{workspace_size:g}] -> image pixels",
        fill=(80, 86, 98),
        font=small_font,
    )

    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize a LeRobot PushT action chunk")
    parser.add_argument("--dataset", default="lerobot/pusht")
    parser.add_argument("--index", type=int, default=100)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--max-frames", type=int, default=6)
    parser.add_argument("--workspace-size", type=float, default=512.0)
    parser.add_argument("--output", type=Path, default=Path("outputs/lesson02/pusht_chunk.png"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    chunk = load_pusht_chunk(args.dataset, index=args.index, horizon=args.horizon)
    output_path = render_chunk(
        chunk,
        args.output,
        workspace_size=args.workspace_size,
        max_frames=args.max_frames,
    )
    print(f"saved visualization: {output_path}")


if __name__ == "__main__":
    main()
