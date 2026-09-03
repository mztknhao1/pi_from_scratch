"""Partial-observability mechanism checks for the MEM lesson."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from pi_from_scratch.memory import LongTermTextMemory, ShortTermVideoMemory, VisualMemoryFrame

METHODS = ("no_memory", "short_only", "long_only", "multi_scale")


def _uses_short(method: str) -> bool:
    return method in {"short_only", "multi_scale"}


def _uses_long(method: str) -> bool:
    return method in {"long_only", "multi_scale"}


def _occlusion_accuracy(method: str, *, episodes: int, seed: int) -> float:
    generator = torch.Generator().manual_seed(seed)
    correct = 0
    for episode in range(episodes):
        target_side = -1.0 if torch.rand((), generator=generator).item() < 0.5 else 1.0
        memory = ShortTermVideoMemory(capacity=6, feature_dim=1)
        memory.append(
            VisualMemoryFrame(
                timestamp_s=0.0,
                features=torch.tensor([target_side]),
                visible_mask=torch.tensor([True]),
            )
        )
        occlusion_steps = int(torch.randint(1, 6, (), generator=generator).item())
        for step in range(1, occlusion_steps + 1):
            memory.append(
                VisualMemoryFrame(
                    timestamp_s=float(step),
                    features=torch.tensor([0.0]),
                    visible_mask=torch.tensor([False]),
                )
            )
        if _uses_short(method):
            prediction = float(memory.encode().values[0].item())
        else:
            prediction = 1.0
        correct += int(prediction == target_side)
    return correct / episodes


def _long_horizon_progress(method: str) -> float:
    subtasks = ("prepare pan", "add bread", "add cheese", "close sandwich")
    memory = LongTermTextMemory()
    completed: set[str] = set()
    for _ in subtasks:
        if _uses_long(method):
            next_subtask = next(item for item in subtasks if not memory.completed(item))
            memory.update(next_subtask, success=True)
        else:
            next_subtask = subtasks[0]
        completed.add(next_subtask)
    return len(completed) / len(subtasks)


def run_memory_experiment(*, episodes: int = 200, seed: int = 13) -> dict[str, object]:
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    results: dict[str, dict[str, float]] = {}
    for method in METHODS:
        occlusion = _occlusion_accuracy(method, episodes=episodes, seed=seed)
        long_horizon = _long_horizon_progress(method)
        results[method] = {
            "occlusion_accuracy": occlusion,
            "long_horizon_progress": long_horizon,
            "mean_mechanism_score": (occlusion + long_horizon) / 2.0,
        }
    return {
        "experiment_scope": "controlled memory mechanism check, not learned-policy performance",
        "episodes": episodes,
        "seed": seed,
        "methods": results,
        "short_term": {
            "capacity_frames": 6,
            "encoder": "latest visible value per feature (teaching proxy)",
        },
        "long_term": {
            "representation": "structured text summary (deterministic teaching proxy)",
        },
    }


def write_memory_metrics(path: Path, metrics: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_memory_comparison_svg(path: Path, metrics: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    methods = metrics["methods"]
    labels = {
        "no_memory": "无记忆",
        "short_only": "仅短期视觉",
        "long_only": "仅长期文本",
        "multi_scale": "多尺度记忆",
    }
    colors = {"occlusion_accuracy": "#287c78", "long_horizon_progress": "#dd6b55"}
    bars = []
    start_x = 95
    for index, method in enumerate(METHODS):
        x = start_x + index * 235
        result = methods[method]  # type: ignore[index]
        for offset, key in ((0, "occlusion_accuracy"), (62, "long_horizon_progress")):
            value = float(result[key])
            height = 250 * value
            bars.append(
                f'<rect x="{x + offset}" y="{385 - height:.1f}" width="52" height="{height:.1f}" rx="6" fill="{colors[key]}"/>'
            )
            bars.append(
                f'<text x="{x + offset + 26}" y="{375 - height:.1f}" text-anchor="middle" font-family="sans-serif" font-size="14" font-weight="700">{value:.2f}</text>'
            )
        bars.append(
            f'<text x="{x + 57}" y="420" text-anchor="middle" font-family="sans-serif" font-size="15">{labels[method]}</text>'
        )
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="500" viewBox="0 0 1100 500">
<rect width="1100" height="500" fill="#f7f4ed"/>
<text x="50" y="50" font-family="sans-serif" font-size="28" font-weight="700" fill="#17212b">MEM：短期视觉与长期语义解决不同的遗忘</text>
<text x="50" y="82" font-family="sans-serif" font-size="15" fill="#4b5563">受控机制实验；分数 1.0 表示该类信息被完整保留</text>
<line x1="65" y1="385" x2="1040" y2="385" stroke="#9ca3af"/>
<line x1="65" y1="135" x2="65" y2="385" stroke="#9ca3af"/>
{"".join(bars)}
<rect x="330" y="455" width="18" height="18" rx="3" fill="#287c78"/><text x="358" y="469" font-family="sans-serif" font-size="15">遮挡后定位</text>
<rect x="565" y="455" width="18" height="18" rx="3" fill="#dd6b55"/><text x="593" y="469" font-family="sans-serif" font-size="15">长任务阶段进度</text>
</svg>
"""
    path.write_text(svg, encoding="utf-8")
