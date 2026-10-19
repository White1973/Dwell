#!/usr/bin/env python3
"""Train, calibrate, gate, and inject a compact multi-horizon Reward Head.

The public functions in this module are also used by the newer Qwen/V-JEPA
reward stage.  Keeping them here removes the last dynamic import of the old
MBRL source tree.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F
from torch import Tensor

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sembelief_wm.data.datasource import TokenizedEpisodeDataset  # noqa: E402
from sembelief_wm.model import QwenTransitionBackbone, WorldModel  # noqa: E402
from sembelief_wm.types import BeliefState  # noqa: E402


def require_two_classes(name: str, labels: Tensor) -> None:
    values = torch.unique(labels.detach().float().cpu())
    if len(values) != 2 or set(values.tolist()) != {0.0, 1.0}:
        raise RuntimeError(
            f"{name} must contain positive and negative samples; "
            f"observed={values.tolist()} count={labels.numel()}"
        )


def _auc(probabilities: Tensor, labels: Tensor) -> float:
    probabilities = probabilities.detach().float().cpu().flatten()
    labels = labels.detach().float().cpu().flatten()
    positives = int((labels > 0.5).sum())
    negatives = int((labels <= 0.5).sum())
    if positives == 0 or negatives == 0:
        return float("nan")
    order = torch.argsort(probabilities, stable=True)
    ranks = torch.empty_like(order, dtype=torch.float64)
    ranks[order] = torch.arange(1, len(order) + 1, dtype=torch.float64)
    # Average tied ranks so exact ties contribute 0.5 to AUC.
    unique, inverse, counts = torch.unique(
        probabilities, sorted=True, return_inverse=True, return_counts=True
    )
    del unique
    if bool((counts > 1).any()):
        for group in torch.nonzero(counts > 1, as_tuple=False).flatten().tolist():
            mask = inverse == group
            ranks[mask] = ranks[mask].mean()
    positive_rank_sum = ranks[labels > 0.5].sum()
    value = (
        positive_rank_sum - positives * (positives + 1) / 2
    ) / (positives * negatives)
    return float(value)


def _average_precision(probabilities: Tensor, labels: Tensor) -> float:
    probabilities = probabilities.detach().float().cpu().flatten()
    labels = labels.detach().float().cpu().flatten()
    positives = int((labels > 0.5).sum())
    if positives == 0:
        return float("nan")
    order = torch.argsort(probabilities, descending=True, stable=True)
    truth = (labels[order] > 0.5).float()
    precision = truth.cumsum(0) / torch.arange(1, len(truth) + 1)
    return float((precision * truth).sum() / positives)


def metrics(
    logits: Tensor,
    labels: Tensor,
    *,
    decision_threshold: float = 0.5,
) -> dict[str, float]:
    logits = logits.detach().float().flatten()
    labels = labels.detach().float().flatten()
    probabilities = torch.sigmoid(logits)
    truth = labels > 0.5
    predicted = probabilities >= decision_threshold
    tp = int((predicted & truth).sum())
    fp = int((predicted & ~truth).sum())
    tn = int((~predicted & ~truth).sum())
    fn = int((~predicted & truth).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    positive_rate = float(labels.mean()) if labels.numel() else float("nan")
    return {
        "auc": _auc(probabilities, labels),
        "average_precision": _average_precision(probabilities, labels),
        "accuracy": (tp + tn) / max(tp + fp + tn + fn, 1),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_rate": fp / max(fp + tn, 1),
        "brier": float((probabilities - labels).square().mean()),
        "brier_baseline": positive_rate * (1.0 - positive_rate),
        "positive_rate": positive_rate,
        "predicted_positive_rate": float(predicted.float().mean()),
        "decision_threshold": float(decision_threshold),
        "samples": float(labels.numel()),
    }


def metrics_by_horizon(
    logits: Tensor,
    labels: Tensor,
    horizon_ids: Tensor,
    *,
    horizons: Iterable[int],
    decision_threshold: float = 0.5,
) -> dict[str, dict[str, float]]:
    return {
        str(horizon): metrics(
            logits[horizon_ids == horizon],
            labels[horizon_ids == horizon],
            decision_threshold=decision_threshold,
        )
        for horizon in horizons
    }


def select_f1_threshold(logits: Tensor, labels: Tensor) -> float:
    probabilities = torch.sigmoid(logits.detach().float()).cpu()
    labels = labels.detach().float().cpu()
    candidates = torch.unique(probabilities).sort(descending=True).values
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in candidates:
        value = metrics(
            logits.detach().cpu(), labels, decision_threshold=float(threshold)
        )["f1"]
        if value > best_f1:
            best_f1 = value
            best_threshold = float(threshold)
    return best_threshold


def select_precision_threshold(
    logits: Tensor,
    labels: Tensor,
    *,
    target_precision: float,
    min_recall: float,
) -> tuple[float, float, float]:
    probabilities = torch.sigmoid(logits.detach().float()).cpu()
    labels = labels.detach().float().cpu()
    candidates = torch.unique(probabilities).sort(descending=True).values
    best: tuple[float, float, float] | None = None
    for threshold in candidates:
        result = metrics(
            logits.detach().cpu(), labels, decision_threshold=float(threshold)
        )
        if result["precision"] < target_precision or result["recall"] < min_recall:
            continue
        candidate = (float(threshold), result["precision"], result["recall"])
        if best is None or (candidate[2], candidate[1]) > (best[2], best[1]):
            best = candidate
    if best is None:
        raise RuntimeError(
            "no calibration threshold satisfies "
            f"precision>={target_precision} and recall>={min_recall}"
        )
    return best


def _last_linear(head: torch.nn.Module) -> torch.nn.Linear:
    classifier = head.net if head.net is not None else head.compact_net
    if isinstance(classifier, torch.nn.Linear):
        return classifier
    if isinstance(classifier, torch.nn.Sequential):
        for module in reversed(classifier):
            if isinstance(module, torch.nn.Linear):
                return module
    raise TypeError("Reward Head has no final linear layer")


def calibrate(
    head: torch.nn.Module,
    features: Tensor,
    labels: Tensor,
    *,
    steps: int = 300,
    lr: float = 5e-2,
) -> tuple[float, float]:
    """Fit scalar temperature/bias and fold them into the Reward Head."""
    head.eval()
    with torch.no_grad():
        raw_logits = head.forward_pooled(features).detach().float()
    log_scale = torch.zeros((), device=raw_logits.device, requires_grad=True)
    bias = torch.zeros((), device=raw_logits.device, requires_grad=True)
    optimizer = torch.optim.Adam([log_scale, bias], lr=lr)
    targets = labels.to(raw_logits.device).float()
    for _ in range(steps):
        calibrated = log_scale.exp() * raw_logits + bias
        loss = F.binary_cross_entropy_with_logits(calibrated, targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    scale_value = float(log_scale.detach().exp())
    bias_value = float(bias.detach())
    last = _last_linear(head)
    with torch.no_grad():
        last.weight.mul_(scale_value)
        if last.bias is None:
            raise ValueError("Reward Head final affine layer requires a bias")
        last.bias.mul_(scale_value).add_(bias_value)
    return scale_value, bias_value


def _sample_specs(
    dataset: TokenizedEpisodeDataset,
    indices: list[int],
    horizons: list[int],
    *,
    seed: int,
    success_threshold: float,
    terminal_sample_probability: float,
    windows_per_episode: int,
    independent_horizon_starts: bool,
) -> list[tuple[int, int, int]]:
    if not independent_horizon_starts:
        raise ValueError(
            "This local implementation requires independent_horizon_starts=true"
        )
    rng = random.Random(seed)
    specs: list[tuple[int, int, int]] = []
    for index in indices:
        episode = dataset.episodes[index]
        length = int(episode.episode_length)
        for horizon in horizons:
            if length < horizon:
                continue
            starts = list(range(length - horizon + 1))
            positive = [
                start
                for start in starts
                if float(episode.rewards[start + horizon - 1]) > success_threshold
            ]
            if episode.env_id == "frozenlake":
                from sembelief_wm.data.termination import episode_events
                events = episode_events(episode)
                positive = [start for start in starts if events[start+horizon-1] != 0]
            for _ in range(windows_per_episode):
                # The probability controls explicit terminal oversampling. A
                # non-forced draw remains uniform and may naturally be positive.
                if positive and rng.random() < terminal_sample_probability:
                    start = rng.choice(positive)
                else:
                    start = rng.choice(starts)
                specs.append((index, start, horizon))
    rng.shuffle(specs)
    return specs


@torch.no_grad()
def collect_multi_horizon_features(
    *,
    world_model: WorldModel,
    dataset: TokenizedEpisodeDataset,
    indices: list[int],
    device: torch.device,
    dtype: torch.dtype,
    null_action: Tensor,
    seed: int,
    horizons: list[int],
    success_threshold: float,
    terminal_sample_probability: float,
    windows_per_episode: int,
    independent_horizon_starts: bool,
    feature_batch_size: int,
    progress_label: str,
) -> tuple[Tensor, Tensor, Tensor]:
    """Collect pooled recursive-prior endpoints from independently sampled starts."""
    if not 0.0 <= terminal_sample_probability <= 1.0:
        raise ValueError("terminal_sample_probability must be in [0, 1]")
    if feature_batch_size <= 0 or windows_per_episode <= 0:
        raise ValueError("feature_batch_size and windows_per_episode must be positive")
    specs = _sample_specs(
        dataset,
        indices,
        horizons,
        seed=seed,
        success_threshold=success_threshold,
        terminal_sample_probability=terminal_sample_probability,
        windows_per_episode=windows_per_episode,
        independent_horizon_starts=independent_horizon_starts,
    )
    if not specs:
        raise RuntimeError(f"{progress_label}: no eligible multi-horizon windows")
    world_model.eval()
    features: list[Tensor] = []
    labels: list[Tensor] = []
    horizon_values: list[Tensor] = []
    env_map = {
        env_id: index for index, env_id in enumerate(world_model.config.env.env_ids)
    }
    # Grouping by H permits one dense recursive prior rollout per mini-batch.
    for horizon in horizons:
        group = [spec for spec in specs if spec[2] == horizon]
        for offset in range(0, len(group), feature_batch_size):
            chunk = group[offset : offset + feature_batch_size]
            episodes = [dataset.episodes[index] for index, _, _ in chunk]
            starts = [start for _, start, _ in chunk]
            batch_size = len(chunk)
            belief = world_model.get_initial_belief(
                batch_size, device=device, dtype=dtype
            )
            env_ids = torch.tensor(
                [env_map[episode.env_id] for episode in episodes],
                device=device,
                dtype=torch.long,
            )
            for step in range(max(starts) + 1):
                observation_tokens = torch.stack(
                    [episode.obs_tokens[min(step, start)] for episode, start in zip(episodes, starts, strict=True)]
                ).to(device=device, dtype=dtype)
                previous_actions = torch.tensor(
                    [
                        int(null_action.flatten()[0])
                        if step == 0
                        else int(episode.actions[min(step - 1, start)])
                        for episode, start in zip(episodes, starts, strict=True)
                    ],
                    device=device,
                    dtype=torch.long,
                )
                candidate = world_model.posterior_step(
                    prev_belief=belief,
                    prev_actions=previous_actions,
                    observation_tokens=observation_tokens,
                    env_ids=env_ids,
                )
                active = torch.tensor(
                    [step <= start for start in starts], device=device
                ).view(batch_size, 1, 1)
                belief = BeliefState(
                    slots=torch.where(active, candidate.slots, belief.slots)
                )
            actions = torch.stack(
                [
                    episode.actions[start : start + horizon].long()
                    for episode, start in zip(episodes, starts, strict=True)
                ]
            ).to(device)
            rollout = world_model.rollout_prior(belief, actions, env_ids=env_ids)
            endpoint = rollout.beliefs[:, -1]
            pooled = world_model.reward_head.readout(endpoint).float().cpu()
            target = torch.tensor(
                [
                    float(episode.rewards[start + horizon - 1]) > success_threshold
                    for episode, start in zip(episodes, starts, strict=True)
                ],
                dtype=torch.float32,
            )
            if getattr(world_model.config.reward, "event_classes", 1) == 3:
                from sembelief_wm.data.termination import episode_events
                target = torch.tensor([episode_events(ep)[start+horizon-1]
                                       for ep, start in zip(episodes, starts)], dtype=torch.long)
            features.append(pooled)
            labels.append(target)
            horizon_values.append(
                torch.full((batch_size,), horizon, dtype=torch.long)
            )
            completed = min(offset + batch_size, len(group))
            if completed == len(group) or completed % 256 == 0:
                print(
                    f"[{progress_label}/H{horizon}] {completed}/{len(group)}",
                    flush=True,
                )
    return torch.cat(features), torch.cat(labels), torch.cat(horizon_values)


def _fold_standardization(head: torch.nn.Module, mean: Tensor, std: Tensor) -> None:
    classifier = head.net if head.net is not None else head.compact_net
    assert classifier is not None
    first = classifier if isinstance(classifier, torch.nn.Linear) else classifier[0]
    if not isinstance(first, torch.nn.Linear):
        raise TypeError("Reward Head must begin with a linear layer")
    with torch.no_grad():
        weight = first.weight.detach().clone()
        first.weight.copy_(weight / std.unsqueeze(0))
        if first.bias is None:
            raise ValueError("Reward Head first affine layer requires a bias")
        first.bias.sub_((weight * (mean / std).unsqueeze(0)).sum(dim=1))


def _index_digest(indices: list[int]) -> str:
    return hashlib.sha256(
        ",".join(str(index) for index in indices).encode("ascii")
    ).hexdigest()


def _atomic_torch_save(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _make_splits(
    *,
    checkpoint_path: Path,
    dataset_size: int,
    seed: int,
    n_train: int,
    n_val: int,
    n_calibration: int,
    n_test: int,
) -> tuple[dict[str, list[int]], str]:
    run_config_path = checkpoint_path.parent / "run_config.json"
    rng = random.Random(seed)
    if run_config_path.is_file():
        run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
        stage1_train = [int(value) for value in run_config.get("train_indices", [])]
        stage1_test = [
            int(value) for value in run_config.get("validation_indices", [])
        ]
        if stage1_train and stage1_test:
            rng.shuffle(stage1_train)
            rng.shuffle(stage1_test)
            needed_train = n_train + n_val + n_calibration
            if len(stage1_train) < needed_train or len(stage1_test) < n_test:
                raise ValueError(
                    "Phase-1 split is too small for requested Reward-Head sizes: "
                    f"train_pool={len(stage1_train)} need={needed_train}, "
                    f"test_pool={len(stage1_test)} need={n_test}"
                )
            return (
                {
                    "train": stage1_train[:n_train],
                    "validation": stage1_train[n_train : n_train + n_val],
                    "calibration": stage1_train[
                        n_train + n_val : n_train + n_val + n_calibration
                    ],
                    "test": stage1_test[:n_test],
                },
                str(run_config_path.resolve()),
            )
    indices = list(range(dataset_size))
    rng.shuffle(indices)
    needed = n_train + n_val + n_calibration + n_test
    if len(indices) < needed:
        raise ValueError(
            f"Dataset has {len(indices)} episodes but Reward Head requests {needed}"
        )
    return (
        {
            "train": indices[:n_train],
            "validation": indices[n_train : n_train + n_val],
            "calibration": indices[n_train + n_val : n_train + n_val + n_calibration],
            "test": indices[n_train + n_val + n_calibration : needed],
        },
        "deterministic_fallback_no_phase1_run_config",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inject a gated H1..H5 compact Reward Head into a Phase-1 WM"
    )
    parser.add_argument("--source-checkpoint", required=True)
    parser.add_argument("--event-classes", type=int, choices=[1, 3], default=1)
    parser.add_argument("--wandb-project", default="jiayu-mbrl")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--output-checkpoint", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--feature-cache", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--n-train", type=int, default=3000)
    parser.add_argument("--n-val", type=int, default=1000)
    parser.add_argument("--n-calibration", type=int, default=1000)
    parser.add_argument("--n-test", type=int, default=1000)
    parser.add_argument("--windows-per-episode", type=int, default=2)
    parser.add_argument("--eval-windows-per-episode", type=int, default=1)
    parser.add_argument("--terminal-sample-probability", type=float, default=0.5)
    parser.add_argument("--evaluation-terminal-sample-probability", type=float, default=0.5)
    parser.add_argument("--success-threshold", type=float, default=1.0)
    parser.add_argument("--feature-batch-size", type=int, default=8)
    parser.add_argument("--head-hidden-dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--pos-weight", type=float, default=1.0)
    parser.add_argument("--selection-metric", choices=("average_precision", "auc"), default="average_precision")
    parser.add_argument("--calibration-steps", type=int, default=300)
    parser.add_argument("--calibration-lr", type=float, default=5e-2)
    parser.add_argument("--target-precision", type=float, default=0.85)
    parser.add_argument("--min-target-recall", type=float, default=0.30)
    parser.add_argument("--min-heldout-precision", type=float, default=0.80)
    parser.add_argument("--min-heldout-recall", type=float, default=0.25)
    args = parser.parse_args()

    if len(set(args.horizons)) != len(args.horizons) or any(
        horizon <= 0 for horizon in args.horizons
    ):
        parser.error("--horizons must be unique positive integers")
    for name in ("n_train", "n_val", "n_calibration", "n_test"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    checkpoint_path = Path(args.source_checkpoint).resolve()
    output_path = Path(args.output_checkpoint).resolve()
    data_dir = Path(args.data_dir).resolve()
    cache_path = Path(args.feature_cache).resolve()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")
    payload = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False, mmap=True
    )
    if not isinstance(payload, dict) or "config" not in payload or "model" not in payload:
        raise ValueError("source checkpoint must contain config and model")
    config = copy.deepcopy(payload["config"])
    config.reward.head_hidden_dim = args.head_hidden_dim
    config.reward.event_classes = args.event_classes
    config.reward.success_reward_threshold = args.success_threshold
    dataset = TokenizedEpisodeDataset.from_directory(data_dir)
    split_indices, split_source = ({name: [i for i, ep in enumerate(dataset.episodes) if ep.split == name]
                                  for name in ("train", "validation", "calibration", "test")}, "map_hash_v1") if args.event_classes == 3 else _make_splits(
        checkpoint_path=checkpoint_path,
        dataset_size=len(dataset),
        seed=args.seed,
        n_train=args.n_train,
        n_val=args.n_val,
        n_calibration=args.n_calibration,
        n_test=args.n_test,
    )
    split_sets = {name: set(values) for name, values in split_indices.items()}
    for left, left_values in split_sets.items():
        for right, right_values in split_sets.items():
            if left < right and left_values & right_values:
                raise RuntimeError(f"Reward split overlap: {left} vs {right}")
    print(
        "Reward splits: "
        + " ".join(f"{name}={len(values)}" for name, values in split_indices.items())
        + f" source={split_source}",
        flush=True,
    )

    backbone = QwenTransitionBackbone.from_config(config, device_map={"": device})
    world_model = WorldModel(config, backbone).to(device)
    source_state = payload["model"]
    frozen_state = {
        key: value for key, value in source_state.items() if not key.startswith("reward_head.")
    }
    incompatible = world_model.load_state_dict(frozen_state, strict=False)
    missing = [
        key for key in incompatible.missing_keys if not key.startswith("reward_head.")
    ]
    if missing or incompatible.unexpected_keys:
        raise RuntimeError(
            f"frozen WM state mismatch: missing={missing[:20]} "
            f"unexpected={list(incompatible.unexpected_keys)[:20]}"
        )
    world_model.eval().requires_grad_(False)
    world_model.reward_head.requires_grad_(True)
    dtype = next(world_model.parameters()).dtype
    null_action = torch.tensor(
        [config.env.null_action_id], device=device, dtype=torch.long
    )

    checkpoint_stat = checkpoint_path.stat()
    manifest_stat = (data_dir / "manifest.jsonl").stat()
    cache_spec = {
        "event_classes": args.event_classes,
        "format": "lewm_reward_features_v1",
        "checkpoint": str(checkpoint_path),
        "checkpoint_size": checkpoint_stat.st_size,
        "checkpoint_mtime_ns": checkpoint_stat.st_mtime_ns,
        "manifest": str((data_dir / "manifest.jsonl").resolve()),
        "manifest_size": manifest_stat.st_size,
        "manifest_mtime_ns": manifest_stat.st_mtime_ns,
        "horizons": args.horizons,
        "split_digests": {
            name: _index_digest(values) for name, values in split_indices.items()
        },
        "seed": args.seed,
        "windows_per_episode": args.windows_per_episode,
        "eval_windows_per_episode": args.eval_windows_per_episode,
        "terminal_sample_probability": args.terminal_sample_probability,
        "evaluation_terminal_sample_probability": args.evaluation_terminal_sample_probability,
        "success_threshold": args.success_threshold,
    }
    if cache_path.is_file():
        cache = torch.load(cache_path, map_location="cpu", weights_only=False)
        if cache.get("spec") != cache_spec:
            raise ValueError(f"incompatible feature cache: {cache_path}")
        collected = cache["features"]
        print(f"Reusing Reward-Head feature cache: {cache_path}", flush=True)
    else:
        collected: dict[str, tuple[Tensor, Tensor, Tensor]] = {}
        for offset, name in enumerate(("train", "validation", "calibration", "test")):
            is_train = name == "train"
            collected[name] = collect_multi_horizon_features(
                world_model=world_model,
                dataset=dataset,
                indices=split_indices[name],
                device=device,
                dtype=dtype,
                null_action=null_action,
                seed=args.seed + offset,
                horizons=args.horizons,
                success_threshold=args.success_threshold,
                terminal_sample_probability=(
                    args.terminal_sample_probability
                    if is_train
                    else args.evaluation_terminal_sample_probability
                ),
                windows_per_episode=(
                    args.windows_per_episode
                    if is_train
                    else args.eval_windows_per_episode
                ),
                independent_horizon_starts=True,
                feature_batch_size=args.feature_batch_size,
                progress_label=name,
            )
        _atomic_torch_save({"spec": cache_spec, "features": collected}, cache_path)
        print(f"Saved Reward-Head feature cache: {cache_path}", flush=True)

    if args.event_classes == 3:
        from sembelief_wm.train.event_reward import fit_event_reward
        fit_event_reward(args=args, head=world_model.reward_head, collected=collected,
                         payload=payload, config=config, source_state=source_state,
                         split_indices=split_indices, output_path=output_path,
                         fold_standardization=_fold_standardization, last_linear=_last_linear,
                         atomic_save=_atomic_torch_save)
        return

    for name, (_, labels, horizon_ids) in collected.items():
        require_two_classes(name, labels)
        for horizon in args.horizons:
            require_two_classes(f"{name}/H{horizon}", labels[horizon_ids == horizon])
        print(
            f"{name}: samples={len(labels)} positive_rate={float(labels.mean()):.4f}",
            flush=True,
        )

    train_x, train_y, _ = collected["train"]
    val_x, val_y, _ = collected["validation"]
    calibration_x, calibration_y, _ = collected["calibration"]
    test_x, test_y, test_h = collected["test"]
    train_x, train_y = train_x.to(device), train_y.to(device)
    val_x, val_y = val_x.to(device), val_y.to(device)
    calibration_x, calibration_y = calibration_x.to(device), calibration_y.to(device)
    test_x, test_y, test_h = test_x.to(device), test_y.to(device), test_h.to(device)
    feature_mean = train_x.mean(0)
    feature_std = train_x.std(0, unbiased=False).clamp_min(1e-4)
    train_z = (train_x - feature_mean) / feature_std
    val_z = (val_x - feature_mean) / feature_std

    optimizer = torch.optim.AdamW(
        world_model.reward_head.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    pos_weight = torch.tensor(args.pos_weight, device=device)
    generator = torch.Generator().manual_seed(args.seed + 17)
    best_score = -float("inf")
    best_state: dict[str, Tensor] | None = None
    stale = 0
    for epoch in range(1, args.epochs + 1):
        world_model.reward_head.train()
        permutation = torch.randperm(len(train_y), generator=generator)
        total_loss = 0.0
        for start in range(0, len(permutation), args.batch_size):
            indices = permutation[start : start + args.batch_size].to(device)
            logits = world_model.reward_head.forward_pooled(train_z[indices])
            loss = F.binary_cross_entropy_with_logits(
                logits, train_y[indices], pos_weight=pos_weight
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(world_model.reward_head.parameters(), 5.0)
            optimizer.step()
            total_loss += float(loss.detach()) * len(indices)
        world_model.reward_head.eval()
        with torch.no_grad():
            validation = metrics(
                world_model.reward_head.forward_pooled(val_z), val_y
            )
        score = float(validation[args.selection_metric])
        if math.isfinite(score) and score > best_score + 1e-6:
            best_score = score
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in world_model.reward_head.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
        print(
            f"[reward epoch {epoch}] loss={total_loss/max(len(train_y), 1):.6f} "
            f"val_auc={validation['auc']:.4f} val_ap={validation['average_precision']:.4f} "
            f"val_brier={validation['brier']:.5f} best={best_score:.4f}",
            flush=True,
        )
        if stale >= args.patience:
            print(f"Reward Head early stopping at epoch {epoch}", flush=True)
            break
    if best_state is None:
        raise RuntimeError("no finite Reward-Head candidate selected")
    world_model.reward_head.load_state_dict(best_state)
    world_model.reward_head.to(device).eval()
    _fold_standardization(world_model.reward_head, feature_mean, feature_std)
    scale, calibration_bias = calibrate(
        world_model.reward_head,
        calibration_x,
        calibration_y,
        steps=args.calibration_steps,
        lr=args.calibration_lr,
    )
    with torch.no_grad():
        calibration_logits = world_model.reward_head.forward_pooled(calibration_x)
        threshold, calibration_precision, calibration_recall = select_precision_threshold(
            calibration_logits,
            calibration_y,
            target_precision=args.target_precision,
            min_recall=args.min_target_recall,
        )
        test_logits = world_model.reward_head.forward_pooled(test_x)
        calibration_values = metrics(
            calibration_logits, calibration_y, decision_threshold=threshold
        )
        test_values = metrics(test_logits, test_y, decision_threshold=threshold)
        test_by_horizon = metrics_by_horizon(
            test_logits,
            test_y,
            test_h,
            horizons=args.horizons,
            decision_threshold=threshold,
        )
    failures: list[str] = []
    if test_values["precision"] < args.min_heldout_precision:
        failures.append(
            f"test precision={test_values['precision']:.4f} < {args.min_heldout_precision:.4f}"
        )
    if test_values["recall"] < args.min_heldout_recall:
        failures.append(
            f"test recall={test_values['recall']:.4f} < {args.min_heldout_recall:.4f}"
        )
    if test_values["brier"] >= test_values["brier_baseline"]:
        failures.append(
            f"test brier={test_values['brier']:.5f} is not below "
            f"baseline={test_values['brier_baseline']:.5f}"
        )
    gate = {"passed": not failures, "failures": failures}

    new_state = {
        key: value for key, value in source_state.items() if not key.startswith("reward_head.")
    }
    for key, value in world_model.reward_head.state_dict().items():
        new_state[f"reward_head.{key}"] = value.detach().cpu()
    new_checkpoint = {
        key: value
        for key, value in payload.items()
        if key not in {"model", "optimizer", "scheduler"}
    }
    new_checkpoint.update(
        {
            "config": config,
            "model": new_state,
            "injected_reward_head": True,
            "reward_head_injection": {
                "version": 6,
                "head_hidden_dim": args.head_hidden_dim,
                "horizon": None,
                "horizons": args.horizons,
                "independent_horizon_starts": True,
                "success_threshold": args.success_threshold,
                "decision_threshold": threshold,
                "calibration_scale": scale,
                "calibration_bias": calibration_bias,
                "calibration_metrics": calibration_values,
                "test_metrics": test_values,
                "test_metrics_by_horizon": test_by_horizon,
                "split_indices": split_indices,
                "split_source": split_source,
                "test_used_for_training_selection_or_calibration": False,
                "gate": gate,
            },
        }
    )
    _atomic_torch_save(new_checkpoint, output_path)
    report_path = output_path.with_suffix(".report.json")
    report_path.write_text(
        json.dumps(
            {
                "format": "lewm_reward_head_gate_v1",
                "source_checkpoint": str(checkpoint_path),
                "output_checkpoint": str(output_path),
                "calibration": calibration_values,
                "test": test_values,
                "test_by_horizon": test_by_horizon,
                "gate": gate,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        f"Reward Head held-out: AUC={test_values['auc']:.4f} "
        f"AP={test_values['average_precision']:.4f} "
        f"precision={test_values['precision']:.4f} recall={test_values['recall']:.4f} "
        f"threshold={threshold:.6f}",
        flush=True,
    )
    if failures:
        raise RuntimeError(
            "Reward Head gate failed; rejected checkpoint retained for diagnosis: "
            + "; ".join(failures)
        )
    print(f"Reward Head Gate: PASS ({output_path})", flush=True)


if __name__ == "__main__":
    main()
