"""Convert online latent trajectories into real-return PPO/critic batches."""
from __future__ import annotations

import torch

from .trajectory import PPOBatch


def collection_to_ppo_batch(
    result, *, gamma: float, device, reward_scale: float = 0.1,
    first_transition_only: bool = False,
) -> PPOBatch | None:
    states, actions, log_probs, old_values, returns, advantages = [], [], [], [], [], []
    for episode in result.episodes:
        trajectory = episode.info.get("_policy_trajectory")
        if not trajectory:
            continue
        rewards = trajectory["rewards"].float() * reward_scale
        episode_returns = torch.empty_like(rewards)
        running = torch.zeros(())
        for index in range(len(rewards) - 1, -1, -1):
            running = rewards[index] + gamma * running
            episode_returns[index] = running
        values = trajectory["values"].float().reshape(-1)
        selected = slice(0, 1) if first_transition_only else slice(None)
        states.append(trajectory["states"][selected])
        actions.append(trajectory["actions"].reshape(-1)[selected])
        log_probs.append(trajectory["log_probs"].reshape(-1)[selected])
        old_values.append(values[selected])
        returns.append(episode_returns[selected])
        advantages.append((episode_returns - values)[selected])
    if not states:
        return None
    return PPOBatch(
        states=torch.cat(states).to(device),
        actions=torch.cat(actions).to(device),
        advantages=torch.cat(advantages).to(device),
        returns=torch.cat(returns).to(device),
        old_log_probs=torch.cat(log_probs).to(device),
        old_values=torch.cat(old_values).to(device),
    )


def normalize_advantages(batch: PPOBatch) -> PPOBatch:
    advantages = batch.advantages.float()
    if advantages.numel() > 1:
        advantages = (advantages - advantages.mean()) / advantages.std().clamp_min(1e-8)
    return PPOBatch(**{
        **{name: getattr(batch, name) for name in PPOBatch.__dataclass_fields__},
        "advantages": advantages.to(batch.advantages.dtype),
    })
