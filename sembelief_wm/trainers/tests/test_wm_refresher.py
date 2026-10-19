"""Tests for WorldModelRefresher."""

import pytest
from dataclasses import dataclass
from typing import Any

import torch

from sembelief_wm.config import Config
from sembelief_wm.trainers.wm_refresher import (
    WorldModelRefresher,
    WMRefreshConfig,
    make_phase1_refresh_config,
)
from sembelief_wm.train.trainer import Phase1Trainer


# --- Fakes ---


class FakeWorldModel:
    """Minimal world model stub for testing."""

    def requires_grad_(self, val: bool) -> None:
        self._grad = val

    def eval(self) -> None:
        self._eval = True

    def train(self) -> None:
        self._eval = False


@dataclass
class FakeStepMetrics:
    dynamics_loss: float = 0.5
    reward_loss: float = 0.1
    total_loss: float = 0.6

    def as_dict(self) -> dict[str, float]:
        # Match real Phase1Trainer key format: "loss/dynamics", "loss/reward", "loss/total"
        return {
            "loss/dynamics": self.dynamics_loss,
            "loss/reward": self.reward_loss,
            "loss/total": self.total_loss,
        }


class FakePhase1Trainer:
    """Minimal Phase1Trainer stub."""

    def __init__(self) -> None:
        self.world_model = FakeWorldModel()
        self.optimizer = FakeOptimizer()
        self.scheduler = FakeScheduler()
        self.step_count = 0

    def train_one_step(self, *, global_step: int, batch: Any) -> FakeStepMetrics:
        self.step_count += 1
        return FakeStepMetrics()


class FakeOptimizer:
    param_groups = [{"group_name": "wm_adaptation"}]

    def state_dict(self) -> dict:
        return {"fake": True}

    def load_state_dict(self, d: dict) -> None:
        pass


class FakeScheduler:
    def get_last_lr(self) -> list[float]:
        return [1e-4]

    def state_dict(self) -> dict:
        return {"fake": True}

    def load_state_dict(self, d: dict) -> None:
        pass


# --- Tests ---


def test_should_refresh():
    config = WMRefreshConfig(refresh_every=5)
    refresher = WorldModelRefresher(FakePhase1Trainer(), config)

    assert not refresher.should_refresh(0)  # never at step 0
    assert not refresher.should_refresh(3)
    assert refresher.should_refresh(5)
    assert refresher.should_refresh(10)
    assert not refresher.should_refresh(7)


def test_should_refresh_disabled():
    config = WMRefreshConfig(refresh_every=0)
    refresher = WorldModelRefresher(FakePhase1Trainer(), config)
    assert not refresher.should_refresh(5)
    assert not refresher.should_refresh(10)


def test_refresh_config_and_optimizer_jointly_train_reward_head():
    config = Config()
    config.phase2.wm_refresh.lr = 2e-5
    config.phase2.wm_refresh.reward_head_lr_factor = 0.1
    refresh_config = make_phase1_refresh_config(config)
    assert refresh_config.training.reward_head_lr_factor == pytest.approx(0.1)

    class TinyWorldModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.wm_adapter = torch.nn.Linear(2, 2)
            self.reward_head = torch.nn.Linear(2, 1)

    trainer = Phase1Trainer.__new__(Phase1Trainer)
    trainer.world_model = TinyWorldModel()
    trainer.config = refresh_config
    optimizer = Phase1Trainer._build_optimizer(trainer)
    groups = {group["group_name"]: group for group in optimizer.param_groups}
    assert groups["wm_adaptation"]["lr"] == pytest.approx(2e-5)
    assert groups["reward_head"]["lr"] == pytest.approx(2e-6)
    reward_ids = {id(parameter) for parameter in trainer.world_model.reward_head.parameters()}
    assert {id(parameter) for parameter in groups["reward_head"]["params"]} == reward_ids


def test_refresh_runs_correct_steps():
    trainer = FakePhase1Trainer()
    config = WMRefreshConfig(updates_per_refresh=3, batch_size=4)
    refresher = WorldModelRefresher(trainer, config)

    fake_batch = "dummy_batch"
    sample_fn = lambda bs: fake_batch

    metrics = refresher.refresh(sample_fn)

    assert trainer.step_count == 3
    assert metrics.num_steps == 3
    assert refresher.refresh_step == 3
    assert abs(metrics.avg_dynamics_loss - 0.5) < 1e-6
    assert abs(metrics.avg_reward_loss - 0.1) < 1e-6


def test_refresh_restores_eval_mode():
    trainer = FakePhase1Trainer()
    config = WMRefreshConfig(updates_per_refresh=2)
    refresher = WorldModelRefresher(trainer, config)

    refresher.refresh(lambda bs: "batch")

    # After refresh, WM should be back to eval mode with no grads
    assert not trainer.world_model._grad
    assert trainer.world_model._eval


def test_refresh_restores_each_real_parameter_ownership():
    class OwnershipWorldModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.wm_adapter = torch.nn.Parameter(torch.tensor(1.0))

    class OwnershipTrainer:
        def __init__(self) -> None:
            self.world_model = OwnershipWorldModel()
            self.optimizer = torch.optim.SGD(
                [self.world_model.wm_adapter], lr=0.1
            )
            self.optimizer.param_groups[0]["group_name"] = "wm_adaptation"
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=1
            )

        def train_one_step(self, *, global_step: int, batch: Any) -> FakeStepMetrics:
            del global_step, batch
            self.optimizer.zero_grad()
            self.world_model.wm_adapter.square().backward()
            self.optimizer.step()
            return FakeStepMetrics()

    trainer = OwnershipTrainer()
    refresher = WorldModelRefresher(
        trainer, WMRefreshConfig(updates_per_refresh=1)
    )
    # Mirror Phase-2 assembly: freeze the WM-owned adapter, then attach a
    # separately owned PPO adapter after the refresher captured its Phase-1
    # trainability mask.
    trainer.world_model.requires_grad_(False)
    trainer.world_model.register_parameter(
        "ppo_adapter", torch.nn.Parameter(torch.tensor(2.0))
    )

    refresher.refresh(lambda batch_size: "batch")

    assert not trainer.world_model.wm_adapter.requires_grad
    assert trainer.world_model.ppo_adapter.requires_grad


def test_state_dict_roundtrip():
    trainer = FakePhase1Trainer()
    config = WMRefreshConfig(updates_per_refresh=2)
    refresher = WorldModelRefresher(trainer, config)

    refresher.refresh(lambda bs: "batch")
    state = refresher.state_dict()
    assert state["refresh_step"] == 2

    # New refresher, load state
    trainer2 = FakePhase1Trainer()
    refresher2 = WorldModelRefresher(trainer2, config)
    refresher2.load_state_dict(state)
    assert refresher2.refresh_step == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
