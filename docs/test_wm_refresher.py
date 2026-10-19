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


class FakeWorldModel(torch.nn.Module):
    """Minimal world model stub for testing."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self._grad = True
        self._eval = False

    def requires_grad_(self, val: bool) -> "FakeWorldModel":
        super().requires_grad_(val)
        self._grad = val
        return self

    def eval(self) -> "FakeWorldModel":
        super().eval()
        self._eval = True
        return self

    def train(self, mode: bool = True) -> "FakeWorldModel":
        super().train(mode)
        self._eval = not mode
        return self


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


class DynamicBuffer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("storage", torch.tensor([7.0]))


class TransactionalWorldModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.sigreg = DynamicBuffer()


class TransactionalTrainer:
    def __init__(self, *, improve: bool = True, fail: bool = False) -> None:
        self.world_model = TransactionalWorldModel()
        self.optimizer = torch.optim.SGD(
            [{
                "params": [self.world_model.weight],
                "lr": 0.1,
                "momentum": 0.9,
                "group_name": "wm_adaptation",
            }]
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=1, gamma=0.5
        )
        self.improve = improve
        self.fail = fail

    def train_one_step(self, *, global_step: int, batch: Any) -> FakeStepMetrics:
        self.optimizer.zero_grad()
        objective = self.world_model.weight.square()
        loss = objective if self.improve else -objective
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        # Mimic SIGReg's RunningBuffer, which rebinds a dynamically sized
        # registered tensor instead of mutating the original object in place.
        self.world_model.sigreg._buffers["storage"] = torch.tensor([1.0, 2.0])
        if self.fail:
            raise RuntimeError("injected refresh failure")
        return FakeStepMetrics()


def validation_metrics(world_model: TransactionalWorldModel) -> dict[str, float]:
    loss = float(world_model.weight.detach().square())
    return {
        "val/loss/total": loss,
        "val/loss/dynamics": loss,
        "val/loss/open_dynamics": loss,
        "val/metric/reward_f1_pri": 0.8,
        "val/metric/reward_positive_count": 4.0,
        "val/metric/effective_rank": 8.0,
    }


def guarded_config(**overrides: Any) -> WMRefreshConfig:
    values = dict(
        updates_per_refresh=1,
        validation_batches=1,
        validation_guard_enabled=True,
        validation_max_total_loss_increase=0.0,
        validation_max_dynamics_loss_increase=0.0,
        validation_max_open_dynamics_loss_increase=0.0,
        validation_loss_absolute_tolerance=0.0,
    )
    values.update(overrides)
    return WMRefreshConfig(**values)


def test_guard_commits_improving_candidate():
    trainer = TransactionalTrainer(improve=True)
    refresher = WorldModelRefresher(
        trainer,
        guarded_config(),
        validation_fn=lambda step: validation_metrics(trainer.world_model),
    )

    result = refresher.refresh(lambda batch_size: "batch")

    assert trainer.world_model.weight.item() < 1.0
    assert trainer.world_model.sigreg.storage.shape == (2,)
    assert result.refresh_step == 1
    assert result.diagnostics["guard/accepted"] == 1.0
    assert result.diagnostics["guard/rejected"] == 0.0


def test_guard_rejects_and_restores_full_refresh_transaction():
    trainer = TransactionalTrainer(improve=False)
    original_lr = trainer.optimizer.param_groups[0]["lr"]
    refresher = WorldModelRefresher(
        trainer,
        guarded_config(max_consecutive_rejections=1),
        validation_fn=lambda step: validation_metrics(trainer.world_model),
    )

    result = refresher.refresh(lambda batch_size: "batch")

    assert trainer.world_model.weight.item() == pytest.approx(1.0)
    assert torch.equal(trainer.world_model.sigreg.storage, torch.tensor([7.0]))
    assert trainer.optimizer.state == {}
    assert trainer.optimizer.param_groups[0]["lr"] == pytest.approx(original_lr)
    assert trainer.scheduler.last_epoch == 0
    assert result.refresh_step == 0
    assert result.diagnostics["guard/rejected"] == 1.0
    assert result.diagnostics["guard/stop_requested"] == 1.0


def test_guard_rolls_back_when_refresh_raises():
    trainer = TransactionalTrainer(improve=True, fail=True)
    refresher = WorldModelRefresher(
        trainer,
        guarded_config(),
        validation_fn=lambda step: validation_metrics(trainer.world_model),
    )

    with pytest.raises(RuntimeError, match="injected refresh failure"):
        refresher.refresh(lambda batch_size: "batch")

    assert trainer.world_model.weight.item() == pytest.approx(1.0)
    assert torch.equal(trainer.world_model.sigreg.storage, torch.tensor([7.0]))
    assert refresher.refresh_step == 0


def test_guard_rolls_back_when_candidate_validation_raises():
    trainer = TransactionalTrainer(improve=True)
    calls = 0

    def failing_validation(step: int) -> dict[str, float]:
        nonlocal calls
        calls += 1
        if calls == 3:  # initial baseline, pre-refresh, then candidate
            raise RuntimeError("injected validation failure")
        return validation_metrics(trainer.world_model)

    refresher = WorldModelRefresher(
        trainer, guarded_config(), validation_fn=failing_validation
    )

    with pytest.raises(RuntimeError, match="injected validation failure"):
        refresher.refresh(lambda batch_size: "batch")

    assert trainer.world_model.weight.item() == pytest.approx(1.0)
    assert torch.equal(trainer.world_model.sigreg.storage, torch.tensor([7.0]))
    assert refresher.refresh_step == 0


def test_guard_requires_fixed_validation_batches():
    config = Config()
    config.phase2.wm_refresh.validation_guard_enabled = True
    config.phase2.wm_refresh.validation_batches = 0
    with pytest.raises(ValueError, match="requires validation_batches > 0"):
        make_phase1_refresh_config(config)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
