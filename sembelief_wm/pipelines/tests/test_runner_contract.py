from __future__ import annotations

import os
from pathlib import Path
import subprocess


def test_alternating_runner_emits_exact_h2_guarded_command(tmp_path: Path) -> None:
    repository = Path(__file__).resolve().parents[3]
    environment = os.environ.copy()
    environment.update({
        "PYTHON_BIN": "/bin/echo",
        "OUTPUT_DIR": str(tmp_path),
        "NO_WANDB": "1",
        "ACTOR_UPDATES": "1",
        "TOTAL_UPDATES": "2",
    })
    result = subprocess.run(
        ["bash", "scripts/run_qwen_vjepa_ppo_alternating_wm.sh"],
        cwd=repository,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    command = result.stdout
    assert "--rollout-horizon 2" in command
    assert "--wm-refresh-horizon 2" in command
    assert "--wm-open-loop-horizon 2" in command
    assert "--wm-refresh-validation-guard" in command
    assert "scripts/train_mbrl.py" in command
    assert "scripts" + "_new" not in command
