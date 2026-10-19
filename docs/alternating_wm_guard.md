# Alternating WM validation guard

The formal Qwen + V-JEPA `alternating_wm` runner treats every supervised world-model refresh as a transaction. It evaluates the shared model on the fixed held-out reward-validation split immediately before and after the candidate update. A candidate is committed only when all configured checks pass against both the pre-refresh state and the initial run baseline.

The checks cover total loss, one-step dynamics loss, open-loop dynamics loss, prior reward F1 when the panel contains positive examples, finite metric values, and belief effective rank. Relative loss thresholds also include a small absolute tolerance for near-zero losses.

On rejection, the implementation restores:

- every Phase-1 optimizer-owned world-model parameter;
- every registered world-model buffer, including dynamically resized SIGReg storage;
- the refresh optimizer state;
- the refresh scheduler state;
- the accepted refresh-step clock.

The checkpoint stores attempted, accepted, rejected, and consecutive-rejection counters. By default, three consecutive rejections stop the run after writing `latest.pt`; the rejected candidate has already been rolled back. Set `WM_REFRESH_CONTINUE_ON_REJECTION=1` to keep training after this limit.

The formal entry point enables the guard by default:

```bash
bash scripts/run_qwen_vjepa_ppo_alternating_wm.sh
```

The main environment controls are:

```text
WM_REFRESH_VALIDATION_GUARD=1
WM_REFRESH_VALIDATION_BATCHES=8
WM_REFRESH_MAX_TOTAL_LOSS_INCREASE=0.05
WM_REFRESH_MAX_DYNAMICS_LOSS_INCREASE=0.05
WM_REFRESH_MAX_OPEN_DYNAMICS_LOSS_INCREASE=0.10
WM_REFRESH_MAX_REWARD_F1_DROP=0.02
WM_REFRESH_MIN_EFFECTIVE_RANK_RATIO=0.90
WM_REFRESH_LOSS_ABSOLUTE_TOLERANCE=1e-4
WM_REFRESH_MAX_CONSECUTIVE_REJECTIONS=3
WM_REFRESH_CONTINUE_ON_REJECTION=0
```

Guard results are logged under `wm_refresh/guard/*`. Held-out values use `wm_refresh/validation_pre/*`, `wm_refresh/validation/*`, and `wm_refresh/validation_baseline/*`.

The formal runner follows the released Critic's exact-H2 target contract while retaining this repository's slotwise actor/critic and Qwen + V-JEPA model design. It does not import the separate H3 bootstrap, `latent_ordered_v`, or real-critic recalibration protocol from `MBRL0901`.
