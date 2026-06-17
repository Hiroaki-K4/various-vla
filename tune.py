"""Optuna hyperparameter search for learning rate and LoRA r / alpha."""

import optuna
import torch

from train import train

# ── fixed settings ────────────────────────────────────────────────────────────
LLM_MODEL_NAME = "meta-llama/Llama-3.2-1B"
BATCH_SIZE = 2
# Use fewer epochs per trial to keep tuning fast.
NUM_EPOCHS = 1
PATIENCE = 2
# Evaluate frequently so early stopping kicks in quickly for bad configs.
EVAL_INTERVAL = 200
NUM_WORKERS = 4
GRADIENT_ACCUMULATION_STEPS = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

N_TRIALS = 20
STUDY_NAME = "vla_lora_lr_search"
STORAGE = f"sqlite:///{STUDY_NAME}.db"  # persists results across restarts


def objective(trial: optuna.Trial) -> float:
    # ── search space ──────────────────────────────────────────────────────────
    lr_rate = trial.suggest_float("lr_rate", 1e-6, 1e-3, log=True)
    lora_r = trial.suggest_categorical("lora_r", [4, 8, 16, 32, 64])
    # alpha is suggested as a multiplier of r so the ratio stays meaningful.
    alpha_multiplier = trial.suggest_categorical("alpha_multiplier", [0.5, 1, 2])
    lora_alpha = int(lora_r * alpha_multiplier)
    lora_dropout = trial.suggest_float("lora_dropout", 0.0, 0.2)

    save_model_path = f"tune_trial_{trial.number}"

    print(
        f"\n[Trial {trial.number}] lr={lr_rate:.2e}  "
        f"lora_r={lora_r}  lora_alpha={lora_alpha}  lora_dropout={lora_dropout:.3f}"
    )

    val_loss = train(
        llm_model_name=LLM_MODEL_NAME,
        batch_size=BATCH_SIZE,
        num_epochs=NUM_EPOCHS,
        lr_rate=lr_rate,
        patience=PATIENCE,
        eval_interval=EVAL_INTERVAL,
        num_workers=NUM_WORKERS,
        device=DEVICE,
        save_model_path=save_model_path,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
    )

    return val_loss


if __name__ == "__main__":
    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(
        direction="minimize",
        study_name=STUDY_NAME,
        storage=STORAGE,
        load_if_exists=True,
        sampler=sampler,
    )

    study.optimize(objective, n_trials=N_TRIALS, gc_after_trial=True)

    print("\n=== Best trial ===")
    best = study.best_trial
    print(f"  Val loss : {best.value:.4f}")
    print(f"  Params   : {best.params}")
    print(
        f"  lora_alpha (derived) : {int(best.params['lora_r'] * best.params['alpha_multiplier'])}"
    )
