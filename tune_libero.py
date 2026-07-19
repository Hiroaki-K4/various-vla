"""Optuna hyperparameter search for LIBERO training."""

import optuna
import torch

from train_libero import train

# ── fixed settings ────────────────────────────────────────────────────────────
DATASET_DIR = "libero/libero/datasets/libero_spatial"
LLM_MODEL_NAME = "meta-llama/Llama-3.2-3B"
BATCH_SIZE = 2
NUM_EPOCHS = 5
PATIENCE = 5
# Keep eval_interval short so early stopping kicks in quickly for bad configs.
EVAL_INTERVAL = 1000
NUM_WORKERS = 4
GRADIENT_ACCUMULATION_STEPS = 2
VAL_DEMOS = 5
CAMERAS = ("agentview_rgb", "eye_in_hand_rgb")  # Two-view input
SEED = 42
IMAGE_AUG = False
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

N_TRIALS = 20
STUDY_NAME = "vla_libero_search"
STORAGE = f"sqlite:///{STUDY_NAME}.db"


def objective(trial: optuna.Trial) -> float:
    lr_rate = trial.suggest_float("lr_rate", 1e-6, 1e-3, log=True)
    lora_r = trial.suggest_categorical("lora_r", [8, 16, 32, 64])
    alpha_multiplier = trial.suggest_categorical("alpha_multiplier", [0.5, 1, 2])
    lora_alpha = int(lora_r * alpha_multiplier)
    lora_dropout = trial.suggest_float("lora_dropout", 0.0, 0.2)
    warmup_ratio = trial.suggest_float("warmup_ratio", 0.01, 0.15)

    print(
        f"\n[Trial {trial.number}] lr={lr_rate:.2e}  "
        f"lora_r={lora_r}  lora_alpha={lora_alpha}  "
        f"lora_dropout={lora_dropout:.3f}  warmup_ratio={warmup_ratio:.3f}"
    )

    val_loss = train(
        dataset_dir=DATASET_DIR,
        llm_model_name=LLM_MODEL_NAME,
        batch_size=BATCH_SIZE,
        num_epochs=NUM_EPOCHS,
        lr_rate=lr_rate,
        patience=PATIENCE,
        eval_interval=EVAL_INTERVAL,
        num_workers=NUM_WORKERS,
        device=DEVICE,
        save_model_path=None,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        val_demos=VAL_DEMOS,
        cameras=CAMERAS,
        warmup_ratio=warmup_ratio,
        seed=SEED,
        image_aug=IMAGE_AUG,
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
        f"  lora_alpha (derived) : "
        f"{int(best.params['lora_r'] * best.params['alpha_multiplier'])}"
    )
