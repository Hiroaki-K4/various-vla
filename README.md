# various-vla

VLA (Vision-Language-Action) model using DINOv2 + SigLIP as vision encoders and Llama-3.2-1B as the backbone.
Supports [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) simulation and [OpenX-Embodiment](https://huggingface.co/datasets/jxu124/OpenX-Embodiment) datasets.

## Setup

```bash
git clone --recurse-submodules <repo-url>
cd various-vla
# Python 3.12 headers (required to build evdev via robosuite -> pynput)
sudo add-apt-repository ppa:deadsnakes/ppa && sudo apt update && sudo apt install python3.12-dev
touch libero/libero/__init__.py
CMAKE_POLICY_VERSION_MINIMUM=3.5 UV_HTTP_TIMEOUT=300 uv sync
uv run huggingface-cli login  # required for Llama-3.2-1B
```

> Re-run `touch libero/libero/__init__.py` after `uv sync` if the file is deleted.

## Usage

| Script | Description |
|---|---|
| `try_libero.py` | Open MuJoCo viewer and run the LIBERO simulator |
| `try_llama.py` | Llama standalone sanity check |
| `train_libero.py` | Train VLA on LIBERO data (`libero_spatial`, ~56k steps) |
| `train.py` | Train VLA on OpenX-Embodiment (streamed from HuggingFace) |
| `tune.py` | Hyperparameter search with Optuna |

```bash
uv run python <script>.py
```

## Downloading LIBERO Datasets

`libero_spatial` is already included. To download additional task suites:

```bash
cd libero/benchmark_scripts
uv run --no-sync python download_libero_datasets.py --datasets <suite>
```

`--datasets` options: `libero_spatial`, `libero_object`, `libero_goal`, `libero_100`, `all`

Downloaded to `libero/libero/datasets/` by default.
