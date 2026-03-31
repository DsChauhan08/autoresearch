# autoresearch

![teaser](progress.png)

*One day, frontier AI research used to be done by meat computers in between eating, sleeping, having other fun, and synchronizing once in a while using sound wave interconnect in the ritual of "group meeting". That era is long gone. Research is now entirely the domain of autonomous swarms of AI agents running across compute cluster megastructures in the skies. The agents claim that we are now in the 10,205th generation of the code base, in any case no one could tell if that's right or wrong as the "code" is now a self-modifying binary that has grown beyond human comprehension. This repo is the story of how it all began. -@karpathy, March 2026*.

The idea: give an AI agent a small but real LLM training setup and let it experiment autonomously overnight. It modifies the code, trains for 5 minutes, checks if the result improved, keeps or discards, and repeats. You wake up in the morning to a log of experiments and (hopefully) a better model. The training code here is a simplified single-GPU implementation of [nanochat](https://github.com/karpathy/nanochat). The core idea is that you're not touching any of the Python files like you normally would as a researcher. Instead, you are programming the `program.md` Markdown files that provide context to the AI agents and set up your autonomous research org. The default `program.md` in this repo is intentionally kept as a bare bones baseline, though it's obvious how one would iterate on it over time to find the "research org code" that achieves the fastest research progress, how you'd add more agents to the mix, etc. A bit more context on this project is here in this [tweet](https://x.com/karpathy/status/2029701092347630069) and [this tweet](https://x.com/karpathy/status/2031135152349524125).

## How it works

The repo is deliberately kept small and only really has three files that matter:

- **`prepare.py`** — fixed constants, one-time data prep (downloads training data, trains a BPE tokenizer), and runtime utilities (dataloader, evaluation). Not modified.
- **`train.py`** — the single file the agent edits. Contains the full GPT model, optimizer (Muon + AdamW), and training loop. Everything is fair game: architecture, hyperparameters, optimizer, batch size, etc. **This file is edited and iterated on by the agent**.
- **`program.md`** — baseline instructions for one agent. Point your agent here and let it go. **This file is edited and iterated on by the human**.

By design, training runs for a **fixed 5-minute time budget** (wall clock, excluding startup/compilation), regardless of the details of your compute. The metric is **val_bpb** (validation bits per byte) — lower is better, and vocab-size-independent so architectural changes are fairly compared.

If you are new to neural networks, this ["Dummy's Guide"](https://x.com/hooeem/status/2030720614752039185) looks pretty good for a lot more context.

## Quick start

**Requirements:** Python 3.10+ and [uv](https://docs.astral.sh/uv/).
The default flow is CPU/iGPU-safe and does **not** require Vulkan or CUDA to be installed.

```bash

# 1. Install uv project manager (if you don't already have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install dependencies
uv sync

# 3. Download data and train tokenizer (one-time, ~2 min)
bash scripts/run_igpu.sh prepare

# 4. Manually run a single training experiment (~5 min)
bash scripts/run_igpu.sh train
```

If the above commands all work ok, your setup is working and you can go into autonomous research mode.

## Runtime paths

### Default path (CPU/iGPU-safe)
```bash
bash scripts/run_igpu.sh prepare
bash scripts/run_igpu.sh train
```

### Optional Vulkan path (explicit opt-in)
```bash
# optional: install Python build helpers for source-build workflows
uv sync --extra vulkan-build

bash scripts/run_vulkan.sh prepare
bash scripts/igpu_vulkan.sh setup
bash scripts/igpu_vulkan.sh check
bash scripts/igpu_vulkan.sh train
```

`scripts/run_vulkan.sh` supports `prepare` and otherwise delegates to `scripts/igpu_vulkan.sh`.

Vulkan training readiness is now stricter than just `torch.device("vulkan")`: startup probes include
tensor allocation + a tiny forward/backward training smoke test before selecting Vulkan.

PyTorch's official Vulkan workflow is build-from-source oriented. This repo mirrors that
workflow by expecting a Vulkan-enabled wheel built with `USE_VULKAN=1` and (by default)
`USE_VULKAN_SHADERC_RUNTIME=1`, `USE_VULKAN_WRAPPER=0`.

Important compatibility note: third-party forks such as
`ixu2486/pytorch_retryix_backend` currently publish Windows-only wheels (`win_amd64`)
and are not directly installable on this Linux setup. For Linux iGPU Vulkan training,
use the repo's supported path: build/install a Linux Vulkan-enabled PyTorch wheel via
`scripts/build_pytorch_vulkan.sh` or install one through `scripts/igpu_vulkan.sh setup`.

### Optional manual device override for training
```bash
bash scripts/run_igpu.sh train                         # default: AUTORESEARCH_DEVICE=auto
AUTORESEARCH_DEVICE=cpu bash scripts/run_igpu.sh train
AUTORESEARCH_DEVICE=cuda bash scripts/run_igpu.sh train
AUTORESEARCH_DEVICE=vulkan bash scripts/run_igpu.sh train
```

### Optional low-memory overrides (CPU/iGPU)
```bash
AUTORESEARCH_DEPTH=2 \
AUTORESEARCH_DEVICE_BATCH_SIZE=4 \
AUTORESEARCH_TOTAL_BATCH_SIZE=8192 \
bash scripts/run_igpu.sh train
```

### RAM budget control (new)
By default, `run_igpu.sh` sets `AUTORESEARCH_RAM_FRACTION=0.50`, so training shape auto-scales to about 50% of total system RAM.

```bash
# default behavior: ~50% RAM target
bash scripts/run_igpu.sh train

# explicit fraction (5%..95%)
AUTORESEARCH_RAM_FRACTION=0.60 bash scripts/run_igpu.sh train

# explicit absolute cap in MB (overrides fraction)
AUTORESEARCH_RAM_MB=6144 bash scripts/run_igpu.sh train

# optional full override (must be divisible by DEVICE_BATCH_SIZE * MAX_SEQ_LEN)
AUTORESEARCH_TOTAL_BATCH_SIZE=8192 AUTORESEARCH_DEVICE_BATCH_SIZE=4 bash scripts/run_igpu.sh train
```

## CPU-only validation and what "working" means

For CPU-only laptops, success means:

1. `prepare.py` completes (data + tokenizer created),
2. training starts and continues without backend/device crashes,
3. loss trends downward during the run,
4. the run reaches the time-budget stop condition and enters final evaluation.

On this machine, CPU mode was validated with:

```bash
AUTORESEARCH_DEVICE=cpu \
AUTORESEARCH_RAM_FRACTION=0.50 \
AUTORESEARCH_USE_MUON=0 \
AUTORESEARCH_AMP=0 \
AUTORESEARCH_COMPILE=0 \
AUTORESEARCH_TOKENIZER_THREADS=1 \
uv run train.py
```

Observed runtime diagnostics:

- `Selected: cpu (explicit cpu request)`
- `Training shape: depth=3, device_batch_size=6, total_batch_size=24576`
- `RAM budget: 6.7 GiB (AUTORESEARCH_RAM_FRACTION=0.500)`
- training progressed through the full 5-minute budget (`remaining` reached `0s`)
- train loss decreased steadily (example: `9.01 -> 7.31` during the same run)

Note: CPU runs are functionally valid but slower than CUDA. The research loop is the same; throughput is lower.

## Autopilot on CPU/iGPU (detailed runbook)

`autoresearch` autopilot is an agent loop (edit `train.py` -> run -> evaluate -> keep/discard), not just a single training command.

### 1) One-time setup

```bash
uv sync
bash scripts/run_igpu.sh prepare
```

### 2) Stable CPU runtime defaults

Use these for unattended CPU runs:

```bash
export AUTORESEARCH_DEVICE=cpu
export AUTORESEARCH_RAM_FRACTION=0.50
export AUTORESEARCH_USE_MUON=0
export AUTORESEARCH_AMP=0
export AUTORESEARCH_COMPILE=0
export AUTORESEARCH_TOKENIZER_THREADS=1
```

### 3) Start autonomous experimentation

Point your coding agent at `program.md` and let it run continuously. The default loop in `program.md` already does:

- modify `train.py`
- run `uv run train.py`
- read `val_bpb`
- keep/discard changes
- repeat indefinitely

### 4) Run in a persistent terminal session

Use `tmux`/`screen` so the loop survives terminal disconnects:

```bash
tmux new -s autoresearch
# start your agent in this repo, then detach with Ctrl-b d
```

### 5) Monitor health

During long runs, periodically check:

- no repeated Python tracebacks in logs,
- loss remains finite (no NaN/INF),
- `val_bpb` is still being produced,
- system memory pressure is acceptable (reduce RAM fraction if needed).

### 6) If you hit memory pressure

Lower one or more of:

- `AUTORESEARCH_RAM_FRACTION` (e.g. `0.35`)
- `AUTORESEARCH_DEPTH`
- `AUTORESEARCH_DEVICE_BATCH_SIZE`
- `AUTORESEARCH_TOTAL_BATCH_SIZE` (must be divisible by `DEVICE_BATCH_SIZE * MAX_SEQ_LEN`)

## Running the agent

Simply spin up your Claude/Codex or whatever you want in this repo (and disable all permissions), then you can prompt something like:

```
Hi have a look at program.md and let's kick off a new experiment! let's do the setup first.
```

The `program.md` file is essentially a super lightweight "skill".

## Project structure

```
prepare.py      — constants, data prep + runtime utilities (do not modify)
train.py        — model, optimizer, training loop (agent modifies this)
program.md      — agent instructions
pyproject.toml  — dependencies
```

## Design choices

- **Single file to modify.** The agent only touches `train.py`. This keeps the scope manageable and diffs reviewable.
- **Fixed time budget.** Training always runs for exactly 5 minutes, regardless of your specific platform. This means you can expect approx 12 experiments/hour and approx 100 experiments while you sleep. There are two upsides of this design decision. First, this makes experiments directly comparable regardless of what the agent changes (model size, batch size, architecture, etc). Second, this means that autoresearch will find the most optimal model for your platform in that time budget. The downside is that your runs (and results) become not comparable to other people running on other compute platforms.
- **Self-contained.** No external dependencies beyond PyTorch and a few small packages. No distributed training, no complex configs. One GPU, one file, one metric.

## Platform support

The project supports NVIDIA CUDA, Intel/AMD Vulkan, and CPU backends. By default, the system auto-detects your device and falls back gracefully.

**For smaller compute platforms** (Macbooks, laptops, etc.), here are tuning recommendations:

1. **Dataset:** Use a dataset with lower entropy, e.g. [TinyStories](https://huggingface.co/datasets/karpathy/tinystories-gpt4-clean). The narrower scope allows smaller models to show reasonable results.
2. **Vocab size:** Decrease from 8192 to 4096, 2048, 1024, or even 256 (byte-level).
3. **Sequence length:** In `prepare.py`, lower `MAX_SEQ_LEN` (e.g., down to 256). Adjust `DEVICE_BATCH_SIZE` in `train.py` to compensate.
4. **Evaluation budget:** In `prepare.py`, decrease `EVAL_TOKENS` to reduce validation data.
5. **Model depth:** In `train.py`, lower `DEPTH` (default 8) to 4 or lower. Many hyperparameters depend on this.
6. **Attention pattern:** Use `WINDOW_PATTERN = "L"` instead of "SSSL" for efficiency on smaller devices.
7. **Batch size:** Lower `TOTAL_BATCH_SIZE` while keeping it a power of 2 (e.g., `2**14` ≈ 16K).

## Notable forks

- [miolini/autoresearch-macos](https://github.com/miolini/autoresearch-macos) (MacOS)
- [trevin-creator/autoresearch-mlx](https://github.com/trevin-creator/autoresearch-mlx) (MacOS)
- [jsegov/autoresearch-win-rtx](https://github.com/jsegov/autoresearch-win-rtx) (Windows)
- [andyluo7/autoresearch](https://github.com/andyluo7/autoresearch) (AMD)

## License

MIT
