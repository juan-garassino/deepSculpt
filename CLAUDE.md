# DeepSculpt

## What this project is
A 3D generative art system that learns to create sculptures from scratch.
It uses two types of AI models:
- A **GAN** (two networks fighting each other — one creates shapes, one judges them)
- A **Diffusion model** (gradually removes noise to reveal a shape, like developing a photo)

The sculptures are stored as 3D grids of numbers (numpy arrays):
- Monochrome shapes: a cube of numbers, each cell says "empty" (0) or "solid" (1)
- Color shapes: same cube but each cell also has a color encoded as 4 numbers (RGBA)

## What I'm trying to do
Train the AI to generate believable 3D shapes it has never seen before,
then navigate the "space" of all possible shapes it has learned — finding
new sculptures, blending between two shapes, or asking "show me something rounder".

## The pipeline (in order)
1. **Generate training data** — create thousands of example 3D shapes using math
2. **Validate the data** — make sure the shapes look right before training
3. **Train the GAN or diffusion model** on those shapes
4. **Evaluate** — check if the AI is learning or getting stuck
5. **Explore** — navigate the space of shapes the AI has learned

## Stack
- Python, NumPy, PyTorch
- Shapes stored as `.npy` files
- Training runs on GPU (CUDA)

## Skills available to help
- `ds-datagen` — help writing shape generation code
- `ds-dataval` — help writing data checking and visualization code
- `ds-gan` — help with GAN architecture and training code
- `ds-diffusion` — help with diffusion model code
- `ds-latent` — help with latent space navigation code
- `ds-improve` — autonomous agent that runs improvement loops on the codebase

## ⚡ AGENT BEHAVIOR — READ THIS FIRST

When the user says anything like:
- "this isn't working" / "it's broken" / "fix this"
- "the training is getting worse"
- "the shapes all look the same"
- "make it better" / "help me improve this"
- "something is wrong"

**Do NOT just give advice. Immediately launch the `ds-improve` agent.**
Run it autonomously: measure diversity and validity of generated shapes,
diagnose what's failing, apply targeted fixes to the code, and report
what changed.

When the user says anything like:
- "explain how X works"
- "I don't understand X"
- "what is X"

Use the skills interactively to explain — don't launch the agent.

When the user says anything like:
- "write the training loop"
- "help me build the generator"
- "implement X"

Use the relevant skill interactively to write the code together.

## How to ask (no jargon needed)
- "The shapes all look the same" → launches ds-improve (mode collapse)
- "Training is crashing" → launches ds-improve
- "Something looks wrong with the data" → launches ds-improve
- "Blend between two sculptures" → interactive, uses ds-latent
- "Explain what a GAN is" → interactive explanation
- "Write the shape generator" → interactive, uses ds-datagen

## Current status
- **PyTorch v2 is the only path.** TensorFlow/Keras legacy is archived under `boilerplate/`.
- **CLI works.** `python -m deepsculpt.main {train-gan,train-diffusion,generate-data,sample-gan,sample-diffusion,preprocess,visualize,benchmark,evaluate,export}`.
- **Architectural data generator is the active mode** — columns + slabs + 3 orthogonal pipes (red/blue/yellow). See recent `git log` for the procedural-shape tuning history.
- **Test suite has pre-existing import bugs** (all `tests/unit/*.py` and `tests/integration/*` import the dead `deepSculpt` casing — package is `deepsculpt`). Don't trust `pytest` as a green light; smoke-test via the CLI instead.
- **Cloud training is live**: `runpod/` directory ships a CUDA 12.8 + Claude Code container that runs on RunPod and syncs checkpoints/results to GCS bucket `garassino-ml-artifacts`. See `runpod/README.md`.

## Cloud training (RunPod + GCS + Claude-in-the-loop)
The `runpod/` directory contains the full deploy. Four modes:
- `MODE=train` — pure training, no Claude in the loop.
- `MODE=research` — one-shot Claude reading `runpod/prompts/research.md`, runs experiments to `TIME_BUDGET`.
- `MODE=improve` — one-shot Claude reading `runpod/prompts/improve.md`, drives the `ds-improve` skill once.
- `MODE=self-improve` — **continuous** Claude loop; each iteration uses `runpod/prompts/self_improve.md` which references `runpod/prompts/autoresearch_program.md` (mirrored from the [020-autoresearch](../020-autoresearch/) reference repo). **Toggleable on/off** via a GCS-synced object (`gs://garassino-ml-artifacts/deepsculpt/control/self_improve.enabled`); `make toggle-on` / `make toggle-off` from the `runpod/` Makefile — no pod restart needed.

GCS layout: `gs://garassino-ml-artifacts/deepsculpt/{data,checkpoints/<run_id>,results/<run_id>,prompts-archive}/`. Crash-safe: periodic background `gsutil rsync` + final sync on `EXIT`. Image: `ghcr.io/juan-garassino/deepsculpt-runpod:latest`. See `runpod/README.md` and `docs/runpod.md`.

## File structure
```
deepsculpt/
├── main.py                       # CLI entry — 11 subcommands
├── config.yaml                   # central hyperparameters
├── core/
│   ├── data/{generation,loaders,sparse,transforms}/   # shape gen, dataloaders, encoding
│   ├── models/
│   │   ├── gan/{generator,discriminator}.py           # 5 gens, 8 discs incl. SelfAttention3D, LightDiscriminator
│   │   ├── diffusion/{unet,noise_scheduler,pipeline,pytorch_diffusion}.py
│   │   ├── base_models.py, model_factory.py
│   ├── training/{gan_trainer,diffusion_trainer,base_trainer,optimizers,schedulers}.py
│   ├── utils/{logger,pytorch_utils,monitoring,performance_optimizer}.py
│   ├── visualization/pytorch_visualization.py
│   └── workflow/{pytorch_workflow,pytorch_mlflow_tracking}.py
notebooks/                        # 18 Jupyter notebooks (Colab + local)
scripts/                          # colab_train.py, colab_train_diffusion.py, autoresearch_report.py
tests/                            # pytest suite (currently broken — see Current status)
runpod/                           # RunPod + GCS deploy (Dockerfile, entrypoint.sh, prompts/, scripts/)
autoresearch/                     # local Claude-Code-in-container research loop (predecessor to runpod/)
docs/                             # architecture, training, operations, inference, colab recipes
boilerplate/                      # archived TF/Keras v1 code
checkpoints/                      # local checkpoint dir (also under data/legacy_samples for samples)
```
