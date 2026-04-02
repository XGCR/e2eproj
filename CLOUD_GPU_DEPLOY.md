# Compass Cloud GPU Deployment

This guide deploys Compass onto a single Ubuntu GPU cloud instance without changing the project architecture.

Recommended target:

- Ubuntu 22.04 or 24.04
- 1 NVIDIA GPU
- 16 GB to 24 GB+ VRAM preferred
- 8 vCPU+
- 32 GB RAM+
- 200 GB disk+

## 1. Clone or upload the project

Keep the project on the instance's local Linux filesystem.

Recommended path:

```bash
/workspace/compass
```

Do not place the project inside a network mount or object-storage-backed path.

## 2. Prepare the machine

From the project root:

```bash
cd /workspace/compass
bash scripts/cloud_setup_ubuntu.sh
```

What it does:

- installs system packages such as `git`, `git-lfs`, `wget`, and `build-essential`
- installs Miniconda under `$HOME/miniconda3` if missing
- creates a `compass` Python 3.11 environment by default
- installs PyTorch CUDA 12.1 wheels
- installs Python dependencies
- installs model repo dependencies
- ensures `numpy<2` for SAM3 compatibility

## 3. Prepare runtime folders

```bash
cd /workspace/compass
bash scripts/cloud_prepare_dirs.sh
```

## 4. Download model weights

The project expects these local paths:

- `models/Depth-Anything-V2/metric_depth/checkpoints/*.pth`
- `models/sam3/checkpoints/sam3.pt`
- `models/Qwen3-VL/checkpoints/Qwen3-VL-8B-Instruct`

Use:

```bash
cd /workspace/compass
bash scripts/cloud_download_models.sh
```

If a provider blocks Hugging Face or ModelScope access, download those weights manually and place them into the same target paths.

## 5. Validate the setup

```bash
cd /workspace/compass
bash scripts/cloud_validate_setup.sh
```

Expected checks:

- `nvidia-smi` works
- `torch.cuda.is_available()` is `True`
- `numpy` is `1.26.x`
- `sam3` imports
- `PipelineManager` imports
- model paths exist

## 6. Upload input images

Put `.jpg` files into:

```text
data/input/image
```

Examples:

```bash
scp -r ./local_images/*.jpg user@your-server:/workspace/compass/data/input/image/
```

Or with `rsync`:

```bash
rsync -av ./local_images/ user@your-server:/workspace/compass/data/input/image/
```

## 7. Run inference

```bash
cd /workspace/compass
bash scripts/run_inference.sh
```

## 8. Run visualization

```bash
cd /workspace/compass
bash scripts/run_visualize.sh
```

## 9. Download results back to local

Inference outputs:

```text
data/output
```

Visualization outputs:

```text
data/visualize
```

Examples:

```bash
rsync -av user@your-server:/workspace/compass/data/output/ ./cloud_output/
rsync -av user@your-server:/workspace/compass/data/visualize/ ./cloud_visualize/
```

## 10. Operational notes

- Stop the instance when not in use to save cost.
- Keep models on disk between runs if the cloud provider preserves the volume.
- If the instance has one GPU, do not set `CUDA_VISIBLE_DEVICES` unless you have a specific reason.
- If the instance has multiple GPUs, set `CUDA_VISIBLE_DEVICES` before running the scripts.

Example:

```bash
export CUDA_VISIBLE_DEVICES=0
bash scripts/run_inference.sh
```

## 11. First troubleshooting checks

Check the GPU:

```bash
nvidia-smi
```

Check Python and CUDA:

```bash
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate compass
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.device_count())"
```

Check the pipeline import:

```bash
python -c "from src.pipeline.pipeline_manager import PipelineManager; print('ok')"
```
