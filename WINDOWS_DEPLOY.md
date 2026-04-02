# Compass Windows Deployment Guide

This project is a local image-processing pipeline, not a web service.
It runs three heavyweight model components in sequence:

1. Depth Anything V2
2. SAM3
3. Qwen3-VL

The main entrypoint is `src/main.py`, and the visualization entrypoint is `src/visualize.py`.

## 1. What this project expects

- Windows 10/11
- Miniconda or Anaconda
- Git
- Git LFS
- Python 3.11
- Preferably an NVIDIA GPU with CUDA support

Notes:

- `configs/default_config.yaml` defaults to `device: "cuda"`.
- The Qwen3-VL config points to `models/Qwen3-VL/checkpoints/Qwen3-VL-8B-Instruct`, which is very large.
- CPU-only deployment is theoretically possible after switching the config to `cpu`, but performance will likely be too slow for normal use.

## 2. Recommended folder layout

After setup, the project should look like this:

```text
compass/
  configs/
  data/
    input/
      image/
    output/
      correct/
      depth_image/
      sam3_json/
    temp/
    visualize/
  models/
    Depth-Anything-V2/
    sam3/
    Qwen3-VL/
  scripts/
  src/
```

## 3. Create the conda environment

Open PowerShell in the `compass` folder and run:

```powershell
conda create -n compass python=3.11 -y
conda activate compass
python -m pip install --upgrade pip
```

## 4. Install PyTorch first

Install the PyTorch build that matches your machine.

Example for CUDA 12.1:

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Example for CPU-only:

```powershell
pip install torch torchvision torchaudio
```

## 5. Clone model repositories

From `compass/models`:

```powershell
cd .\models
git clone https://github.com/DepthAnything/Depth-Anything-V2
git clone https://github.com/facebookresearch/sam3.git
git clone https://github.com/QwenLM/Qwen3-VL.git
```

## 6. Install Python dependencies

Still inside the `compass` environment:

```powershell
pip install opencv-python pillow numpy pyyaml tifffile einops decord pycocotools transformers accelerate sentencepiece
pip install modelscope
```

Then install model-specific dependencies:

```powershell
pip install -r .\models\Qwen3-VL\requirements_web_demo.txt
pip install -e .\models\sam3
pip install -r .\models\Depth-Anything-V2\metric_depth\requirements.txt
```

If `pycocotools` fails on Windows, try:

```powershell
pip install pycocotools-windows
```

## 7. Download model weights

### 7.1 Depth Anything V2 checkpoints

Create the folder:

```powershell
New-Item -ItemType Directory -Force .\models\Depth-Anything-V2\metric_depth\checkpoints
```

Download these files into that folder:

- `depth_anything_v2_metric_hypersim_vits.pth`
- `depth_anything_v2_metric_vkitti_vits.pth`
- `depth_anything_v2_metric_hypersim_vitb.pth`
- `depth_anything_v2_metric_vkitti_vitb.pth`
- `depth_anything_v2_metric_hypersim_vitl.pth`
- `depth_anything_v2_metric_vkitti_vitl.pth`

Their expected target path is controlled by [configs/default_config.yaml](d:/e2eProj/compass/configs/default_config.yaml).

### 7.2 SAM3 checkpoint

Create:

```powershell
New-Item -ItemType Directory -Force .\models\sam3\checkpoints
```

Then download the SAM3 weights into:

```text
models/sam3/checkpoints/sam3.pt
```

The original Linux script used:

```powershell
modelscope download --model facebook/sam3 --local_dir .\models\sam3\checkpoints
```

### 7.3 Qwen3-VL checkpoint

Create:

```powershell
New-Item -ItemType Directory -Force .\models\Qwen3-VL\checkpoints
```

Then clone or download the target model directory so that this path exists:

```text
models/Qwen3-VL/checkpoints/Qwen3-VL-8B-Instruct
```

The original project also mentioned these optional variants:

- `Qwen3-VL-4B-Instruct`
- `Qwen3-VL-8B-Instruct`
- `Qwen3-VL-8B-Instruct-FP8`

## 8. Prepare input and output folders

The current config expects input images in:

```text
data/input/image
```

Create folders:

```powershell
New-Item -ItemType Directory -Force .\data\input\image
New-Item -ItemType Directory -Force .\data\output\depth_image
New-Item -ItemType Directory -Force .\data\output\sam3_json
New-Item -ItemType Directory -Force .\data\output\correct
New-Item -ItemType Directory -Force .\data\temp
New-Item -ItemType Directory -Force .\data\visualize\image
New-Item -ItemType Directory -Force .\data\visualize\depth
New-Item -ItemType Directory -Force .\data\visualize\image_depth
New-Item -ItemType Directory -Force .\data\visualize\video
```

Put your `.jpg` input images into `data/input/image`.

## 9. Run inference

Use the provided PowerShell wrapper:

```powershell
.\scripts\run_inference.ps1
```

Or run directly:

```powershell
$env:PYTHONPATH = (Get-Location).Path
python .\src\main.py --config .\configs\default_config.yaml
```

## 10. Run visualization

```powershell
.\scripts\run_visualize.ps1
```

Or run directly:

```powershell
$env:PYTHONPATH = (Get-Location).Path
python .\src\visualize.py --input .\data\output\correct
```

## 11. Important Windows caveats

- The repository was originally authored for Linux shell scripts.
- Model installation and checkpoint download are the most fragile part of the setup.
- `CUDA_VISIBLE_DEVICES=4,5,6,7` in the original scripts does not make sense on a normal Windows laptop and is intentionally not used in the PowerShell wrappers.
- If your laptop has only one GPU, leaving device selection to PyTorch is the safer default.
- If you do not have a capable NVIDIA GPU, switch [configs/default_config.yaml](d:/e2eProj/compass/configs/default_config.yaml) from `cuda` to `cpu` before testing, but expect very slow execution.

## 12. First troubleshooting checks

Verify Python can see the project:

```powershell
$env:PYTHONPATH = (Get-Location).Path
python -c "from src.pipeline.pipeline_manager import PipelineManager; print('import ok')"
```

Verify CUDA:

```powershell
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.device_count())"
```

If inference fails immediately, the usual causes are:

1. Missing model repositories under `models/`
2. Missing checkpoint files
3. Wrong `device` value in the config
4. Missing dependency from one of the model repos
