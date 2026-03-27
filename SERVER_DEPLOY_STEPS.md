# Server Deployment Steps

This file is for manual deployment when Codex cannot directly access the target server.

Target server:

- Host: `158.132.21.97`
- Port: `22`
- User: `mamaoheng`

Do not store the password in scripts. Enter it manually when prompted.

## 1. Upload the project package from Windows

Use PowerShell on your local machine:

```powershell
scp -P 22 D:\e2eProj\compass-cloud-upload.zip mamaoheng@158.132.21.97:~/
```

If `scp` is blocked or unstable, use WinSCP and upload `D:\e2eProj\compass-cloud-upload.zip` to the server home directory.

## 2. Log into the server

```powershell
ssh -p 22 mamaoheng@158.132.21.97
```

## 3. Unpack the project on the server

After login:

```bash
mkdir -p ~/deploy
cd ~/deploy
unzip -o ~/compass-cloud-upload.zip
cd compass
```

If `unzip` is missing:

```bash
sudo apt-get update
sudo apt-get install -y unzip
```

## 4. Bootstrap the environment

From `~/deploy/compass`:

```bash
bash scripts/cloud_bootstrap.sh
```

This will:

- install system packages
- install Miniconda
- create the `compass` conda environment
- install Python dependencies
- prepare local runtime directories

## 5. Download model weights

Still in `~/deploy/compass`:

```bash
bash scripts/cloud_download_models.sh
```

If network access to Hugging Face or ModelScope is restricted on campus, download those model files by other means and place them into:

- `models/Depth-Anything-V2/metric_depth/checkpoints/`
- `models/sam3/checkpoints/`
- `models/Qwen3-VL/checkpoints/Qwen3-VL-8B-Instruct/`

## 6. Validate the setup

```bash
bash scripts/cloud_validate_setup.sh
```

You should confirm:

- `nvidia-smi` works
- CUDA is available to PyTorch
- `numpy` is `1.26.x`
- `sam3 import ok`
- `pipeline import ok`

## 7. Upload input images

Back on your local Windows machine:

```powershell
scp -P 22 D:\path\to\your\images\*.jpg mamaoheng@158.132.21.97:~/deploy/compass/data/input/image/
```

## 8. Run inference and visualization

On the server:

```bash
cd ~/deploy/compass
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate compass
bash scripts/run_inference.sh
bash scripts/run_visualize.sh
```

## 9. Download results back to local

On your local Windows machine:

```powershell
scp -P 22 -r mamaoheng@158.132.21.97:~/deploy/compass/data/output D:\e2eProj\server-results\
scp -P 22 -r mamaoheng@158.132.21.97:~/deploy/compass/data/visualize D:\e2eProj\server-results\
```

## 10. First server-side checks if something fails

```bash
nvidia-smi
python3 --version
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate compass
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.device_count())"
python -c "from src.pipeline.pipeline_manager import PipelineManager; print('ok')"
```
