#!/bin/bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

cd "$PROJECT_ROOT"

bash scripts/cloud_setup_ubuntu.sh
bash scripts/cloud_prepare_dirs.sh

echo
echo "Bootstrap completed."
echo "Next steps:"
echo "1. Download models: bash scripts/cloud_download_models.sh"
echo "2. Validate setup: bash scripts/cloud_validate_setup.sh"
echo "3. Run inference: bash scripts/run_inference.sh"
echo "4. Run visualization: bash scripts/run_visualize.sh"
