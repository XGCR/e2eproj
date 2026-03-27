#!/bin/bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="${PYTHONPATH:-}:$PROJECT_ROOT"

cd "$PROJECT_ROOT"
python src/main.py --config "configs/default_config.yaml"
