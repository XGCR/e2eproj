Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot

$env:PYTHONPATH = $projectRoot

python .\src\main.py --config .\configs\default_config.yaml
