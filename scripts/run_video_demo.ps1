Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot

$env:PYTHONPATH = $projectRoot

python .\src\video_demo.py --video ..\outdoor2.mp4 @args
