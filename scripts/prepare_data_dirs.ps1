Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot

$dirs = @(
    ".\data\input\image",
    ".\data\output\depth_image",
    ".\data\output\sam3_json",
    ".\data\output\correct",
    ".\data\temp",
    ".\data\visualize\image",
    ".\data\visualize\depth",
    ".\data\visualize\image_depth",
    ".\data\visualize\video"
)

foreach ($dir in $dirs) {
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
}

Write-Host "Data directories are ready."
