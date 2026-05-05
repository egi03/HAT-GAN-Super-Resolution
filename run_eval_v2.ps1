#!/usr/bin/env pwsh
# Phase 6 wrapper: evaluate the latest v2 checkpoint against v1 baselines
# and write a publication-ready CSV.

param(
    [string]$V2Checkpoint,                    # default: latest under v2/models
    [string]$Out = "results\sota_synth.csv",
    [switch]$NoRef                            # also run on tests\historical
)

Set-Location $PSScriptRoot

if (-not $V2Checkpoint) {
    $latest = Get-ChildItem "BasicSR\experiments\HAT_GAN_Historical_v2\models\net_g_*.pth" -ErrorAction SilentlyContinue |
              Sort-Object { [int](([IO.Path]::GetFileNameWithoutExtension($_.Name)) -replace 'net_g_','') } -Descending |
              Select-Object -First 1
    if (-not $latest) {
        throw "No v2 checkpoints found. Pass -V2Checkpoint or run training first."
    }
    $V2Checkpoint = $latest.FullName
}

Write-Host "==> v2 checkpoint: $V2Checkpoint" -ForegroundColor Cyan

# Per-checkpoint baseline JSONs (cheap if results/ already exists; reruns are fine)
$tag = "v2_" + (([IO.Path]::GetFileNameWithoutExtension($V2Checkpoint)) -replace 'net_g_','')
& python scripts/run_baseline_metrics.py --checkpoint $V2Checkpoint --tag $tag
if ($LASTEXITCODE -ne 0) { throw "run_baseline_metrics.py failed" }

# Cross-method CSV
& python compare_sota.py --input tests\synthetic --output $Out --hat-v2 $V2Checkpoint
if ($LASTEXITCODE -ne 0) { throw "compare_sota.py failed" }

if ($NoRef) {
    $histOut = $Out -replace "\.csv$", "_hist.csv"
    & python compare_sota.py --input tests\historical --output $histOut --no-ref-only --hat-v2 $V2Checkpoint
}

Write-Host "`n==> CSVs ready" -ForegroundColor Cyan
Get-ChildItem results\*.csv -ErrorAction SilentlyContinue | ForEach-Object { Write-Host "  $($_.FullName)" }
