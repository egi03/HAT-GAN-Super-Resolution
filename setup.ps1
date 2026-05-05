#!/usr/bin/env pwsh
# One-time setup: install Python deps needed for the new scripts.
# Safe to re-run; pip is idempotent.

param(
    [switch]$Optional, # Also install CodeFormer + Contextual-Loss (Phase 4/5)
    [switch]$Colorize  # Also install modelscope for DDColor colorization
)

Set-Location $PSScriptRoot

function Test-PyImport($module) {
    $code = "import importlib,sys; sys.exit(0 if importlib.util.find_spec('$module') else 1)"
    & python -c $code
    return ($LASTEXITCODE -eq 0)
}

Write-Host "==> Verifying Python + torch" -ForegroundColor Cyan
& python -c "import torch; print(f'  torch={torch.__version__}  cuda={torch.cuda.is_available()}  device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"
if ($LASTEXITCODE -ne 0) {
    throw "python or torch missing. Install Python 3.10+ and a CUDA build of torch first."
}

$core = @("lpips", "pyiqa", "DISTS-pytorch")
foreach ($pkg in $core) {
    $modName = $pkg.ToLower().Replace("-", "_")
    if (Test-PyImport $modName) {
        Write-Host "  $pkg already installed" -ForegroundColor Green
    } else {
        Write-Host "  installing $pkg..." -ForegroundColor Yellow
        & python -m pip install $pkg
        if ($LASTEXITCODE -ne 0) { throw "pip install $pkg failed" }
    }
}

if ($Optional) {
    Write-Host "==> Optional extras (Phase 4/5)" -ForegroundColor Cyan
    if (-not (Test-PyImport "codeformer")) {
        & python -m pip install codeformer-pip
    }
    if (-not (Test-PyImport "contextual_loss")) {
        & python -m pip install "git+https://github.com/S-aiueo32/contextual_loss_pytorch"
    }
}

if ($Colorize) {
    Write-Host "==> Colorization extras (DDColor via modelscope)" -ForegroundColor Cyan
    if (Test-PyImport "modelscope") {
        Write-Host "  modelscope already installed" -ForegroundColor Green
    } else {
        Write-Host "  installing modelscope (this pulls several deps; may take a few minutes)" -ForegroundColor Yellow
        & python -m pip install modelscope
        if ($LASTEXITCODE -ne 0) { throw "pip install modelscope failed" }
    }
    Write-Host "  Note: first colorize call downloads ~600 MB of weights to ~\.cache\modelscope" -ForegroundColor Yellow
}

Write-Host "`n==> Verifying project files" -ForegroundColor Cyan
$req = @(
    "BasicSR\basicsr\train.py",
    "BasicSR\basicsr\data\historical_degradation_dataset.py",
    "BasicSR\experiments\HAT_GAN_Historical_v1\models\net_g_180000.pth",
    "datasets\HR_train",
    "datasets\meta_info_train.txt",
    "train_hat_gan_v2.yml"
)
$missing = @()
foreach ($p in $req) {
    if (Test-Path $p) {
        Write-Host "  ok   $p" -ForegroundColor Green
    } else {
        Write-Host "  MISS $p" -ForegroundColor Red
        $missing += $p
    }
}
if ($missing.Count -gt 0) {
    throw "Missing files: $($missing -join ', ')"
}

Write-Host "`n==> Setup complete." -ForegroundColor Cyan
Write-Host "Next: .\run_baseline.ps1   # builds test set + baseline metrics on 180k & 240k"
Write-Host "Then: .\run_finetune.ps1   # launches v2 fine-tune (~3-4 days on 4070 Super)"
