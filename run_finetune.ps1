#!/usr/bin/env pwsh
# Phase 2: fine-tune HAT-GAN Historical v2 from net_g_180000.pth using the
# new HistoricalDegradationDataset. ~3-4 days on RTX 4070 Super.
#
# Defaults:
#   - config: train_hat_gan_v2.yml
#   - logs/checkpoints saved under BasicSR/experiments/HAT_GAN_Historical_v2/
#
# Use -Resume to auto-resume the latest .state in v2's training_states dir.
# Use -Background to launch detached so closing the shell doesn't kill it.

param(
    [string]$Config = "train_hat_gan_v2.yml",
    [switch]$Resume,
    [switch]$Background,
    [switch]$DryRun
)

Set-Location $PSScriptRoot

# --- Sanity check: config + pretrain checkpoint + meta info ------------------
if (-not (Test-Path $Config)) { throw "Config not found: $Config" }

$pretrain = "BasicSR\experiments\HAT_GAN_Historical_v1\models\net_g_180000.pth"
if (-not (Test-Path $pretrain)) {
    throw "Pretrain checkpoint missing: $pretrain"
}
if (-not (Test-Path "datasets\HR_train")) { throw "datasets\HR_train missing" }
if (-not (Test-Path "datasets\meta_info_train.txt")) { throw "meta_info_train.txt missing" }

# --- Refuse to silently overwrite an existing v2 run -------------------------
$expDir = "BasicSR\experiments\HAT_GAN_Historical_v2"
if (Test-Path $expDir) {
    if ($Resume) {
        $latestState = Get-ChildItem "$expDir\training_states\*.state" -ErrorAction SilentlyContinue |
                       Sort-Object { [int]([IO.Path]::GetFileNameWithoutExtension($_.Name)) } -Descending |
                       Select-Object -First 1
        if (-not $latestState) {
            throw "No .state files in $expDir\training_states. Cannot resume."
        }
        Write-Host "==> Resuming from $($latestState.FullName)" -ForegroundColor Cyan
    } else {
        Write-Host "==> v2 experiment dir already exists: $expDir" -ForegroundColor Yellow
        Write-Host "    Re-run with -Resume to continue from the latest .state, or"
        Write-Host "    move/rename the dir to start a fresh v2 run." -ForegroundColor Yellow
        exit 1
    }
}

# --- Quick GPU smoke test ----------------------------------------------------
Write-Host "==> GPU check" -ForegroundColor Cyan
& python -c "import torch; assert torch.cuda.is_available(), 'no CUDA'; print('  device:', torch.cuda.get_device_name(0)); print('  free GB:', round(torch.cuda.mem_get_info()[0]/1e9, 2))"
if ($LASTEXITCODE -ne 0) { throw "GPU not available; aborting." }

# --- Smoke-test the dataset class once before launching ----------------------
Write-Host "==> Dataset smoke test (1 sample)" -ForegroundColor Cyan
$smokeTestCode = @'
import sys; sys.path.insert(0, "BasicSR")
from basicsr.data.historical_degradation_dataset import HistoricalDegradationDataset
ds = HistoricalDegradationDataset({
    "dataroot_gt": "datasets/HR_train",
    "meta_info": "datasets/meta_info_train.txt",
    "io_backend": {"type": "disk"},
    "gt_size": 192,
    "use_hflip": True, "use_rot": True,
})
s = ds[0]
print(f"  ok: {len(ds)} images, gt={tuple(s['gt'].shape)}, lq={tuple(s['lq'].shape)}")
'@
$smokeTestCode | python
if ($LASTEXITCODE -ne 0) { throw "dataset smoke test failed" }

# --- Build the command -------------------------------------------------------
if ($env:PYTHONPATH) {
    $env:PYTHONPATH = "BasicSR;" + $env:PYTHONPATH
} else {
    $env:PYTHONPATH = "BasicSR"
}
$trainArgs = @("BasicSR\basicsr\train.py", "-opt", $Config)
if ($Resume) { $trainArgs += "--auto_resume" }

Write-Host "`n==> Launching:" -ForegroundColor Cyan
Write-Host "    python $($trainArgs -join ' ')"
if ($DryRun) {
    Write-Host "(dry run; not launching)" -ForegroundColor Yellow
    exit 0
}

# --- Launch ------------------------------------------------------------------
if ($Background) {
    $logDir = "logs"
    if (-not (Test-Path $logDir)) { New-Item -ItemType Directory $logDir | Out-Null }
    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $stdout = "$logDir\v2_$stamp.out.log"
    $stderr = "$logDir\v2_$stamp.err.log"
    $proc = Start-Process -FilePath "python" -ArgumentList $trainArgs `
                          -RedirectStandardOutput $stdout `
                          -RedirectStandardError $stderr `
                          -PassThru -NoNewWindow
    Write-Host "  pid=$($proc.Id)  stdout=$stdout  stderr=$stderr" -ForegroundColor Green
    Write-Host "  tail with: Get-Content $stdout -Wait"
} else {
    & python @trainArgs
}
