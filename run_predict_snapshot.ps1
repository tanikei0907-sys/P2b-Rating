param(
    [string]$Date
)

Set-Location $PSScriptRoot
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONDONTWRITEBYTECODE = "1"
$outDir = Join-Path $PSScriptRoot "rating_jra\p7_forward"
New-Item -ItemType Directory -Force -Path $outDir | Out-Null
$logPath = Join-Path $outDir "predict_snapshot_last.log"
$arguments = @("predict_snapshot", "--config", "rating_jra/p7_forward/forward_config.yaml")
if ($Date) {
    $arguments += @("--date", $Date)
}
& python "rating_jra/p7_update_for_features.py" @arguments *>&1 | Tee-Object -FilePath $logPath
$rc = $LASTEXITCODE
if ($rc -ne 0) {
    Write-Error "P7 snapshot generation failed with exit code $rc. See $logPath"
    exit $rc
}
$syncLogPath = Join-Path $outDir "github_sync_last.log"
& python "rating_jra/github_sync.py" *>&1 | Tee-Object -FilePath $syncLogPath
$syncRc = $LASTEXITCODE
if ($syncRc -ne 0) {
    Write-Error "Local P7 snapshot generation succeeded, but GitHub synchronization failed. See $syncLogPath"
    exit $syncRc
}
