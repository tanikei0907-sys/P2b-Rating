param(
    [string]$Date
)

Set-Location $PSScriptRoot
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$outDir = Join-Path $PSScriptRoot "rating_jra\p2b_forward"
New-Item -ItemType Directory -Force -Path $outDir | Out-Null
$logPath = Join-Path $outDir "predict_snapshot_last.log"
$arguments = @("predict_snapshot", "--config", "rating_jra/forward_config.yaml")
if ($Date) {
    $arguments += @("--date", $Date)
}
& python "rating_jra/rating_update_for_features.py" @arguments *>&1 | Tee-Object -FilePath $logPath
$rc = $LASTEXITCODE
if ($rc -ne 0) {
    Write-Error "Snapshot generation failed with exit code $rc. See $logPath"
    exit $rc
}
$syncLogPath = Join-Path $outDir "github_sync_last.log"
& python "rating_jra/github_sync.py" *>&1 | Tee-Object -FilePath $syncLogPath
$syncRc = $LASTEXITCODE
if ($syncRc -ne 0) {
    Write-Error "Local snapshot generation succeeded, but GitHub synchronization failed. See $syncLogPath"
    exit $syncRc
}
