@echo off
setlocal
cd /d "%~dp0"
if not exist "rating_jra\p7_forward" mkdir "rating_jra\p7_forward"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "PYTHONDONTWRITEBYTECODE=1"
python "rating_jra\p7_update_for_features.py" predict_snapshot --config "rating_jra\p7_forward\forward_config.yaml" %* > "rating_jra\p7_forward\predict_snapshot_last.log" 2>&1
set "RC=%ERRORLEVEL%"
type "rating_jra\p7_forward\predict_snapshot_last.log"
if not "%RC%"=="0" exit /b %RC%
python "rating_jra\github_sync.py" > "rating_jra\p7_forward\github_sync_last.log" 2>&1
set "SYNC_RC=%ERRORLEVEL%"
type "rating_jra\p7_forward\github_sync_last.log"
exit /b %SYNC_RC%
