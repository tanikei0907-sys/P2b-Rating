@echo off
setlocal
cd /d "%~dp0"
if not exist "rating_jra\p2b_forward" mkdir "rating_jra\p2b_forward"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
python "rating_jra\rating_update_for_features.py" update --config "rating_jra\forward_config.yaml" %* > "rating_jra\p2b_forward\update_last.log" 2>&1
set "RC=%ERRORLEVEL%"
type "rating_jra\p2b_forward\update_last.log"
if not "%RC%"=="0" exit /b %RC%
python "rating_jra\github_sync.py" > "rating_jra\p2b_forward\github_sync_last.log" 2>&1
set "SYNC_RC=%ERRORLEVEL%"
type "rating_jra\p2b_forward\github_sync_last.log"
exit /b %SYNC_RC%
