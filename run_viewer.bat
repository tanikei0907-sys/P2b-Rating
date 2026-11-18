@echo off
setlocal
cd /d "%~dp0rating_jra"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
start "" "http://127.0.0.1:8501"
python -m streamlit run rating_viewer.py --server.headless true --server.port 8501 --server.address 127.0.0.1 > viewer_server.log 2>&1
set "RC=%ERRORLEVEL%"
type "viewer_server.log"
exit /b %RC%
