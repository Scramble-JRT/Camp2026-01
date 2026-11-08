@echo off
cd /d %~dp0
if exist .venv\Scripts\python.exe (
  set PY=.venv\Scripts\python
) else (
  echo .venv not found. Run setup.bat first.
  pause
  exit /b 1
)
%PY% face_recognition_sample.py
if errorlevel 1 pause
