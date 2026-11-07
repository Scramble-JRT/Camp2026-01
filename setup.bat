@echo off
cd /d %~dp0
where py >nul 2>nul && (set PY=py) || (set PY=python)
if not exist .venv (
  echo [1/3] Creating virtual environment .venv ...
  %PY% -m venv .venv || goto :error
)
echo [2/3] Installing packages ...
.venv\Scripts\python -m pip install --upgrade pip || goto :error
.venv\Scripts\python -m pip install -r requirements.txt || goto :error
if not exist known_faces mkdir known_faces
echo [3/3] Checking OpenCV ...
.venv\Scripts\python -c "import cv2; print('OpenCV', cv2.__version__, 'OK')" || goto :error
echo.
echo Setup complete. Register a face with the built-in webcam:
echo   .venv\Scripts\python face_recognition_sample.py register taro
echo Then start recognition:
echo   .venv\Scripts\python face_recognition_sample.py
pause
exit /b 0
:error
echo Setup failed. Make sure Python 3.9+ is installed (https://www.python.org/).
pause
exit /b 1
