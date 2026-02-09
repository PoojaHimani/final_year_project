@echo off
echo ========================================
echo Installing Required Dependencies
echo ========================================
echo.

REM Try pip first
where pip >nul 2>&1
if %errorlevel% == 0 (
    echo [1/4] Using pip...
    pip install opencv-python mediapipe numpy pyttsx3
    if %errorlevel% == 0 goto :success
)

REM Try python -m pip
where python >nul 2>&1
if %errorlevel% == 0 (
    echo [2/4] Using python -m pip...
    python -m pip install opencv-python mediapipe numpy pyttsx3
    if %errorlevel% == 0 goto :success
)

REM Try py -m pip
where py >nul 2>&1
if %errorlevel% == 0 (
    echo [3/4] Using py -m pip...
    py -m pip install opencv-python mediapipe numpy pyttsx3
    if %errorlevel% == 0 goto :success
)

REM Try conda
where conda >nul 2>&1
if %errorlevel% == 0 (
    echo [4/4] Using conda...
    conda install -y opencv mediapipe numpy pyttsx3 -c conda-forge
    if %errorlevel% == 0 goto :success
)

REM Try Anaconda Python directly
if exist "C:\Users\%USERNAME%\anaconda3\python.exe" (
    echo [Alternative] Using Anaconda Python...
    "C:\Users\%USERNAME%\anaconda3\python.exe" -m pip install opencv-python mediapipe numpy pyttsx3
    if %errorlevel% == 0 goto :success
)

echo.
echo ========================================
echo ERROR: Could not find Python/pip!
echo ========================================
echo.
echo Please install Python first:
echo 1. Download from: https://www.python.org/downloads/
echo 2. Make sure to check "Add Python to PATH" during installation
echo 3. Then run this script again
echo.
echo OR manually install:
echo   pip install opencv-python mediapipe numpy pyttsx3
echo.
pause
exit /b 1

:success
echo.
echo ========================================
echo Installation Complete!
echo ========================================
echo.
echo You can now run the app with:
echo   python st_hdc_airwriting.py
echo   OR
echo   Double-click run_app.bat
echo.
pause

