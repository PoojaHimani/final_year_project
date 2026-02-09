@echo off
echo ========================================
echo Complete Installation Script
echo Installing Python + All Dependencies
echo ========================================
echo.

REM Check if Python is already installed and working
python --version >nul 2>&1
if %errorlevel% == 0 (
    echo Python is already installed!
    python --version
    echo.
    echo Installing required packages...
    python -m pip install opencv-python mediapipe numpy pyttsx3
    if %errorlevel% == 0 (
        echo.
        echo ========================================
        echo SUCCESS! All packages installed!
        echo ========================================
        echo.
        echo You can now run: python st_hdc_airwriting.py
        pause
        exit /b 0
    )
)

REM Try py launcher
py --version >nul 2>&1
if %errorlevel% == 0 (
    echo Python found via py launcher!
    py --version
    echo.
    echo Installing required packages...
    py -m pip install opencv-python mediapipe numpy pyttsx3
    if %errorlevel% == 0 (
        echo.
        echo ========================================
        echo SUCCESS! All packages installed!
        echo ========================================
        echo.
        echo You can now run: py st_hdc_airwriting.py
        pause
        exit /b 0
    )
)

echo.
echo ========================================
echo Python is NOT installed or not in PATH
echo ========================================
echo.
echo You have 3 options:
echo.
echo OPTION 1: Install Python from Microsoft Store (Easiest)
echo   1. Press Win key, search "Microsoft Store"
echo   2. Search for "Python 3.11" or "Python 3.12"
echo   3. Click Install
echo   4. After installation, run this script again
echo.
echo OPTION 2: Install Python from python.org (Recommended)
echo   1. Open browser and go to: https://www.python.org/downloads/
echo   2. Download Python 3.11 or 3.12
echo   3. Run the installer
echo   4. IMPORTANT: Check "Add Python to PATH" during installation
echo   5. Click "Install Now"
echo   6. After installation, RESTART this terminal
echo   7. Run this script again
echo.
echo OPTION 3: Use Anaconda (If you have it)
echo   1. Open "Anaconda Prompt" from Start menu
echo   2. Navigate to: cd C:\Users\pooja\finalyearproject
echo   3. Run: conda install -y opencv mediapipe numpy pyttsx3 -c conda-forge
echo.
echo ========================================
echo.
echo After installing Python, run this script again!
echo.
pause

