@echo off
echo ========================================
echo Testing Python Installation
echo ========================================
echo.

REM Test Python
python --version >nul 2>&1
if %errorlevel% == 0 (
    echo [SUCCESS] Python found!
    python --version
    echo.
    echo Installing required packages...
    echo This may take a few minutes...
    echo.
    python -m pip install opencv-python mediapipe numpy pyttsx3
    if %errorlevel% == 0 (
        echo.
        echo ========================================
        echo [SUCCESS] All packages installed!
        echo ========================================
        echo.
        echo Testing installation...
        python -c "import cv2, mediapipe, numpy, pyttsx3; print('[SUCCESS] All packages work!')"
        echo.
        echo You can now run the app with:
        echo   python st_hdc_airwriting.py
        echo   OR double-click run_app.bat
        pause
        exit /b 0
    ) else (
        echo.
        echo [ERROR] Package installation failed
        echo Try running as Administrator
        pause
        exit /b 1
    )
)

REM Try py launcher
py --version >nul 2>&1
if %errorlevel% == 0 (
    echo [SUCCESS] Python found via py launcher!
    py --version
    echo.
    echo Installing required packages...
    echo This may take a few minutes...
    echo.
    py -m pip install opencv-python mediapipe numpy pyttsx3
    if %errorlevel% == 0 (
        echo.
        echo ========================================
        echo [SUCCESS] All packages installed!
        echo ========================================
        echo.
        echo Testing installation...
        py -c "import cv2, mediapipe, numpy, pyttsx3; print('[SUCCESS] All packages work!')"
        echo.
        echo You can now run the app with:
        echo   py st_hdc_airwriting.py
        echo   OR double-click run_app.bat
        pause
        exit /b 0
    )
)

echo.
echo ========================================
echo [ERROR] Python not found in PATH
echo ========================================
echo.
echo Python is installed but not accessible.
echo.
echo SOLUTION:
echo 1. CLOSE this window completely
echo 2. Open a NEW PowerShell or Command Prompt
echo 3. Navigate to: cd C:\Users\pooja\finalyearproject
echo 4. Run this script again: test_and_install.bat
echo.
echo OR manually test:
echo   python --version
echo   python -m pip install opencv-python mediapipe numpy pyttsx3
echo.
pause

