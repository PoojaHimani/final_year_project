@echo off
echo ========================================
echo ST-HDC Air Writing App Launcher
echo ========================================
echo.

REM Try different Python commands
where python >nul 2>&1
if %errorlevel% == 0 (
    echo Using: python
    python st_hdc_airwriting.py
    if %errorlevel% == 0 goto :end
)

where py >nul 2>&1
if %errorlevel% == 0 (
    echo Using: py
    py st_hdc_airwriting.py
    if %errorlevel% == 0 goto :end
)

REM Try conda if available
where conda >nul 2>&1
if %errorlevel% == 0 (
    echo Using: conda python
    conda run python st_hdc_airwriting.py
    if %errorlevel% == 0 goto :end
)

REM Try common Python paths
if exist "C:\Users\%USERNAME%\anaconda3\python.exe" (
    echo Using: Anaconda Python
    "C:\Users\%USERNAME%\anaconda3\python.exe" st_hdc_airwriting.py
    if %errorlevel% == 0 goto :end
)

if exist "C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python*\python.exe" (
    echo Using: Local Python
    for %%P in ("C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python*\python.exe") do (
        "%%P" st_hdc_airwriting_fixed.py
        goto :end
    )
)

echo ========================================
echo ERROR: Python not found!
echo ========================================
echo.
echo Python is required to run this app.
echo.
echo QUICK FIX:
echo   1. Install Python from Microsoft Store
echo      - Press Win key, search "Microsoft Store"
echo      - Search "Python 3.11" and install
echo   2. OR download from: https://www.python.org/downloads/
echo      - Make sure to check "Add Python to PATH"
echo   3. After installation, RESTART this terminal
echo   4. Run this script again
echo.
echo For detailed instructions, see: FIX_NOT_RUNNING.md
echo.
pause

:end
pause

