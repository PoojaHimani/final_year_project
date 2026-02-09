# Install Dependencies Script
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Installing Required Dependencies" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

$packages = @("opencv-python", "mediapipe", "numpy", "pyttsx3")
$pythonFound = $false

# Function to try installing with a Python command
function Install-WithPython {
    param($pythonCmd)
    
    Write-Host "Trying: $pythonCmd -m pip install ..." -ForegroundColor Yellow
    
    # Check if command exists
    $cmd = Get-Command $pythonCmd -ErrorAction SilentlyContinue
    if ($cmd) {
        try {
            & $pythonCmd -m pip install $packages
            if ($LASTEXITCODE -eq 0) {
                Write-Host "[SUCCESS] Successfully installed packages!" -ForegroundColor Green
                return $true
            }
        } catch {
            Write-Host "[FAILED] $_" -ForegroundColor Red
        }
    }
    return $false
}

# Try different Python commands
$pythonCommands = @("python", "py", "python3", "python3.11", "python3.10", "python3.9")

foreach ($cmd in $pythonCommands) {
    if (Install-WithPython $cmd) {
        $pythonFound = $true
        break
    }
}

# Try to find Python in common locations
if (-not $pythonFound) {
    Write-Host "`nSearching for Python in common locations..." -ForegroundColor Yellow
    
    $pythonPaths = @(
        "$env:USERPROFILE\anaconda3\python.exe",
        "$env:USERPROFILE\miniconda3\python.exe",
        "$env:LOCALAPPDATA\Programs\Python\*\python.exe",
        "C:\Python*\python.exe",
        "$env:PROGRAMFILES\Python*\python.exe"
    )
    
    foreach ($pathPattern in $pythonPaths) {
        $found = Get-ChildItem -Path $pathPattern -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($found) {
            Write-Host "Found Python at: $($found.FullName)" -ForegroundColor Green
            try {
                & $found.FullName -m pip install $packages
                if ($LASTEXITCODE -eq 0) {
                    $pythonFound = $true
                    break
                }
            } catch {
                Write-Host "Failed to install: $_" -ForegroundColor Red
            }
        }
    }
}

# Check if packages are already installed
if (-not $pythonFound) {
    Write-Host "`nChecking if packages are already installed..." -ForegroundColor Yellow
    
    # Try to import to check
    $testScript = @"
import sys
try:
    import cv2
    import mediapipe
    import numpy
    import pyttsx3
    print('[SUCCESS] All packages are already installed!')
    sys.exit(0)
except ImportError as e:
    print(f'[MISSING] {e}')
    sys.exit(1)
"@
    
    foreach ($cmd in $pythonCommands) {
        $testCmd = Get-Command $cmd -ErrorAction SilentlyContinue
        if ($testCmd) {
            try {
                $testScript | & $cmd
                if ($LASTEXITCODE -eq 0) {
                    Write-Host "`n[SUCCESS] All required packages are already installed!" -ForegroundColor Green
                    $pythonFound = $true
                    break
                }
            } catch {}
        }
    }
}

if (-not $pythonFound) {
    Write-Host "`n========================================" -ForegroundColor Red
    Write-Host "ERROR: Could not install packages!" -ForegroundColor Red
    Write-Host "========================================" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please install Python first:" -ForegroundColor Yellow
    Write-Host "1. Download from: https://www.python.org/downloads/" -ForegroundColor White
    Write-Host "2. Make sure to check 'Add Python to PATH' during installation" -ForegroundColor White
    Write-Host "3. Then run this script again" -ForegroundColor White
    Write-Host ""
    Write-Host "OR manually install in PowerShell:" -ForegroundColor Yellow
    Write-Host "  pip install opencv-python mediapipe numpy pyttsx3" -ForegroundColor White
    Write-Host ""
    exit 1
} else {
    Write-Host "`n========================================" -ForegroundColor Green
    Write-Host "Installation Complete!" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "You can now run the app with:" -ForegroundColor Cyan
    Write-Host "  python st_hdc_airwriting.py" -ForegroundColor White
    Write-Host "  OR" -ForegroundColor White
    Write-Host "  Double-click run_app.bat" -ForegroundColor White
    Write-Host ""
}

