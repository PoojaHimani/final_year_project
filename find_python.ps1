# Find Python Installation
Write-Host "Searching for Python installations..." -ForegroundColor Cyan
Write-Host ""

$found = $false

# Check PATH
Write-Host "Checking PATH environment variable..." -ForegroundColor Yellow
$pythonCommands = @("python", "py", "python3", "python3.11", "python3.10", "python3.9")
foreach ($cmd in $pythonCommands) {
    $result = Get-Command $cmd -ErrorAction SilentlyContinue
    if ($result) {
        Write-Host "  Found: $cmd at $($result.Source)" -ForegroundColor Green
        Write-Host "  Testing: " -NoNewline
        $version = & $cmd --version 2>&1
        Write-Host $version -ForegroundColor Cyan
        $found = $true
    }
}

# Check common installation locations
Write-Host "`nChecking common installation locations..." -ForegroundColor Yellow
$locations = @(
    "$env:USERPROFILE\anaconda3\python.exe",
    "$env:USERPROFILE\miniconda3\python.exe",
    "$env:LOCALAPPDATA\Programs\Python",
    "$env:PROGRAMFILES\Python*",
    "C:\Python*",
    "$env:APPDATA\Python"
)

foreach ($loc in $locations) {
    $paths = Get-ChildItem -Path $loc -Filter "python.exe" -Recurse -ErrorAction SilentlyContinue -Depth 2 | Select-Object -First 3
    if ($paths) {
        foreach ($path in $paths) {
            Write-Host "  Found: $($path.FullName)" -ForegroundColor Green
            Write-Host "    Testing: " -NoNewline
            try {
                $version = & $path.FullName --version 2>&1
                Write-Host $version -ForegroundColor Cyan
                $found = $true
            } catch {
                Write-Host "Error" -ForegroundColor Red
            }
        }
    }
}

# Check Windows Store Python
Write-Host "`nChecking Windows Store Python..." -ForegroundColor Yellow
$storePython = "$env:LOCALAPPDATA\Microsoft\WindowsApps\python.exe"
if (Test-Path $storePython) {
    Write-Host "  Found Windows Store Python stub" -ForegroundColor Yellow
    Write-Host "  Note: This is usually just a launcher, not the actual Python" -ForegroundColor Yellow
}

# Check if packages are installed anywhere
Write-Host "`nChecking for installed packages..." -ForegroundColor Yellow
$sitePackages = @(
    "$env:USERPROFILE\anaconda3\Lib\site-packages",
    "$env:USERPROFILE\miniconda3\Lib\site-packages",
    "$env:LOCALAPPDATA\Programs\Python\*\Lib\site-packages"
)

foreach ($sp in $sitePackages) {
    $paths = Get-ChildItem -Path $sp -ErrorAction SilentlyContinue -Depth 1
    if ($paths) {
        $hasMediapipe = Get-ChildItem -Path $sp -Filter "mediapipe" -Directory -ErrorAction SilentlyContinue
        $hasOpencv = Get-ChildItem -Path $sp -Filter "cv2" -Directory -ErrorAction SilentlyContinue
        if ($hasMediapipe -or $hasOpencv) {
            Write-Host "  Found packages at: $sp" -ForegroundColor Green
            if ($hasMediapipe) { Write-Host "    - mediapipe installed" -ForegroundColor Cyan }
            if ($hasOpencv) { Write-Host "    - opencv installed" -ForegroundColor Cyan }
            $found = $true
        }
    }
}

Write-Host ""
if (-not $found) {
    Write-Host "========================================" -ForegroundColor Red
    Write-Host "Python NOT FOUND!" -ForegroundColor Red
    Write-Host "========================================" -ForegroundColor Red
    Write-Host ""
    Write-Host "You need to install Python first:" -ForegroundColor Yellow
    Write-Host "1. Download from: https://www.python.org/downloads/" -ForegroundColor White
    Write-Host "2. During installation, check 'Add Python to PATH'" -ForegroundColor White
    Write-Host "3. Restart your terminal/PowerShell after installation" -ForegroundColor White
} else {
    Write-Host "========================================" -ForegroundColor Green
    Write-Host "Python found! You can now install packages." -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Green
}

