# 📦 Installation Guide - Install All Dependencies

## ⚠️ Important: Python Must Be Installed First

If you see "Python was not found", you need to install Python first.

---

## Step 1: Install Python (If Not Already Installed)

### Option A: Download Python
1. Go to: https://www.python.org/downloads/
2. Download Python 3.9 or higher
3. **IMPORTANT:** During installation, check ✅ **"Add Python to PATH"**
4. Click "Install Now"
5. Wait for installation to complete

### Option B: Install via Microsoft Store
1. Open Microsoft Store
2. Search for "Python 3.11" or "Python 3.12"
3. Click "Install"
4. This automatically adds Python to PATH

### Option C: If You Have Anaconda
If you have Anaconda installed:
1. Open "Anaconda Prompt" (search in Start menu)
2. Navigate to your project folder:
   ```bash
   cd C:\Users\pooja\finalyearproject
   ```
3. Run the installation commands below

---

## Step 2: Install Required Packages

### Method 1: Using the Installation Script (Easiest)

**Double-click:** `install_dependencies.bat`

**OR** in PowerShell:
```powershell
.\install_dependencies.bat
```

### Method 2: Manual Installation

Open PowerShell in your project folder and run:

```powershell
pip install opencv-python mediapipe numpy pyttsx3
```

**If `pip` doesn't work, try:**
```powershell
python -m pip install opencv-python mediapipe numpy pyttsx3
```

**OR:**
```powershell
py -m pip install opencv-python mediapipe numpy pyttsx3
```

### Method 3: Using Anaconda (If You Have It)

Open **Anaconda Prompt** and run:
```bash
conda install -y opencv mediapipe numpy pyttsx3 -c conda-forge
```

---

## Step 3: Verify Installation

Test if everything is installed:

```powershell
python test_mediapipe.py
```

**OR:**
```powershell
python -c "import cv2, mediapipe, numpy, pyttsx3; print('All packages installed!')"
```

If you see "All packages installed!" - you're ready! ✅

---

## 📋 What Gets Installed

- **opencv-python** - For webcam and image processing
- **mediapipe** - For hand tracking (3D landmarks)
- **numpy** - For mathematical operations (hypervectors)
- **pyttsx3** - For text-to-speech output

---

## ❌ Troubleshooting

### "Python was not found"
**Solution:** Install Python first (see Step 1)

### "pip is not recognized"
**Solution:** Try:
- `python -m pip install ...`
- `py -m pip install ...`
- Or reinstall Python with "Add to PATH" checked

### "Permission denied" or "Access denied"
**Solution:** Run PowerShell as Administrator:
1. Right-click PowerShell
2. Select "Run as Administrator"
3. Navigate to project folder
4. Run installation command

### "ModuleNotFoundError" after installation
**Solution:** Make sure you're using the same Python that has the packages:
```powershell
python -m pip list
```
Check if packages appear in the list.

### MediaPipe installation fails
**Solution:** 
```powershell
pip install --upgrade pip
pip install mediapipe
```

---

## ✅ Success!

Once installation is complete:
1. Run the app: `python st_hdc_airwriting.py`
2. OR double-click: `run_app.bat`

---

## 🆘 Still Having Issues?

1. **Check Python version:**
   ```powershell
   python --version
   ```
   Should show 3.7 or higher

2. **Check if pip works:**
   ```powershell
   pip --version
   ```

3. **List installed packages:**
   ```powershell
   pip list
   ```

4. **Share the exact error message** if you need help

---

**After installation, you're ready to run the ST-HDC Air Writing App! 🎉**

