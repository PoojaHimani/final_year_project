# Troubleshooting Guide

## If the app is not running, please share the EXACT error message

When you run `python st_hdc_airwriting.py` (or `py st_hdc_airwriting.py`), copy the **entire error message** from PowerShell and share it.

## Common Issues:

### 1. "Python was not found"
**Solution:** 
- Use: `py st_hdc_airwriting.py` instead of `python`
- Or activate your conda environment first:
  ```powershell
  conda activate base
  python st_hdc_airwriting.py
  ```

### 2. "ModuleNotFoundError: No module named 'mediapipe'"
**Solution:**
```powershell
pip install mediapipe opencv-python numpy pyttsx3
```

### 3. "AttributeError: module 'mediapipe' has no attribute 'solutions'"
**Solution:** This means you have MediaPipe 0.10.x which uses a different API. The code should handle this, but if you see this error:
```powershell
pip install --upgrade mediapipe
```

### 4. "Error: Hand landmark model unavailable"
**Solution:** 
- Check your internet connection (model downloads on first run)
- Wait for the download to complete
- The model file `hand_landmarker.task` should appear in your project folder

### 5. "Error: Could not open webcam"
**Solution:**
- Make sure your webcam is connected
- Close other apps using the webcam (Zoom, Teams, etc.)
- Try a different camera index: Change `cv2.VideoCapture(0)` to `cv2.VideoCapture(1)` in the code

### 6. Import errors with Image
**Solution:** The code now tries multiple import methods. If you see Image-related errors, try:
```powershell
pip uninstall mediapipe
pip install mediapipe==0.10.31
```

## Test Your Setup

Run this test script first to check if everything is installed:

```powershell
python test_mediapipe.py
```

This will show you which imports work and which don't.

## Still Not Working?

1. **Share the complete error message** from PowerShell
2. **Share your Python version:** `python --version`
3. **Share your MediaPipe version:** `pip show mediapipe`

