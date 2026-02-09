# 🚀 START HERE - Get Your App Working!

## ✅ Quick Start (3 Steps)

### Step 1: Install Dependencies
Open PowerShell in your project folder and run:
```powershell
pip install opencv-python mediapipe numpy pyttsx3
```

**OR** if `pip` doesn't work, try:
```powershell
python -m pip install opencv-python mediapipe numpy pyttsx3
```

**OR** if you have Anaconda:
```powershell
conda install opencv mediapipe numpy pyttsx3
```

### Step 2: Run the App

**Option A - Easiest (Double-click):**
- Double-click `run_app.bat` in File Explorer

**Option B - PowerShell:**
```powershell
python st_hdc_airwriting_fixed.py
```

**OR** if that doesn't work:
```powershell
py st_hdc_airwriting_fixed.py
```

### Step 3: Use the App
1. A window will open showing your webcam
2. **Enroll a letter first:**
   - Press `e` until you see "Mode: enroll"
   - Press `n` to choose a letter (A, B, C, etc.)
   - Press `r` to START recording
   - Write the letter in the air with your hand
   - Press `r` again to STOP
   - You should see "Enrolled 'A'" (or your letter)

3. **Recognize letters:**
   - Press `e` to switch to "Mode: recognize"
   - Press `r` to START recording
   - Write the same letter in the air
   - Press `r` to STOP
   - The letter appears on screen and is spoken!

---

## 🎮 Controls

| Key | Action |
|-----|--------|
| `r` | Start/Stop recording a gesture |
| `e` | Toggle between enroll/recognize mode |
| `n` | Change enrollment label (A-Z) |
| `q` | Quit the application |

---

## ❌ Troubleshooting

### "Python was not found"
**Solution:** Try these in order:
1. `py st_hdc_airwriting_fixed.py`
2. `python3 st_hdc_airwriting_fixed.py`
3. Double-click `run_app.bat`

### "ModuleNotFoundError: No module named 'mediapipe'"
**Solution:**
```powershell
pip install mediapipe opencv-python numpy pyttsx3
```

### "Error: Hand landmark model unavailable"
**Solution:**
- Check your internet connection
- Wait for the download to complete (first run only)
- The file `hand_landmarker.task` should appear in your folder

### "Error: Could not open webcam"
**Solution:**
- Close other apps using the webcam (Zoom, Teams, etc.)
- Make sure your webcam is connected
- Try restarting the app

### App window doesn't appear
**Solution:**
- Check the PowerShell window for error messages
- Make sure all dependencies are installed
- Try running `python test_mediapipe.py` first to test imports

---

## 📋 What You'll See

When the app runs successfully, you'll see:
- A window titled "ST-HDC Air Writing"
- Your webcam feed
- Green dots on your hand (landmarks)
- Status text showing:
  - Mode: enroll/recognize
  - Enroll label: A-Z
  - Recording: True/False
  - Last: Last action result
  - Text: Recognized letters appear here

---

## 🎯 How It Works

1. **Enrollment (One-Shot Learning):**
   - You perform a gesture once
   - The app encodes it as a Spatio-Temporal Hypervector (ST-HDC)
   - Stores it in associative memory

2. **Recognition:**
   - You perform the same gesture
   - App computes its ST-HDC hypervector
   - Compares it to stored gestures using cosine similarity
   - Outputs the best match as text + speech

---

## 💡 Tips for Best Results

1. **Consistent gestures:** Try to perform the same gesture the same way each time
2. **Good lighting:** Make sure your hand is well-lit
3. **Centered hand:** Keep your hand in the center of the camera view
4. **Complete gestures:** Finish the entire letter/word before stopping recording
5. **Practice:** Enroll each letter a few times for better recognition

---

## 📞 Still Not Working?

1. **Share the exact error message** from PowerShell
2. **Run the test script:**
   ```powershell
   python test_mediapipe.py
   ```
   Share the output

3. **Check your Python version:**
   ```powershell
   python --version
   ```
   (Should be 3.7 or higher)

4. **Check installed packages:**
   ```powershell
   pip list | findstr "mediapipe opencv numpy"
   ```

---

## ✨ Success!

Once it's working:
- Enroll letters A-Z
- Write words in the air
- See text appear on screen
- Hear text-to-speech output
- Enjoy your brain-inspired ST-HDC system! 🎉

