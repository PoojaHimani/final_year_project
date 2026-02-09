# ST-HDC Air Writing App
## Spatio-Temporal Hyperdimensional Computing for Hand Gesture Recognition

A real-time system that converts 3D hand gestures into text using Hyperdimensional Computing (HDC) instead of deep learning.

---

## 🚀 Quick Start

### 1. Install Dependencies
```powershell
pip install opencv-python mediapipe numpy pyttsx3
```

### 2. Run the App

**Easiest way:** Double-click `run_app.bat`

**OR** in PowerShell:
```powershell
python st_hdc_airwriting.py
```

### 3. Use It!
- Press `e` to switch to "enroll" mode
- Press `n` to choose a letter (A-Z)
- Press `r` to start recording, write the letter in air, press `r` again to stop
- Press `e` to switch to "recognize" mode
- Press `r` to record and recognize your gesture
- See text appear on screen and hear it spoken!

---

## 📁 Files in This Project

- **`st_hdc_airwriting.py`** - Main application (FIXED VERSION - ready to use!)
- **`run_app.bat`** - Easy launcher (double-click to run)
- **`START_HERE.md`** - Detailed step-by-step guide
- **`requirements.txt`** - Python dependencies
- **`test_mediapipe.py`** - Test script to check your setup
- **`TROUBLESHOOTING.md`** - Common issues and solutions

---

## 🎮 Controls

| Key | Action |
|-----|--------|
| `r` | Start/Stop recording a gesture |
| `e` | Toggle between enroll/recognize mode |
| `n` | Change enrollment label (A-Z) |
| `q` | Quit the application |

---

## 🧠 How It Works

### Spatio-Temporal Hyperdimensional Computing (ST-HDC)

1. **Spatial Encoding:** Each frame's 3D hand pose is encoded as a spatial hypervector
   - 21 hand landmarks × 3D positions → quantized bins
   - Binding: landmark identity ⊗ position bins
   - Superposition: combine all landmarks

2. **Temporal Encoding:** Frame sequence is encoded with circular shifts
   - Each frame hypervector is circularly shifted by frame index
   - Superposition creates the final spatio-temporal hypervector

3. **One-Shot Learning:** Store gesture once, recognize immediately
   - Associative memory stores hypervectors
   - Recognition via cosine similarity

4. **Output:** Text on screen + Text-to-Speech

---

## ✨ Features

- ✅ Real-time 3D hand tracking (MediaPipe)
- ✅ Spatio-Temporal Hyperdimensional Computing
- ✅ One-shot learning (no training required)
- ✅ Low power, fast inference
- ✅ Text output + Speech synthesis
- ✅ Accessible for deaf, mute, blind users

---

## 📋 Requirements

- Python 3.7+
- Webcam
- Internet connection (for first-time model download)
- Windows/Linux/Mac

---

## 🐛 Troubleshooting

See **`START_HERE.md`** for detailed troubleshooting steps.

Common issues:
- **Python not found:** Try `py` instead of `python`, or use `run_app.bat`
- **MediaPipe errors:** Run `pip install --upgrade mediapipe`
- **Webcam not working:** Close other apps using the camera
- **Model download fails:** Check internet connection

---

## 📚 Technical Details

- **Hypervector Dimension:** 10,000
- **Spatial Bins:** 8 per axis (x, y, z)
- **Temporal Shift Step:** 1
- **Recognition Threshold:** 0.05 (cosine similarity)

---

## 🎓 For Your Final Year Project

This implementation demonstrates:
- Brain-inspired computing (HDC)
- Volumetric gesture recognition
- One-shot learning
- Real-time inference
- Accessibility applications

---

## 📞 Need Help?

1. Check `START_HERE.md` for step-by-step instructions
2. Check `TROUBLESHOOTING.md` for common issues
3. Run `python test_mediapipe.py` to test your setup
4. Share the exact error message if still having issues

---

## 🎉 Success!

Once working:
- Enroll letters A-Z
- Write words in the air
- See text appear in real-time
- Hear text-to-speech output
- Enjoy your ST-HDC system!

---

**Made with ❤️ for your Final Year Project**

