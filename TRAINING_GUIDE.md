# 🎓 Training Guide - Train All Letters Automatically

## 🚀 Quick Start: Training Mode

The app now has a **TRAINING MODE** that makes it easy to train all letters A-Z!

---

## 📝 Step-by-Step Training

### Step 1: Enter Training Mode

1. **Run the app:**
   ```powershell
   python st_hdc_airwriting.py
   ```

2. **Press `t` key** to enter Training Mode
   - You'll see: `Mode: enroll` and `TRAINING MODE: Training all letters A-Z`
   - The app automatically starts with letter **A**

### Step 2: Train Each Letter

For each letter (A, B, C, ... Z):

1. **Look at the screen** - it shows: `Enroll label: A` (or current letter)

2. **Press `r` key** to START recording
   - You'll see: `Recording: True`

3. **Write the letter in the air** with your hand:
   - Make the letter shape clearly
   - Take 2-3 seconds
   - Keep your hand visible to the camera

4. **Press `r` key again** to STOP recording
   - You'll see: `Enrolled 'A'` (or current letter)
   - **The app automatically moves to the next letter!**
   - Example: After training A, it shows `Enroll label: B`

5. **Repeat** for the next letter (B, C, D, ...)

### Step 3: Complete Training

- After training **Z**, it cycles back to **A**
- You can train letters multiple times (overwrites previous training)
- Train all 26 letters: A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y, Z

### Step 4: Save Your Training

1. **Press `s` key** to save all trained letters
   - You'll see: `Saved X letters: A, B, C, ...`
   - Models are saved to: `trained_letters.npz`

2. **Done!** Your training is saved permanently

---

## ✅ What Happens After Training

### Auto-Load on Next Run

When you run the app again:
- **It automatically loads your trained letters!**
- You'll see: `✓ Loaded X trained letters: A, B, C, ...`
- **No need to train again!**

### Start Recognizing

1. **Press `e`** to switch to `Mode: recognize`
2. **Press `r`** to record a gesture
3. **Write any letter** you trained
4. **Press `r`** to stop
5. **See the letter appear on screen!**

---

## 🎮 All Controls

| Key | Action |
|-----|--------|
| `t` | **Enter/Exit Training Mode** (trains all letters A-Z) |
| `r` | Start/Stop recording a gesture |
| `e` | Toggle enroll/recognize mode |
| `n` | Change enrollment label (A-Z) |
| `s` | **Save trained models** (saves to file) |
| `q` | Quit the application |

---

## 💡 Training Tips

1. **Consistent gestures:** Write each letter the same way every time
2. **Clear shapes:** Make distinct letter shapes (not too fast)
3. **Good lighting:** Make sure your hand is well-lit
4. **Centered:** Keep your hand in the center of the camera view
5. **Complete gestures:** Finish the entire letter before stopping
6. **Practice:** You can re-train any letter by going back to it

---

## 📊 Training Progress

The app shows:
- **Trained: X/26 letters** - How many letters you've trained
- **Current letter** - Which letter you're training now
- **Auto-advance** - Automatically moves to next letter after training

---

## 🔄 Re-Training Letters

If you want to re-train a specific letter:

1. **Press `n`** to navigate to that letter
2. **Press `r`** to record
3. **Write the letter** again
4. **Press `r`** to stop
5. It overwrites the previous training for that letter

---

## 💾 Saving and Loading

### Save Models
- **Press `s`** anytime to save
- Models are saved to: `trained_letters.npz`
- **Auto-saves** after each letter in training mode

### Load Models
- **Automatic** - loads on app startup
- If `trained_letters.npz` exists, it loads automatically
- You'll see: `✓ Loaded X trained letters`

### Delete Models
- Delete the file: `trained_letters.npz`
- App will start fresh

---

## ❓ Common Questions

### Q: Do I need to train every time?
**A:** No! Once you train and save, models auto-load on next run.

### Q: Can I train some letters now and more later?
**A:** Yes! Train what you need, save, and add more later.

### Q: What if I make a mistake while training?
**A:** Just navigate to that letter with `n` and re-train it.

### Q: How many times should I train each letter?
**A:** Once is enough! But you can train multiple times to refine.

### Q: Can I train words instead of letters?
**A:** Currently it's letter-based, but you can train whole words as single gestures (e.g., train "HELLO" as one gesture).

---

## 🎯 Recommended Workflow

1. **First time:** Press `t`, train all 26 letters, press `s` to save
2. **Next time:** Just run the app - it loads automatically!
3. **Use it:** Switch to recognize mode and start writing!

---

## 🆘 Troubleshooting

### "No saved models found"
- You need to train letters first
- Press `t` to enter training mode

### Training not saving
- Make sure you press `s` to save
- Check if `trained_letters.npz` file exists in your project folder

### Letters not recognizing
- Make sure you trained them first
- Write the letter the same way you trained it
- Try re-training the letter

---

**Happy Training! Once trained, the app will recognize your letters automatically! 🎉**

