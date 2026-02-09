# 🎯 How to Get Text Output - Step by Step

## ⚠️ IMPORTANT: You MUST Enroll Letters First!

The app uses **one-shot learning** - you need to **teach it** each letter before it can recognize them.

---

## 📝 Step-by-Step Process

### Step 1: Enroll a Letter (TEACH the app)

1. **Make sure you see your hand** in the camera window (green dots on your hand)

2. **Press `e` key** until you see:
   ```
   Mode: enroll
   ```

3. **Press `n` key** to choose a letter:
   - Keep pressing `n` until you see: `Enroll label: A` (or B, C, etc.)

4. **Press `r` key** to START recording
   - You should see: `Recording: True`

5. **Write the letter in the air** with your hand:
   - Make the letter shape clearly
   - Keep your hand visible to the camera
   - Take 2-3 seconds to complete the letter

6. **Press `r` key again** to STOP recording
   - You should see: `Last: Enrolled 'A'` (or your letter)
   - In the PowerShell window, you should see: `✓ Enrolled 'A'`

7. **Repeat for more letters:**
   - Press `n` to change to letter B
   - Press `r`, write B, press `r` again
   - Continue for C, D, E, etc.

---

### Step 2: Recognize Letters (GET OUTPUT)

1. **Press `e` key** to switch to:
   ```
   Mode: recognize
   ```

2. **Press `r` key** to START recording

3. **Write the SAME letter** you enrolled (e.g., if you enrolled 'A', write 'A')

4. **Press `r` key** to STOP recording

5. **You should see:**
   - On screen: `Last: Recognized 'A' (0.xx)`
   - On screen: `Text: A` (the letter appears!)
   - The letter is spoken aloud
   - In PowerShell: `✓ Recognized 'A' (similarity: 0.xx)`

---

## ✅ Success Checklist

- [ ] I can see my hand with green dots in the camera
- [ ] I enrolled at least one letter (saw "Enrolled 'X'")
- [ ] I switched to recognize mode
- [ ] I wrote the same letter I enrolled
- [ ] I see text appearing on screen

---

## ❌ Common Issues

### "Unknown gesture" appears
**Problem:** The gesture doesn't match what you enrolled
**Solution:**
- Write the letter EXACTLY the same way you enrolled it
- Make sure you enrolled it first
- Try enrolling the same letter again (it will overwrite)

### No text appears
**Problem:** Recognition threshold too high or gesture too different
**Solution:**
- Make sure you enrolled the letter first
- Write the letter the same way you enrolled it
- Try enrolling the letter again with a clearer gesture

### "Recording: True" but nothing happens
**Problem:** Hand not detected
**Solution:**
- Make sure your hand is visible in the camera
- Make sure you see green dots on your hand
- Try better lighting
- Move closer to the camera

### Similarity score is low (e.g., 0.02)
**Problem:** Gesture doesn't match enrolled gesture
**Solution:**
- The gesture you're writing is too different from what you enrolled
- Try enrolling again with a more consistent gesture
- Write the letter the same way each time

---

## 💡 Tips for Best Results

1. **Consistent gestures:** Write each letter the same way every time
2. **Clear shapes:** Make distinct letter shapes (not too fast)
3. **Good lighting:** Make sure your hand is well-lit
4. **Centered:** Keep your hand in the center of the camera view
5. **Complete gestures:** Finish the entire letter before stopping
6. **Practice:** Enroll each letter 2-3 times for better recognition

---

## 🎮 Quick Reference

| Action | Key | What You See |
|--------|-----|--------------|
| Start/Stop recording | `r` | Recording: True/False |
| Switch mode | `e` | Mode: enroll/recognize |
| Change letter | `n` | Enroll label: A→B→C... |
| Quit | `q` | App closes |

---

## 📊 Understanding the Output

When you recognize a letter, you'll see:
- **Similarity score** (e.g., 0.85): How well it matches (higher = better, >0.05 = recognized)
- **Text line**: Letters appear here as you recognize them
- **Last status**: Shows what just happened

---

## 🆘 Still Not Working?

1. **Check PowerShell window** - look for error messages
2. **Make sure you enrolled first** - you can't recognize without enrolling
3. **Try a simple letter** - Start with 'A' or 'O' (easier shapes)
4. **Share the similarity score** - if it's very low (<0.05), the gesture doesn't match

---

**Remember: Enroll FIRST, then Recognize!** 🎯

