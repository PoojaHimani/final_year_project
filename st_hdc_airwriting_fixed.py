"""
ST-HDC Air Writing App - Fixed Version
Spatio-Temporal Hyperdimensional Computing for Hand Gesture Recognition
"""
import cv2
import numpy as np
import os
import sys
import urllib.request
import pyttsx3

# Try to import MediaPipe with multiple fallback methods
try:
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
    from mediapipe.tasks.python.vision.core import image as mp_image
    MEDIAPIPE_VERSION = "tasks"
    print("Using MediaPipe Tasks API")
except ImportError:
    try:
        import mediapipe as mp
        MEDIAPIPE_VERSION = "solutions"
        print("Using MediaPipe Solutions API (legacy)")
    except ImportError:
        print("ERROR: MediaPipe not installed!")
        print("Install it with: pip install mediapipe")
        sys.exit(1)

# ==============================
# Configuration
# ==============================
D = 10000          # Hypervector dimensionality
Q = 8              # Spatial quantization bins per axis
SHIFT_STEP = 1     # Temporal shift step per frame
HAND_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
HAND_MODEL_PATH = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")

# ==============================
# HDC Primitives
# ==============================
def random_hv():
    """Generate a random bipolar hypervector."""
    return np.random.choice([-1, 1], size=D)

# Pre-generate landmark and spatial bin hypervectors
LANDMARK_HV = [random_hv() for _ in range(21)]          # 21 MediaPipe landmarks
X_BIN = [random_hv() for _ in range(Q)]
Y_BIN = [random_hv() for _ in range(Q)]
Z_BIN = [random_hv() for _ in range(Q)]

def quantize01(v, bins=Q):
    """Quantize a value in [0,1] into an integer bin index."""
    v = max(0.0, min(1.0, float(v)))
    return min(bins - 1, int(v * bins))

def circ_shift(hv, shift):
    """Circularly shift a hypervector."""
    return np.roll(hv, shift)

def frame_hv(norm_landmarks):
    """
    Compute spatial hypervector S_t for a single frame.
    norm_landmarks: list of (x, y, z) in [0,1]
    """
    acc = np.zeros(D)
    for i, (x, y, z) in enumerate(norm_landmarks):
        if i >= len(LANDMARK_HV):
            break
        j = quantize01(x)
        k = quantize01(y)
        m = quantize01(z)
        h = LANDMARK_HV[i] * X_BIN[j] * Y_BIN[k] * Z_BIN[m]
        acc += h
    return acc

def encode_gesture(frame_hvs):
    """
    Encode a full gesture as a spatio-temporal hypervector.
    frame_hvs: list of spatial frame hypervectors S_t
    """
    if not frame_hvs:
        return None
    acc = np.zeros(D)
    for t, S_t in enumerate(frame_hvs):
        acc += circ_shift(S_t, t * SHIFT_STEP)
    return np.sign(acc)

def cosine_similarity(a, b):
    """Cosine similarity between two hypervectors."""
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    return float(np.dot(a, b) / denom)

# ==============================
# Associative Memory (One-shot)
# ==============================
class OneShotMemory:
    def __init__(self):
        self.store = {}  # label -> hypervector

    def enroll(self, label, hv):
        """Store / overwrite the hypervector for a label."""
        if hv is None:
            return
        self.store[label] = hv

    def recognize(self, hv, threshold=0.05):
        """
        Recognize a hypervector against stored labels.
        Returns (label or None, similarity).
        """
        if hv is None or not self.store:
            return None, -1.0

        best_label = None
        best_sim = -1.0
        for label, stored_hv in self.store.items():
            sim = cosine_similarity(hv, stored_hv)
            if sim > best_sim:
                best_sim = sim
                best_label = label

        if best_sim < threshold:
            return None, best_sim
        return best_label, best_sim

# ==============================
# Text-to-Speech
# ==============================
def init_tts():
    try:
        engine = pyttsx3.init()
        return engine
    except Exception:
        return None

def speak(engine, text):
    if engine is None or not text:
        return
    try:
        engine.say(text)
        engine.runAndWait()
    except Exception:
        pass

# ==============================
# MediaPipe Hand Tracking Setup
# ==============================
def ensure_hand_model():
    """Download the MediaPipe hand landmarker model if missing."""
    if os.path.exists(HAND_MODEL_PATH):
        return HAND_MODEL_PATH
    try:
        print("Downloading hand landmark model (this may take a minute)...")
        urllib.request.urlretrieve(HAND_MODEL_URL, HAND_MODEL_PATH)
        print("Model downloaded successfully!")
        return HAND_MODEL_PATH
    except Exception as exc:
        print(f"Failed to download model: {exc}")
        print("Please check your internet connection and try again.")
        return None

def setup_hand_tracker():
    """Setup hand tracker based on available MediaPipe version."""
    if MEDIAPIPE_VERSION == "tasks":
        model_path = ensure_hand_model()
        if model_path is None or not os.path.exists(model_path):
            return None
        
        try:
            base_options = mp_python.BaseOptions(model_asset_path=model_path)
            options = vision.HandLandmarkerOptions(
                base_options=base_options,
                num_hands=1,
                min_hand_detection_confidence=0.5,
                min_hand_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            return vision.HandLandmarker.create_from_options(options)
        except Exception as e:
            print(f"Error creating HandLandmarker: {e}")
            return None
    else:
        # Legacy solutions API
        try:
            return mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        except Exception as e:
            print(f"Error creating Hands: {e}")
            return None

def detect_hands_tasks(image_rgb, landmarker):
    """Detect hands using Tasks API."""
    try:
        image_rgb = np.ascontiguousarray(image_rgb)
        mp_img = mp_image.Image(image_format=mp_image.ImageFormat.SRGB, data=image_rgb)
        results = landmarker.detect(mp_img)
        
        if results and results.hand_landmarks:
            hand_landmarks = results.hand_landmarks[0]
            return [(lm.x, lm.y, lm.z) for lm in hand_landmarks]
        return None
    except Exception as e:
        return None

def detect_hands_solutions(image_rgb, hands):
    """Detect hands using Solutions API (legacy)."""
    try:
        results = hands.process(image_rgb)
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            return [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
        return None
    except Exception as e:
        return None

def normalize_landmarks(landmarks):
    """
    Normalize raw MediaPipe landmarks to a canonical [0,1] space.
    landmarks: list of (x, y, z) in MediaPipe normalized image coordinates.
    """
    if not landmarks:
        return []

    xs, ys, zs = zip(*landmarks)
    xs = np.array(xs, dtype=float)
    ys = np.array(ys, dtype=float)
    zs = np.array(zs, dtype=float)

    # Translate wrist (index 0) to origin
    xs -= xs[0]
    ys -= ys[0]
    zs -= zs[0]

    # Scale based on max 2D distance from wrist to any point
    scale = np.linalg.norm(np.vstack((xs, ys)), axis=0).max() + 1e-6
    xs = xs / scale * 0.5 + 0.5
    ys = ys / scale * 0.5 + 0.5

    # Normalize depth to [0,1]
    z_min = zs.min()
    z_max = zs.max()
    if z_max - z_min < 1e-6:
        zs = np.zeros_like(zs)
    else:
        zs = (zs - z_min) / (z_max - z_min)

    return list(zip(xs, ys, zs))

# ==============================
# Main Application Loop
# ==============================
def main():
    print("=" * 50)
    print("ST-HDC Air-Writing App")
    print("=" * 50)
    
    # Setup hand tracker
    print("\nInitializing hand tracker...")
    hand_tracker = setup_hand_tracker()
    if hand_tracker is None:
        print("ERROR: Could not initialize hand tracker!")
        print("\nTroubleshooting:")
        print("1. Make sure mediapipe is installed: pip install mediapipe")
        print("2. Check your internet connection (needed for model download)")
        print("3. Try: pip install --upgrade mediapipe")
        return
    
    print("Hand tracker initialized successfully!")
    
    # Open webcam
    print("\nOpening webcam...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam!")
        print("Make sure your webcam is connected and not used by another app.")
        return
    print("Webcam opened successfully!")
    
    memory = OneShotMemory()
    tts_engine = init_tts()
    
    recording = False
    gesture_frames = []
    mode = "recognize"   # or "enroll"
    enroll_label = "A"
    text_buffer = ""
    last_status = "Ready - Press 'r' to start"
    
    print("\n" + "=" * 50)
    print("Controls:")
    print("  r : start/stop recording a gesture")
    print("  e : toggle enroll/recognize mode")
    print("  n : next label for enrollment (A-Z)")
    print("  q : quit")
    print("=" * 50)
    print("\nStarting application...\n")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error reading from webcam")
            break
        
        h, w, _ = frame.shape
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect hands
        if MEDIAPIPE_VERSION == "tasks":
            pts = detect_hands_tasks(image_rgb, hand_tracker)
        else:
            pts = detect_hands_solutions(image_rgb, hand_tracker)
        
        if pts:
            norm_pts = normalize_landmarks(pts)
            S_t = frame_hv(norm_pts)
            
            if recording:
                gesture_frames.append(S_t)
            
            # Draw landmarks on the frame
            for (x, y, z) in pts:
                cx, cy = int(x * w), int(y * h)
                cv2.circle(frame, (cx, cy), 3, (0, 255, 0), -1)
        
        # Overlay UI text
        cv2.putText(frame, f"Mode: {mode}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Enroll label: {enroll_label}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Recording: {recording}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Last: {last_status}", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, f"Text: {text_buffer[-40:]}", (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        cv2.imshow("ST-HDC Air Writing", frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord("q"):
            break
        
        if key == ord("r"):
            recording = not recording
            if not recording:
                if gesture_frames:
                    G = encode_gesture(gesture_frames)
                    if mode == "enroll":
                        memory.enroll(enroll_label, G)
                        last_status = f"Enrolled '{enroll_label}'"
                        print(f"✓ Enrolled '{enroll_label}'")
                    else:
                        label, sim = memory.recognize(G)
                        if label is not None:
                            text_buffer += label
                            last_status = f"Recognized '{label}' ({sim:.2f})"
                            print(f"✓ Recognized '{label}' (similarity: {sim:.2f})")
                            speak(tts_engine, label)
                        else:
                            last_status = f"Unknown gesture ({sim:.2f})"
                            print(f"✗ Unknown gesture (similarity: {sim:.2f})")
                gesture_frames = []
        
        if key == ord("e"):
            mode = "enroll" if mode == "recognize" else "recognize"
            last_status = f"Mode switched to {mode}"
            print(f"Mode: {mode}")
        
        if key == ord("n"):
            if enroll_label == "Z":
                enroll_label = "A"
            else:
                enroll_label = chr(ord(enroll_label) + 1)
            last_status = f"Next enroll label: {enroll_label}"
            print(f"Enroll label: {enroll_label}")
    
    cap.release()
    cv2.destroyAllWindows()
    print("\nApplication closed. Goodbye!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Closing...")
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        cv2.destroyAllWindows()

