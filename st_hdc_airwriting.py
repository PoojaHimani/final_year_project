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
# Default Letter Templates
# ==============================
def create_default_letter_templates():
    """
    Create default letter templates based on typical air-writing patterns.
    These represent common letter shapes and trajectories.
    
    Note: These are generic templates. For best results, train your own
    letters using Training Mode (press 't' key) to match your writing style.
    """
    templates = {}
    
    # Define typical letter trajectories (simplified patterns)
    # Each letter is represented as a sequence of (x, y, z) positions over time
    letter_patterns = {
        'A': [(0.3, 0.8, 0.5), (0.5, 0.2, 0.5), (0.7, 0.8, 0.5), (0.4, 0.5, 0.5), (0.6, 0.5, 0.5)],
        'B': [(0.3, 0.2, 0.5), (0.3, 0.8, 0.5), (0.3, 0.5, 0.5), (0.6, 0.3, 0.5), (0.3, 0.5, 0.5), (0.6, 0.7, 0.5)],
        'C': [(0.7, 0.5, 0.5), (0.4, 0.3, 0.5), (0.4, 0.7, 0.5), (0.6, 0.5, 0.5)],
        'D': [(0.3, 0.2, 0.5), (0.3, 0.8, 0.5), (0.3, 0.5, 0.5), (0.6, 0.3, 0.5), (0.6, 0.7, 0.5), (0.3, 0.5, 0.5)],
        'E': [(0.3, 0.2, 0.5), (0.3, 0.8, 0.5), (0.3, 0.5, 0.5), (0.6, 0.5, 0.5), (0.3, 0.2, 0.5), (0.6, 0.2, 0.5), (0.3, 0.8, 0.5), (0.6, 0.8, 0.5)],
        'F': [(0.3, 0.2, 0.5), (0.3, 0.8, 0.5), (0.3, 0.5, 0.5), (0.6, 0.5, 0.5), (0.3, 0.2, 0.5), (0.6, 0.2, 0.5)],
        'G': [(0.7, 0.5, 0.5), (0.4, 0.3, 0.5), (0.4, 0.7, 0.5), (0.6, 0.5, 0.5), (0.6, 0.7, 0.5), (0.5, 0.7, 0.5)],
        'H': [(0.3, 0.2, 0.5), (0.3, 0.8, 0.5), (0.3, 0.5, 0.5), (0.7, 0.5, 0.5), (0.7, 0.2, 0.5), (0.7, 0.8, 0.5)],
        'I': [(0.5, 0.2, 0.5), (0.5, 0.8, 0.5)],
        'J': [(0.7, 0.2, 0.5), (0.7, 0.7, 0.5), (0.5, 0.8, 0.5), (0.4, 0.7, 0.5)],
        'K': [(0.3, 0.2, 0.5), (0.3, 0.8, 0.5), (0.3, 0.5, 0.5), (0.7, 0.2, 0.5), (0.3, 0.5, 0.5), (0.7, 0.8, 0.5)],
        'L': [(0.3, 0.2, 0.5), (0.3, 0.8, 0.5), (0.3, 0.8, 0.5), (0.7, 0.8, 0.5)],
        'M': [(0.3, 0.8, 0.5), (0.3, 0.2, 0.5), (0.5, 0.5, 0.5), (0.7, 0.2, 0.5), (0.7, 0.8, 0.5)],
        'N': [(0.3, 0.8, 0.5), (0.3, 0.2, 0.5), (0.7, 0.8, 0.5), (0.7, 0.2, 0.5)],
        'O': [(0.5, 0.3, 0.5), (0.7, 0.5, 0.5), (0.5, 0.7, 0.5), (0.3, 0.5, 0.5), (0.5, 0.3, 0.5)],
        'P': [(0.3, 0.2, 0.5), (0.3, 0.8, 0.5), (0.3, 0.5, 0.5), (0.6, 0.4, 0.5), (0.3, 0.5, 0.5)],
        'Q': [(0.5, 0.3, 0.5), (0.7, 0.5, 0.5), (0.5, 0.7, 0.5), (0.3, 0.5, 0.5), (0.5, 0.3, 0.5), (0.6, 0.7, 0.5)],
        'R': [(0.3, 0.2, 0.5), (0.3, 0.8, 0.5), (0.3, 0.5, 0.5), (0.6, 0.4, 0.5), (0.3, 0.5, 0.5), (0.6, 0.8, 0.5)],
        'S': [(0.6, 0.3, 0.5), (0.4, 0.3, 0.5), (0.4, 0.5, 0.5), (0.6, 0.5, 0.5), (0.6, 0.7, 0.5), (0.4, 0.7, 0.5)],
        'T': [(0.5, 0.2, 0.5), (0.5, 0.8, 0.5), (0.3, 0.2, 0.5), (0.7, 0.2, 0.5)],
        'U': [(0.3, 0.2, 0.5), (0.3, 0.7, 0.5), (0.5, 0.8, 0.5), (0.7, 0.7, 0.5), (0.7, 0.2, 0.5)],
        'V': [(0.3, 0.2, 0.5), (0.5, 0.8, 0.5), (0.7, 0.2, 0.5)],
        'W': [(0.3, 0.2, 0.5), (0.3, 0.8, 0.5), (0.5, 0.6, 0.5), (0.7, 0.8, 0.5), (0.7, 0.2, 0.5)],
        'X': [(0.3, 0.2, 0.5), (0.7, 0.8, 0.5), (0.5, 0.5, 0.5), (0.7, 0.2, 0.5), (0.3, 0.8, 0.5)],
        'Y': [(0.3, 0.2, 0.5), (0.5, 0.5, 0.5), (0.7, 0.2, 0.5), (0.5, 0.5, 0.5), (0.5, 0.8, 0.5)],
        'Z': [(0.3, 0.2, 0.5), (0.7, 0.2, 0.5), (0.3, 0.8, 0.5), (0.7, 0.8, 0.5)],
        # Lowercase letters (different shapes)
        'a': [(0.5, 0.6, 0.5), (0.7, 0.5, 0.5), (0.5, 0.7, 0.5), (0.3, 0.5, 0.5), (0.5, 0.6, 0.5), (0.5, 0.8, 0.5)],
        'b': [(0.3, 0.2, 0.5), (0.3, 0.8, 0.5), (0.3, 0.6, 0.5), (0.6, 0.5, 0.5), (0.3, 0.6, 0.5)],
        'c': [(0.6, 0.5, 0.5), (0.4, 0.4, 0.5), (0.4, 0.6, 0.5), (0.5, 0.5, 0.5)],
        'd': [(0.7, 0.2, 0.5), (0.7, 0.8, 0.5), (0.7, 0.6, 0.5), (0.4, 0.5, 0.5), (0.7, 0.6, 0.5)],
        'e': [(0.3, 0.5, 0.5), (0.6, 0.5, 0.5), (0.3, 0.4, 0.5), (0.5, 0.4, 0.5), (0.3, 0.5, 0.5), (0.3, 0.6, 0.5), (0.5, 0.6, 0.5)],
        'f': [(0.5, 0.2, 0.5), (0.5, 0.8, 0.5), (0.3, 0.5, 0.5), (0.6, 0.5, 0.5), (0.3, 0.2, 0.5), (0.6, 0.2, 0.5)],
        'g': [(0.5, 0.6, 0.5), (0.7, 0.5, 0.5), (0.5, 0.7, 0.5), (0.3, 0.5, 0.5), (0.5, 0.6, 0.5), (0.5, 0.8, 0.5), (0.4, 0.9, 0.5)],
        'h': [(0.3, 0.2, 0.5), (0.3, 0.8, 0.5), (0.3, 0.6, 0.5), (0.7, 0.6, 0.5), (0.7, 0.8, 0.5)],
        'i': [(0.5, 0.2, 0.5), (0.5, 0.7, 0.5), (0.5, 0.2, 0.5)],
        'j': [(0.7, 0.2, 0.5), (0.7, 0.7, 0.5), (0.5, 0.8, 0.5), (0.4, 0.7, 0.5)],
        'k': [(0.3, 0.2, 0.5), (0.3, 0.8, 0.5), (0.3, 0.6, 0.5), (0.7, 0.4, 0.5), (0.3, 0.6, 0.5), (0.7, 0.8, 0.5)],
        'l': [(0.3, 0.2, 0.5), (0.3, 0.8, 0.5), (0.3, 0.8, 0.5), (0.5, 0.8, 0.5)],
        'm': [(0.3, 0.6, 0.5), (0.3, 0.8, 0.5), (0.4, 0.5, 0.5), (0.6, 0.5, 0.5), (0.7, 0.6, 0.5), (0.7, 0.8, 0.5)],
        'n': [(0.3, 0.6, 0.5), (0.3, 0.8, 0.5), (0.5, 0.5, 0.5), (0.7, 0.6, 0.5), (0.7, 0.8, 0.5)],
        'o': [(0.5, 0.5, 0.5), (0.6, 0.4, 0.5), (0.5, 0.6, 0.5), (0.4, 0.4, 0.5), (0.5, 0.5, 0.5)],
        'p': [(0.3, 0.2, 0.5), (0.3, 0.8, 0.5), (0.3, 0.6, 0.5), (0.6, 0.5, 0.5), (0.3, 0.6, 0.5)],
        'q': [(0.5, 0.5, 0.5), (0.6, 0.4, 0.5), (0.5, 0.6, 0.5), (0.4, 0.4, 0.5), (0.5, 0.5, 0.5), (0.5, 0.8, 0.5), (0.4, 0.9, 0.5)],
        'r': [(0.3, 0.2, 0.5), (0.3, 0.8, 0.5), (0.3, 0.6, 0.5), (0.6, 0.5, 0.5), (0.3, 0.6, 0.5), (0.5, 0.7, 0.5)],
        's': [(0.5, 0.4, 0.5), (0.4, 0.4, 0.5), (0.4, 0.5, 0.5), (0.5, 0.5, 0.5), (0.5, 0.6, 0.5), (0.4, 0.6, 0.5)],
        't': [(0.5, 0.2, 0.5), (0.5, 0.8, 0.5), (0.3, 0.2, 0.5), (0.7, 0.2, 0.5)],
        'u': [(0.3, 0.6, 0.5), (0.3, 0.8, 0.5), (0.5, 0.8, 0.5), (0.7, 0.8, 0.5), (0.7, 0.6, 0.5)],
        'v': [(0.3, 0.6, 0.5), (0.5, 0.8, 0.5), (0.7, 0.6, 0.5)],
        'w': [(0.3, 0.6, 0.5), (0.3, 0.8, 0.5), (0.5, 0.7, 0.5), (0.7, 0.8, 0.5), (0.7, 0.6, 0.5)],
        'x': [(0.3, 0.6, 0.5), (0.7, 0.8, 0.5), (0.5, 0.7, 0.5), (0.7, 0.6, 0.5), (0.3, 0.8, 0.5)],
        'y': [(0.3, 0.6, 0.5), (0.5, 0.7, 0.5), (0.7, 0.6, 0.5), (0.5, 0.7, 0.5), (0.5, 0.9, 0.5)],
        'z': [(0.3, 0.6, 0.5), (0.7, 0.6, 0.5), (0.3, 0.8, 0.5), (0.7, 0.8, 0.5)],
    }
    
    # Convert patterns to hypervectors
    for letter, pattern in letter_patterns.items():
        # Create frame hypervectors for each point in the pattern
        frame_hvs = []
        for point in pattern:
            # Create a normalized landmark set (simplified - using index finger tip as primary)
            # In real usage, we'd have 21 landmarks, but for templates we simulate
            landmarks = []
            # Use the point as the index finger tip (landmark 8), and create relative positions
            base_x, base_y, base_z = point
            for i in range(21):
                # Create landmarks relative to the base point with some variation
                offset = 0.1 * (i / 21.0)  # Small offset for each landmark
                x = max(0.0, min(1.0, base_x + (i % 3 - 1) * offset))
                y = max(0.0, min(1.0, base_y + ((i // 3) % 3 - 1) * offset))
                z = max(0.0, min(1.0, base_z + ((i // 9) % 3 - 1) * offset * 0.5))
                landmarks.append((x, y, z))
            
            # Normalize landmarks
            norm_landmarks = normalize_landmarks(landmarks)
            # Create frame hypervector
            S_t = frame_hv(norm_landmarks)
            frame_hvs.append(S_t)
        
        # Encode the full gesture
        G = encode_gesture(frame_hvs)
        if G is not None:
            templates[letter] = G
    
    return templates

# ==============================
# Associative Memory (One-shot)
# ==============================
class OneShotMemory:
    def __init__(self):
        self.store = {}  # label -> hypervector
        self.model_file = "trained_letters.npz"

    def enroll(self, label, hv):
        """Store / overwrite the hypervector for a label."""
        if hv is None:
            return
        self.store[label] = hv

    def recognize(self, hv, threshold=0.03):
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
    
    def save(self):
        """Save trained models to file."""
        try:
            np.savez(self.model_file, **{k: v for k, v in self.store.items()})
            return True
        except Exception as e:
            print(f"Error saving model: {e}")
            return False
    
    def load(self):
        """Load trained models from file."""
        try:
            if os.path.exists(self.model_file):
                data = np.load(self.model_file)
                self.store = {k: data[k] for k in data.keys()}
                return True
        except Exception as e:
            print(f"Error loading model: {e}")
        return False
    
    def load_defaults(self):
        """Load default letter templates."""
        print("Loading default letter templates...")
        defaults = create_default_letter_templates()
        self.store.update(defaults)
        uppercase = sum(1 for k in defaults.keys() if k.isupper())
        lowercase = sum(1 for k in defaults.keys() if k.islower())
        print(f"✓ Loaded {len(defaults)} default letter templates ({uppercase} uppercase + {lowercase} lowercase)")
        return len(defaults) > 0
    
    def get_trained_letters(self):
        """Get list of trained letters."""
        return sorted(list(self.store.keys()))
    
    def clear(self):
        """Clear all trained models."""
        self.store = {}

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
    
    # Try to load previously trained models
    if memory.load():
        trained = memory.get_trained_letters()
        if trained:
            print(f"✓ Loaded {len(trained)} trained letters from file: {', '.join(trained)}")
        else:
            # Load default templates if file exists but is empty
            memory.load_defaults()
    else:
        # No saved file - load default templates
        print("No saved models found. Loading default letter templates...")
        memory.load_defaults()
        print("✓ Default templates loaded! You can start recognizing letters immediately.")
        print("  (You can still train your own letters with 't' key to improve accuracy)")
    
    recording = False
    gesture_frames = []
    mode = "recognize"   # or "enroll" or "train"
    enroll_label = "A"
    text_buffer = ""
    last_status = "Ready - Press 'r' to start"
    training_mode = False  # Special training mode for all letters
    
    print("\n" + "=" * 50)
    print("Controls:")
    print("  r : start/stop recording a gesture")
    print("  e : toggle enroll/recognize mode")
    print("  n : next label for enrollment (A-Z, a-z)")
    print("  t : TRAINING MODE - train all letters A-Z and a-z automatically")
    print("  s : save trained models to file")
    print("  q : quit")
    print("=" * 50)
    print("\nTRAINING MODE (Recommended):")
    print("  1. Press 't' to enter training mode")
    print("  2. Train each letter A-Z, then a-z (auto-advances)")
    print("  3. Press 's' to save when done")
    print("  4. Models auto-load on next run!")
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
        trained_count = len(memory.store)
        cv2.putText(frame, f"Mode: {mode}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Enroll label: {enroll_label}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Trained: {trained_count}/26 letters", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
        cv2.putText(frame, f"Recording: {recording}", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Last: {last_status}", (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, f"Text: {text_buffer[-40:]}", (10, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Add instructions
        if mode == "enroll":
            if training_mode:
                cv2.putText(frame, "TRAINING MODE: Training all letters A-Z, then a-z", 
                           (10, h - 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.putText(frame, f"Current: {enroll_label} - Press 'r' to record, write letter, press 'r' again", 
                           (10, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(frame, "Press 's' to save all trained letters, 't' to exit training mode", 
                           (10, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            else:
                cv2.putText(frame, "INSTRUCTIONS: Press 'r' to record, write letter in air, press 'r' again", 
                           (10, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(frame, "Press 't' for TRAINING MODE (train all letters), 's' to save", 
                           (10, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        else:
            if not memory.store:
                cv2.putText(frame, "WARNING: No letters trained! Press 't' for training mode or 'e' to enroll", 
                           (10, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.putText(frame, "INSTRUCTIONS: Press 'r' to record, write letter, press 'r' again", 
                       (10, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
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
                        # Auto-save after each enrollment
                        if memory.save():
                            print(f"  (Auto-saved)")
                        # Auto-advance in training mode
                        if training_mode:
                            current_letter = enroll_label
                            if enroll_label == "Z":
                                enroll_label = "a"  # Switch to lowercase after Z
                                last_status = f"Enrolled '{current_letter}'. Next: {enroll_label} (lowercase)"
                            elif enroll_label == "z":
                                enroll_label = "A"  # Cycle back to uppercase
                                last_status = "Training complete! All letters (A-Z, a-z) trained. Press 's' to save."
                            else:
                                enroll_label = chr(ord(enroll_label) + 1)
                                last_status = f"Enrolled '{current_letter}'. Next: {enroll_label}"
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
                enroll_label = "a"  # Switch to lowercase after Z
            elif enroll_label == "z":
                enroll_label = "A"  # Cycle back to uppercase
            else:
                enroll_label = chr(ord(enroll_label) + 1)
            last_status = f"Next enroll label: {enroll_label}"
            print(f"Enroll label: {enroll_label}")
        
        if key == ord("t"):
            # Toggle training mode
            if mode == "enroll":
                training_mode = not training_mode
                if training_mode:
                    enroll_label = "A"
                    last_status = "TRAINING MODE: Start with letter A. Press 'r' to record each letter."
                    print("=" * 50)
                    print("TRAINING MODE ACTIVATED")
                    print("Train all letters A-Z, then a-z. Auto-advances after each letter.")
                    print("Press 's' to save, 't' again to exit training mode.")
                    print("=" * 50)
                else:
                    last_status = "Training mode deactivated"
                    print("Training mode deactivated")
            else:
                # Switch to enroll mode and activate training
                mode = "enroll"
                training_mode = True
                enroll_label = "A"
                last_status = "TRAINING MODE: Start with letter A"
                print("=" * 50)
                print("TRAINING MODE ACTIVATED")
                print("Train all letters A-Z, then a-z. Auto-advances after each letter.")
                print("=" * 50)
        
        if key == ord("s"):
            # Save trained models
            if memory.save():
                trained = memory.get_trained_letters()
                last_status = f"Saved {len(trained)} letters: {', '.join(trained)}"
                print(f"✓ Saved {len(trained)} trained letters to {memory.model_file}")
            else:
                last_status = "Error saving models"
                print("✗ Error saving models")
    
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

