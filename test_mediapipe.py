"""Quick test to check MediaPipe imports and usage"""
import sys

try:
    import cv2
    print("✓ cv2 imported")
except Exception as e:
    print(f"✗ cv2 error: {e}")
    sys.exit(1)

try:
    import mediapipe as mp
    print("✓ mediapipe imported")
    print(f"  Version: {getattr(mp, '__version__', 'unknown')}")
except Exception as e:
    print(f"✗ mediapipe error: {e}")
    sys.exit(1)

try:
    from mediapipe.tasks import python as mp_python
    print("✓ mediapipe.tasks.python imported")
except Exception as e:
    print(f"✗ mediapipe.tasks.python error: {e}")
    sys.exit(1)

try:
    from mediapipe.tasks.python import vision
    print("✓ mediapipe.tasks.python.vision imported")
except Exception as e:
    print(f"✗ vision error: {e}")
    sys.exit(1)

try:
    from mediapipe.tasks.python.vision.core import image as mp_image
    print("✓ mp_image imported")
    print(f"  ImageFormat available: {hasattr(mp_image, 'ImageFormat')}")
except Exception as e:
    print(f"✗ mp_image error: {e}")
    print("  Trying alternative import...")
    try:
        # Alternative import
        from mediapipe import Image, ImageFormat
        print("✓ Alternative import worked")
    except Exception as e2:
        print(f"✗ Alternative also failed: {e2}")

print("\nAll imports successful!")

