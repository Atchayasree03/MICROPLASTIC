# realtime_test.py
import torch
from ultralytics import YOLO
import cv2

# ========================================================
# ✅ FIX FOR PYTORCH 2.6 LOADING ERROR
# ========================================================
_original_load = torch.load

def unsafe_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _original_load(*args, **kwargs)

torch.load = unsafe_load

# ========================================================
# ✅ CONFIGURATION
# ========================================================
model_path = "/home/rithish/runs/detect/train3/weights/best.pt"   # Change this
source_type = 2  # 0 = default webcam, or give path to video file
confidence_threshold = 0.5
# ========================================================

# Load YOLO model
print("⏳ Loading model...")
model = YOLO(model_path)
print("✅ Model loaded successfully!")

# Initialize video source
cap = cv2.VideoCapture(source_type)

if not cap.isOpened():
    print("❌ Error: Could not open video source.")
    exit()

print("🎥 Real-time detection started... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ End of video or camera error.")
        break

    # Run YOLO inference
    results = model.predict(frame, conf=confidence_threshold, verbose=False)

    # Draw predictions
    annotated_frame = results[0].plot()

    # Display output
    cv2.imshow("YOLO Real-Time Detection", annotated_frame)

    # Quit with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
print("✅ Inference stopped. Window closed.")

