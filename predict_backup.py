import cv2
import joblib
from utils import extract_hand_landmarks
import numpy as np

MODEL_PATH = "svm_model2.pkl"
# TEST_IMAGE = r"C:\Users\rohan\Python\sign_recog\data\test_sign.jpg"  # change to your test image path
TEST_IMAGE = (
    r"C:\Users\rohan\Python\sign_recog\data\testA.jpg"  # change to your test image path
)

# Load trained model
clf = joblib.load(MODEL_PATH)


# Load test image
img = cv2.imread(TEST_IMAGE)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (256, 256))
if img is None:
    raise ValueError(f"Could not read {TEST_IMAGE}")

# load image from webcam
#cap = cv2.VideoCapture(0)  # 0 = default camera

# while True:
#     ret, img = cap.read()
#     if not ret:
#         print("Failed to grab frame")
#         break

features = extract_hand_landmarks(img)
if features is None:
    print("No hand detected in the image.")
else:
    # Get probabilities
    probs = clf.predict_proba([features])[0]
    max_prob = np.max(probs)
    pred_class = clf.classes_[np.argmax(probs)]
    print(f"maz probability: {max_prob}")
    # Apply threshold
    threshold = 0.6  # you can tune this (0.5–0.8 typical)

    if max_prob < threshold:
        print("Prediction uncertain → Unknown gesture")
        label_text = "Unknown"
    else:
        #prediction = clf.predict([features])[0]
        #print("Predicted sign:", prediction)
        print(f"Predicted sign: {pred_class} (confidence {max_prob:.2f})")
        label_text = f"{pred_class} ({max_prob:.2f})"

    # Draw result on image
    cv2.putText(img, label_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Prediction", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
