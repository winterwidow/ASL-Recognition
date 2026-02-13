import cv2
import joblib
from utils import extract_hand_landmarks
from collections import deque
import time

MODEL_PATH = "svm_model.pkl"
# TEST_IMAGE = r"C:\Users\rohan\Python\sign_recog\data\test_sign.jpg"  # change to your test image path
TEST_IMAGE = (
    r"C:\Users\rohan\Python\sign_recog\data\test.jpg"  # change to your test image path
)

# Load trained model
clf = joblib.load(MODEL_PATH)

prediction_buffer = deque(maxlen=10)  # keep a queue of 10 predictions

last_prediction = None  # keep the last valid prediction till new prediction is possible

print("trying to access webcam")
cap = cv2.VideoCapture(0)  # 0 = default camera
print("webcam activated")

while True:
    print("enter image")
    time.sleep(5)
    ret, img = cap.read()

    if not ret:
        print("Failed to grab frame")
        break

    print("frame captured")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (256, 256))
    features = extract_hand_landmarks(img)

    if features is None:
        print("No hand detected in the image.")
    else:
        prediction = clf.predict([features])[0]
        prediction_buffer.append(prediction)
        last_prediction = prediction

        # Majority vote smoothing
        stable_pred = max(set(prediction_buffer), key=prediction_buffer.count)
        cv2.putText(
            img, stable_pred, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
        )

        print("Predicted sign:", prediction)

        # Draw result on image
        cv2.putText(
            img, prediction, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
        )
    if last_prediction:
        cv2.putText(
            img, prediction, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
        )
    cv2.imshow("Prediction", img)
    # v2.waitKey(0)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()
