import cv2
import numpy as np
import tensorflow as tf
import pyttsx3

# Load the trained currency classification model
model = tf.keras.models.load_model("currency_classifier.h5")
print("Model loaded successfully!")

# Print model input shape for debugging
print("Expected model input shape:", model.input_shape)

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Define class labels (Update based on your model's training labels)
class_labels = {
    0: "10 Rupees", 1: "20 Rupees", 2: "50 Rupees", 
    3: "100 Rupees", 4: "500 Rupees", 5: "2000 Rupees"
}

def preprocess_image(frame):
    """ Preprocess the image to match the model's expected input shape """
    img = cv2.resize(frame, (224, 224))  # Resize to model input size
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB (if needed)
    img = img / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    # If the model expects a flattened input, reshape it
    if model.input_shape[-1] == 6272:  # Check if flattening is required
        img = img.reshape(1, -1)

    print("Processed image shape:", img.shape)  # Debugging
    return img

def announce_currency(label):
    """ Announce detected currency via audio feedback """
    engine.say(f"Detected {label}")
    engine.runAndWait()

# Open camera
cap = cv2.VideoCapture(0)

# Keep track of last detected currency to avoid repeating announcements
last_detected_label = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess frame and predict
    img = preprocess_image(frame)
    predictions = model.predict(img)
    class_index = np.argmax(predictions)
    confidence = predictions[0][class_index]

    if confidence > 0.7:  # Confidence threshold
        label = class_labels.get(class_index, "Unknown Currency")

        # Draw bounding box and label
        cv2.rectangle(frame, (50, 50), (400, 400), (0, 255, 0), 2)
        cv2.putText(frame, label, (50, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Announce currency (only if different from last detected)
        if label != last_detected_label:
            announce_currency(label)
            last_detected_label = label

    # Show frame
    cv2.imshow("Currency Detection", frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
