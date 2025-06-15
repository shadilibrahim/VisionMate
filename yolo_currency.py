import cv2
import pyttsx3
from queue import Queue
from threading import Thread
import time

# Import the InferencePipeline object
from inference import InferencePipeline

# Initialize the text-to-speech engine
engine = pyttsx3.init()

# Queue for TTS messages
tts_queue = Queue()

def speak_text(text):
    engine.say(text)
    engine.runAndWait()

def tts_worker():
    while True:
        text = tts_queue.get()
        if text == "STOP":
            break
        speak_text(text)

# Start TTS thread
tts_thread = Thread(target=tts_worker, daemon=True)
tts_thread.start()

# Variable to track the last detection time
last_detection_time = 0

# Callback function to handle predictions
def my_sink(result, video_frame):
    global last_detection_time

    # Get the current time
    current_time = time.time()

    # Check if 2 seconds have passed since the last detection
    if current_time - last_detection_time < 2:
        return  # Skip processing if less than 2 seconds have passed

    if result.get("output_image"):  # Display an image from the workflow response
        cv2.imshow("Workflow Image", result["output_image"].numpy_image)
        cv2.waitKey(1)

    predictions = result.get("predictions", [])
    for prediction in predictions:
        label = prediction.get("class", "Unknown")
        confidence = prediction.get("confidence", 0)

        # Only process if the detected object is a currency note
        if "currency" in label.lower():  # Adjust this condition based on your model's output
            detection_info = f"Detected: {label} ({confidence:.2f})"
            print(detection_info)
            tts_queue.put(detection_info)
            tts_queue.put(f"Announcement: {label} detected with {confidence * 100:.1f} percent confidence")  # Audio feedback

            # Update the last detection time
            last_detection_time = current_time

# Initialize the InferencePipeline object
pipeline = InferencePipeline.init_with_workflow(
    api_key="9p0CEWcQsJC7euMxImuU",
    workspace_name="shadil-ibrahim-2klvc",
    workflow_id="detect-count-and-visualize",
    video_reference=0,  # Path to video, device ID (int, usually 0 for built-in webcams), or RTSP stream URL
    max_fps=30,
    on_prediction=my_sink
)

# Start the pipeline
pipeline.start()

# Wait for the pipeline thread to finish
pipeline.join()

tts_queue.put("STOP")
tts_thread.join()