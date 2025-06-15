import cv2
import pyttsx3
import time
from queue import Queue
from threading import Thread

# Import the InferencePipeline object
from inference import InferencePipeline

# Initialize the text-to-speech engine inside the worker thread
def tts_worker():
    engine = pyttsx3.init()  # Initialize inside the thread for safety
    while True:
        text = tts_queue.get()
        if text == "STOP":
            break
        print(f"TTS speaking: {text}")  # Debug print to check if TTS is processing text
        engine.say(text)
        engine.runAndWait()
        time.sleep(0.5)  # Prevent overlapping speech

# Start TTS queue and worker thread
tts_queue = Queue()
tts_thread = Thread(target=tts_worker, daemon=True)
tts_thread.start()

# Keep track of last detected object to avoid looping
last_detected_label = None  

def my_sink(result, video_frame):
    global last_detected_label  # Access the global variable

    if result.get("output_image"):  
        cv2.imshow("Workflow Image", result["output_image"].numpy_image)
        cv2.waitKey(1)

    predictions = result.get("predictions", [])  # This is a `Detections` object
    print("Predictions received from API:", predictions)  # Debugging print

    if not hasattr(predictions, "xyxy") or len(predictions.xyxy) == 0:
        print("No valid predictions received. Skipping TTS.")
        last_detected_label = None  # Reset when no detection is found
        return

    for i in range(len(predictions.xyxy)):  
        label = predictions.data["class_name"][i]  # Extract class name
        confidence = float(predictions.confidence[i])  # Extract confidence score

        # Skip duplicate detections
        if label == last_detected_label:
            print(f"Skipping duplicate detection: {label}")
            continue

        last_detected_label = label  # Update last detected object

        # New natural announcement for currency
        detection_info = f"You are holding a {label} rupee note."
        print(detection_info)  # Debug print
        tts_queue.put(detection_info)  # Add to TTS queue
        print("Added to TTS queue")  

# Initialize the InferencePipeline object
pipeline = InferencePipeline.init_with_workflow(
    api_key="9p0CEWcQsJC7euMxImuU",
    workspace_name="shadil-ibrahim-2klvc",
    workflow_id="detect-count-and-visualize",
    video_reference=0,  # Webcam or video file
    max_fps=10,  # Reduce FPS to avoid overwhelming API and TTS
    on_prediction=my_sink
)

# Run the pipeline in a separate thread to prevent blocking
pipeline_thread = Thread(target=pipeline.start, daemon=True)
pipeline_thread.start()

# Add quit button functionality: listen for 'q' key to quit
while True:
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # If 'q' key is pressed
        print("Quit requested. Stopping.")
        break

# Stop the TTS worker thread gracefully
tts_queue.put("STOP")
tts_thread.join()

# Stop the pipeline
pipeline.stop()  # Assuming you have a stop method to clean up the pipeline
cv2.destroyAllWindows()
