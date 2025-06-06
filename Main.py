
import cv2
import numpy as np
from flask import Flask, Response
import threading
from keras_facenet import FaceNet
from keras import models
from keras import utils
import time
import requests
import json
from datetime import datetime

# Initialize Flask app
app = Flask(__name__)

# Load the custom TripletLossLayer and the triplet model
from project.triplet_5 import TripletLossLayer
utils.get_custom_objects().update({'TripletLossLayer': TripletLossLayer})
triplet_model = models.load_model('triplet_model.keras', compile=False)

# Load stored embeddings, labels, and label names
stored_embeddings = np.load('embeddings.npy')
stored_labels = np.load('associated_labels.npy')  # This is assumed to be numeric IDs
label_names = np.load('label_names.npy')  # Mapping from numeric IDs to student names

# Create a mapping from numeric IDs to student names
id_to_name = {id: name for id, name in enumerate(label_names)}

# Load the pre-trained SSD model and its configuration
net = cv2.dnn.readNetFromCaffe(r'M:\fp\fpdemo\deploy.prototxt', r'M:\fp\fpdemo\res10_300x300_ssd_iter_140000.caffemodel')
embedder = FaceNet()

# Global variables
cap = None
recognition_running = False
video_stream_running = True
recognized_students = set()  # Store recognized students' names in a set


# Time-based 
start_time = "09:40"
end_time = "12:55" 

def is_genuine_face(face):
    gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    blurred_face = cv2.GaussianBlur(gray_face, (5, 5), 0)
    laplacian = cv2.Laplacian(blurred_face, cv2.CV_64F)
    variance = laplacian.var()
    threshold = 5
    print(f'Face Variance: {variance}')
    return variance > threshold



def find_closest_embedding(embedding, stored_embeddings):
    # Reshape to ensure compatibility
    embedding = embedding.flatten()

    # Compute similarities safely
    similarities = np.dot(stored_embeddings, embedding) / (
        np.linalg.norm(stored_embeddings, axis=1) * np.linalg.norm(embedding)
    )

    if len(similarities) == 0:
        return -1, 0  # Return a default value if no embeddings exist

    max_index = np.argmax(similarities)

    # Ensure the index is within bounds
    if max_index >= stored_embeddings.shape[0]:
        print(f"Error: max_index {max_index} out of bounds for stored embeddings size {stored_embeddings.shape[0]}")
        return -1, 0  # Return default values

    return max_index, similarities[max_index]



def is_within_recognition_time():
    """Check if the current time is within the start and end times."""
    current_time = datetime.now().strftime("%H:%M")
    return start_time <= current_time <= end_time

def run_face_recognition():
    global recognition_running, cap, recognized_students
    last_capture_time = time.time()
    capture_interval = 5

    while True:  # Continuous loop
        # Check if we are within the allowed recognition time
        if is_within_recognition_time():
            recognition_running = True
        else:
            recognition_running = False
            if recognized_students:
                send_recognized_students()  # Send data when recognition stops
            time.sleep(1)  # Sleep for a while before rechecking

        if not recognition_running or cap is None:
            time.sleep(1)  # Wait if recognition is not running or cap is not initialized
            continue

        ret, frame = cap.read()
        if not ret:
            break

        current_time = time.time()
        if (current_time - last_capture_time) >= capture_interval:
            last_capture_time = current_time
            (h, w) = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
            net.setInput(blob)
            detections = net.forward()

            for i in range(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.5:
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")
                    face = frame[startY:endY, startX:endX]

                    if face.size > 0:
                        face_resized = cv2.resize(face, (224, 224))

                        if is_genuine_face(face_resized):
                            face_embedding = embedder.embeddings([face_resized])[0].reshape(1, -1)

                            index, similarity = find_closest_embedding(face_embedding, stored_embeddings)

                            threshold = 0.7 #Adjust as needed
                            
                            if similarity > threshold:
                                predicted_label_id = stored_labels[index]
                                predicted_label = id_to_name.get(predicted_label_id, "Unknown")
                                recognized_students.add(predicted_label)  # Store recognized student's name
                            else:
                                predicted_label = "Unknown"
                                print("Unrecognized face detected")
                            
                            print(f'Predicted Label: {predicted_label}')

                            # Draw bounding box and label on the frame
                            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                            cv2.putText(frame, str(predicted_label), (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                        else:
                            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
                            cv2.putText(frame, "Spoof Detected", (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        # Optionally, display the frame for debugging purposes
        cv2.imshow('Face Recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            recognition_running = False
            cv2.destroyWindow('Face Recognition')



# Function to send recognized student names to the backend after recognition stops
def send_recognized_students():
    url = "http://192.168.32.173:8000/api/update_attendance/"
    headers = {'Content-Type': 'application/json'}
    global recognized_students
    try:
        if recognized_students:
            # Convert recognized students to the expected format
            data = [{"student_id": student} for student in recognized_students]
            print("Sending data:", data)  # Print data to verify format
            
            # Send the data to the backend
            response = requests.post(url, data=json.dumps(data), headers=headers)
            
            if response.status_code == 200:
                print("Recognized students data sent successfully.")
                # Reset the recognized_students set after sending
                recognized_students.clear()
            else:
                print(f"Failed to send data. Status code: {response.status_code}, Response: {response.text}")
    except Exception as e:
        print(f"Error sending data to backend: {e}")



def gen_frames():
    global cap, video_stream_running
    while video_stream_running:
        if cap is None:
            time.sleep(1)  # Wait if cap is not initialized
            continue
        
        success, frame = cap.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return "Real-Time Face Recognition with Video Feed"

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def video_stream_thread():
    app.run(host='0.0.0.0', port=5000, use_reloader=False)

def start_video_stream():
    global cap, video_stream_running
    cap = cv2.VideoCapture(0)
    video_stream_running = True
    video_thread = threading.Thread(target=video_stream_thread)
    video_thread.daemon = True
    video_thread.start()

if __name__ == '__main__':
    # Start video stream (Flask server) in a separate thread
    start_video_stream()

    # Start face recognition in a background thread
    recognition_thread = threading.Thread(target=run_face_recognition)
    recognition_thread.daemon = True
    recognition_thread.start()

    # Keep the main thread alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        # Stop recognition and send recognized students to backend
        recognition_running = False
        send_recognized_students()

        # Release video capture
        video_stream_running = False
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()


