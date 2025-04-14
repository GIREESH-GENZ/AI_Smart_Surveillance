import os
import cv2
import numpy as np
import pandas as pd
import smtplib
from email.mime.text import MIMEText
from datetime import datetime

class FaceDetectionSystem:
    def __init__(self, student_folder, metadata_file, warden_email, camera_id):
    # Load Haar Cascade Classifier
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        self.student_folder = student_folder
        self.metadata_file = metadata_file
        self.warden_email = warden_email
        self.camera_id = camera_id

        # Load student metadata from the CSV file
        self.student_metadata = self._load_metadata()

        # Check if any student data is loaded
        if self.student_metadata.empty:
            print(f"Warning: No student data found in '{metadata_file}'.")
    def _load_metadata(self):
        if not os.path.exists(self.metadata_file):
            print(f"Error: Metadata file '{self.metadata_file}' does not exist.")
            return pd.DataFrame()  # Return an empty DataFrame

        self.student_metadata = pd.read_csv(self.metadata_file)
    
    # Clean column names
        self.student_metadata.columns = self.student_metadata.columns.str.strip()
    
    # Check if the necessary columns exist
        if 'Name' not in self.student_metadata.columns or 'Status' not in self.student_metadata.columns or 'Image' not in self.student_metadata.columns:
            print("Error: Metadata file is missing required columns ('Name', 'Status', 'Image').")
            return pd.DataFrame()  # Return an empty DataFrame if the required columns are missing

        return self.student_metadata

    def detect_faces(self, frame):
        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        detected_faces = []
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        for (x, y, w, h) in faces:
            # Extract face region
            face_roi = gray[y:y+h, x:x+w]

            # Identify student
            name = self._match_face(face_roi)
            status = self._get_student_status(name)

            detected_faces.append({
                'name': name,
                'status': status,
                'timestamp': current_time,
                'camera_id': self.camera_id
            })

        return detected_faces
    
    def _match_face(self, face_roi):
        """Match face using template matching."""
        best_match = "Unknown"
        best_score = float('inf')  # Lower score is better for template matching

        for student, image_file in zip(self.student_metadata['Name'], self.student_metadata['Image']):
            student_image_path = os.path.join(self.student_folder, image_file)
            print(f"Checking student image: {student_image_path}")

            if os.path.exists(student_image_path):
                known_face = cv2.imread(student_image_path, cv2.IMREAD_GRAYSCALE)
                known_face_resized = cv2.resize(known_face, (face_roi.shape[1], face_roi.shape[0]))

            # Perform template matching
                result = cv2.matchTemplate(face_roi, known_face_resized, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, _ = cv2.minMaxLoc(result)

                print(f"Template match score for {student}: {max_val}")

            # If the matching score is high enough, assign it as the best match
                if max_val > 0.8 and max_val > best_score:  # Adjust threshold as needed
                    best_score = max_val
                    best_match = student

        print(f"Best match: {best_match}")
        return best_match
    def _get_student_status(self, name):
        """Retrieve the status (Hosteller/Day Scholar) of a student from the metadata."""
        print(f"Getting status for: {name}")
        if name == "Unknown":
            return "Unknown"

        student_info = self.student_metadata[self.student_metadata['Name'] == name]
        if not student_info.empty:
            status = student_info.iloc[0]['Status']
            print(f"Status found: {status}")
            return status

        print("Status not found, returning Unknown.")
        return "Unknown"


    def send_email(self, detected_faces):
        if not detected_faces:
            return
        
        # Email configuration
        sender_email = "srikars779@gmail.com"
        sender_password = "reyundhgakhxtxkp"  # Replace with the correct app password
        
        # Compose email
        subject = "Non-Hosteller Detection Alert"
        body = "Persons Detected:\n"
        
        for face in detected_faces:
            body += f"Name: {face['name']}\n"
            body += f"Status: {face['status']}\n"
            body += f"Timestamp: {face['timestamp']}\n"
            body += f"Camera ID: {face['camera_id']}\n\n"
        
        try:
            msg = MIMEText(body)
            msg['Subject'] = subject
            msg['From'] = sender_email
            msg['To'] = self.warden_email
            
            with smtplib.SMTP('smtp.gmail.com', 587) as server:
                server.starttls()
                server.login(sender_email, sender_password)
                server.send_message(msg)
                print("Email sent successfully")
        except Exception as e:
            print(f"Email sending failed: {e}")

    def run(self):
        video_capture = cv2.VideoCapture(0)
        
        while True:
            ret, frame = video_capture.read()
            if not ret:
                print("Failed to read from camera. Exiting...")
                break
            
            detected_faces = self.detect_faces(frame)
            
            if detected_faces:
                print(f"Detected faces: {detected_faces}")
                self.send_email(detected_faces)
            
            # Display the frame
            for face in detected_faces:
                if face['name'] != "Unknown":
                    text = f"{face['name']} ({face['status']})"
                    cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow('Face Detection', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 27 is the keycode for Esc key
                print("Exiting program...")
                break
    
        video_capture.release()
        cv2.destroyAllWindows()

# Example usage

if __name__ == "__main__":  # Fixed __name__ syntax
    face_system = FaceDetectionSystem(
        student_folder='C:\\Users\\pavan\\OneDrive\\Desktop\\Python\\student_folder',  # Fixed missing closing quote and escaped backslashes
        metadata_file='C:\\Users\\pavan\\OneDrive\\Desktop\\Python\\student_folder\\metadata.csv.txt',  # Escaped backslashes
        warden_email='jagujayanth6@gmail.com',
        camera_id='CAM_MAIN_ENTRANCE'
    )
    
    # Run the actual face detection system
    face_system.run()