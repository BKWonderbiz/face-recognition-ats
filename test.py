import cv2
import os
import numpy as np
import face_recognition
import sqlite3
import pickle
from datetime import datetime
import tkinter as tk
from tkinter import messagebox
from simple_facerec import SimpleFacerec

class FaceCapture:
    def __init__(self, db_path='face_encodings.db', images_path='images/'):
        self.db_path = db_path
        self.images_path = images_path
        self.create_db()

        self.root = tk.Tk()
        self.root.title("Face Capture")

        # Video Capture
        self.cap = cv2.VideoCapture(0)

        # UI Elements
        self.name_entry = tk.Entry(self.root)
        self.name_entry.pack(pady=10)
        self.capture_button = tk.Button(self.root, text="Capture Face", command=self.capture_face)
        self.capture_button.pack(pady=10)
        self.save_button = tk.Button(self.root, text="Save Encodings", command=self.save_encodings)
        self.save_button.pack(pady=10)
        self.detect_button = tk.Button(self.root, text="Detect Employees", command=self.detect_employees)
        self.detect_button.pack(pady=10)

        self.captured_images = []
        self.show_frame()

    def create_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS faces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE,
                encoding BLOB
            )
        ''')
        conn.commit()
        conn.close()

    def save_encoding_to_db(self, name, encoding):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Check if the name already exists
        cursor.execute('SELECT name FROM faces WHERE name = ?', (name,))
        result = cursor.fetchone()

        if result is None:
            cursor.execute('''
                INSERT INTO faces (name, encoding) VALUES (?, ?)
            ''', (name, pickle.dumps(encoding)))
            conn.commit()
            print(f"Encoding for {name} saved to the database.")
        else:
            print(f"Encoding for {name} already exists in the database.")
        
        conn.close()

    def capture_face(self):
        name = self.name_entry.get().strip()
        if not name:
            messagebox.showwarning("Input Error", "Name cannot be empty!")
            return

        person_dir = os.path.join(self.images_path, name)
        os.makedirs(person_dir, exist_ok=True)

        ret, frame = self.cap.read()
        if ret:
            # Save the captured image
            filename = f"{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
            img_path = os.path.join(person_dir, filename)
            cv2.imwrite(img_path, frame)
            print(f"Captured and saved {filename}")
            self.captured_images.append(img_path)
        else:
            messagebox.showerror("Capture Error", "Failed to capture image.")

    def save_encodings(self):
        name = self.name_entry.get().strip()
        if not name:
            messagebox.showwarning("Input Error", "Name cannot be empty!")
            return
        
        if self.captured_images:
            self.process_and_save_encodings(self.captured_images, name)
            messagebox.showinfo("Success", f"Encodings for {name} saved successfully!")
        else:
            messagebox.showwarning("No Images", "No images captured to save.")

    def process_and_save_encodings(self, img_paths, name):
        encodings = []

        for img_path in img_paths:
            img = cv2.imread(img_path)
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Get encoding
            if len(face_recognition.face_encodings(rgb_img)) > 0:
                img_encoding = face_recognition.face_encodings(rgb_img)[0]
                encodings.append(img_encoding)

        # Average the encodings
        if encodings:
            avg_encoding = np.mean(encodings, axis=0)

            # Save averaged encoding to the database
            self.save_encoding_to_db(name, avg_encoding)
        else:
            print("No encodings found for the captured images.")

    def detect_employees(self):
        # Close the main window
        self.root.destroy()

        # Initialize SimpleFacerec and load encodings from SQLite
        sfr = SimpleFacerec(db_path=self.db_path)

        # Open a new window for face detection
        detect_window = tk.Tk()
        detect_window.title("Detect Employees")

        # Load Camera for detection
        cap = cv2.VideoCapture(0)

        def show_detect_frame():
            ret, frame = cap.read()
            if ret:
                # Detect Faces
                face_locations, face_names = sfr.detect_known_faces(frame)
                for face_loc, name in zip(face_locations, face_names):
                    y1, x2, y2, x1 = face_loc
                    cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 4)

                cv2.imshow("Frame", frame)

            # Schedule the next frame
            detect_window.after(10, show_detect_frame)

        # Start showing the frames
        show_detect_frame()

        # Close resources on window close
        def on_closing():
            cap.release()
            cv2.destroyAllWindows()
            detect_window.destroy()
        detect_window.protocol("WM_DELETE_WINDOW", on_closing)

    def show_frame(self):
        ret, frame = self.cap.read()
        if ret:
            # Convert the image to RGB and then display it
            cv2.imshow("Video Feed", frame)
            self.root.after(10, self.show_frame)
        else:
            messagebox.showerror("Capture Error", "Failed to read from camera.")

    def quit(self):
        self.cap.release()
        cv2.destroyAllWindows()
        self.root.destroy()

if __name__ == "__main__":
    face_capture = FaceCapture()
    face_capture.root.mainloop()
