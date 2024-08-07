import cv2
import os
import numpy as np
import face_recognition
import sqlite3
import pickle
from datetime import datetime
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk

class FaceCapture:
    def __init__(self, db_path='face_encodings.db', images_path='images/'):
        def create_db(db_path):
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS Faces (
                    Id INTEGER PRIMARY KEY AUTOINCREMENT,
                    Name TEXT UNIQUE,
                    Encoding BLOB
                )
            ''')
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS Attendance (
                    Id INTEGER PRIMARY KEY AUTOINCREMENT,
                    UserId INTEGER,
                    Timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (UserId) REFERENCES Faces (Id)
                )
            ''')
            conn.commit()
            conn.close()

        self.db_path = db_path
        self.images_path = images_path
        create_db(self.db_path)

        self.root = tk.Tk()
        self.root.title("Face Capture")

        self.root.geometry("1000x500")
        self.root.configure(bg='#2E2E2E')

        # Video Capture
        self.cap = cv2.VideoCapture(0)

        # GUI left part for video stream
        self.frame_left_camera = tk.Frame(self.root, bg='#2E2E2E')
        self.label = tk.Label(self.frame_left_camera, bg='#2E2E2E')
        self.label.pack(padx=10, pady=10)
        self.frame_left_camera.pack(side=tk.LEFT, padx=10, pady=10)

        # GUI right part for inputs and buttons
        self.frame_right_info = tk.Frame(self.root, bg='#2E2E2E')
        self.frame_right_info.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        # UI Elements
        self.name_label = tk.Label(self.frame_right_info, text="Enter Name:", font=('Helvetica', 12), bg='#2E2E2E', fg='#FFFFFF')
        self.name_label.pack(pady=10)
        self.name_entry = tk.Entry(self.frame_right_info, font=('Helvetica', 12), bg='#4A4A4A', fg='#FFFFFF')
        self.name_entry.pack(pady=10, ipadx=5, ipady=5)
        self.capture_button = tk.Button(self.frame_right_info, text="Capture Face", font=('Helvetica', 12), bg='#4A4A4A', fg='#FFFFFF', command=self.capture_face)
        self.capture_button.pack(pady=10, ipadx=5, ipady=5)
        self.save_button = tk.Button(self.frame_right_info, text="Save Encodings", font=('Helvetica', 12), bg='#4A4A4A', fg='#FFFFFF', command=self.save_encodings)
        self.save_button.pack(pady=10, ipadx=5, ipady=5)
        self.detect_button = tk.Button(self.frame_right_info, text="Detect Employees", font=('Helvetica', 12), bg='#4A4A4A', fg='#FFFFFF', command=self.detect_employees)
        self.detect_button.pack(pady=10, ipadx=5, ipady=5)

        self.captured_images = []
        self.show_frame()

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
        def process_and_save_encodings(img_paths, name, db_path):
            def save_encoding_to_db(db_path, name, encoding):
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                cursor.execute('SELECT Name FROM Faces WHERE Name = ?', (name,))
                result = cursor.fetchone()

                if result is None:
                    cursor.execute('''
                            INSERT INTO Faces (Name, Encoding) VALUES (?, ?)
                        ''', (name, pickle.dumps(encoding)))
                    conn.commit()
                    print(f"Encoding for {name} saved to the database.")
                else:
                    print(f"Encoding for {name} already exists in the database.")
                conn.close()

            encodings = []
            for img_path in img_paths:
                print(img_path)
                img = cv2.imread(img_path)
                rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                if len(face_recognition.face_encodings(rgb_img)) > 0:
                    img_encoding = face_recognition.face_encodings(rgb_img)[0]
                    encodings.append(img_encoding)
                    os.remove(img_path)
                    print(f"image deleted {img_path}")
            avg_encoding = np.mean(encodings, axis=0)
            save_encoding_to_db(db_path, name, avg_encoding)
        name = self.name_entry.get().strip()
        if not name:
            messagebox.showwarning("Input Error", "Name cannot be empty!")
            return

        if self.captured_images:
            process_and_save_encodings(self.captured_images, name, self.db_path)
            messagebox.showinfo("Success", f"Encodings for {name} saved successfully!")
        else:
            messagebox.showwarning("No Images", "No images captured to save.")

    def detect_employees(self):
        def load_encodings_from_db(db_path):
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute('SELECT Name, Encoding FROM faces')
            rows = cursor.fetchall()
            conn.close()
            return [row[0] for row in rows], [pickle.loads(row[1]) for row in rows]
        self.root.destroy()

        # Initialize SimpleFacerec and load encodings from SQLite
        known_face_names, known_face_encodings = load_encodings_from_db(self.db_path)
        cap = cv2.VideoCapture(0)
        attended_users = set()

        # Open a new window for face detection
        detect_window = tk.Tk()
        detect_window.title("Detect Employees")
        detect_window.geometry("1000x500")
        detect_window.configure(bg='#2E2E2E')

        # GUI frame for video stream
        frame_video = tk.Frame(detect_window, bg='#2E2E2E')
        frame_video.pack(fill=tk.BOTH, expand=True)
        label_video = tk.Label(frame_video, bg='#2E2E2E')
        label_video.pack()

        def show_detect_frame():
            def detect_known_faces(known_face_encodings,known_face_names,frame, frame_resizing,db_path,attended_users):
                def mark_attendance(db_path, name):
                    conn = sqlite3.connect(db_path)
                    cursor = conn.cursor()
                    cursor.execute('SELECT Id FROM Faces WHERE Name = ?', (name,))
                    result = cursor.fetchone()
                    if result:
                        cursor.execute('''
                                INSERT INTO Attendance (UserId) VALUES (?)
                            ''', (result[0],))
                        conn.commit()
                        print(f"Marked Attendance for {name}")
                    conn.close()

                small_frame = cv2.resize(frame, (0, 0), fx=frame_resizing, fy=frame_resizing)
                rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                face_locations = face_recognition.face_locations(rgb_small_frame)
                face_encodings = face_recognition.face_encodings(
                    rgb_small_frame, face_locations)
                face_names = []
                for face_encoding in face_encodings:
                    matches = face_recognition.compare_faces(
                        known_face_encodings, face_encoding)
                    name = "Unknown"

                    face_distances = face_recognition.face_distance(
                        known_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index] and face_distances[best_match_index] < 0.45:
                        name = known_face_names[best_match_index]
                        if face_distances[best_match_index] < 0.3 and name not in attended_users:
                            attended_users.add(name)
                            mark_attendance(db_path, name)
                    face_names.append(name)
                face_locations = np.array(face_locations)
                face_locations = face_locations / frame_resizing
                return face_locations.astype(int), face_names

            ret, frame = cap.read()
            if ret:
                face_locations, face_names = detect_known_faces(
                    known_face_encodings, known_face_names, frame, 0.25, self.db_path, attended_users)
                for face_loc, name in zip(face_locations, face_names):
                    y1, x2, y2, x1 = face_loc
                    cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 4)

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(rgb_frame)
                imgtk = ImageTk.PhotoImage(image=img)
                label_video.imgtk = imgtk
                label_video.configure(image=imgtk)

            detect_window.after(10, show_detect_frame)

        show_detect_frame()

        def on_closing():
            cap.release()
            detect_window.destroy()

        detect_window.protocol("WM_DELETE_WINDOW", on_closing)

    def show_frame(self):
        ret, frame = self.cap.read()
        if ret:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb_frame)
            imgtk = ImageTk.PhotoImage(image=img)
            self.label.imgtk = imgtk
            self.label.configure(image=imgtk)
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
