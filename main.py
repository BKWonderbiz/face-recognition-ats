import cv2
import os
import numpy as np
import face_recognition
import pymssql
import pickle
from datetime import datetime
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import random
import time
import config

class FaceCapture:
    def __init__(self, server, user, password, database, images_path='images/'):
        self.server = server
        self.user = user
        self.password = password
        self.database = database
        self.images_path = images_path
        self.conn = pymssql.connect(server, user, password, database)
        
        self.root = tk.Tk()
        self.root.title("Employee Verification")

        self.root.geometry("400x200")
        self.root.configure(bg='#2E2E2E')

        # UI Elements for Employee ID Input
        self.id_label = tk.Label(self.root, text="Enter Employee ID:", font=('Helvetica', 12), bg='#2E2E2E', fg='#FFFFFF')
        self.id_label.pack(pady=10)
        self.id_entry = tk.Entry(self.root, font=('Helvetica', 12), bg='#4A4A4A', fg='#FFFFFF')
        self.id_entry.pack(pady=10, ipadx=5, ipady=5)
        self.check_button = tk.Button(self.root, text="Check Employee", font=('Helvetica', 12), bg='#4A4A4A', fg='#FFFFFF', command=self.check_employee)
        self.check_button.pack(pady=10, ipadx=5, ipady=5)

    def check_employee(self):
        employee_id = self.id_entry.get().strip()
        if not employee_id:
            messagebox.showwarning("Input Error", "Employee ID cannot be empty!")
            return

        cursor = self.conn.cursor()
        cursor.execute('SELECT Id FROM EmployeeDetails WHERE UserId = %s', (employee_id,))
        result = cursor.fetchone()
        cursor.execute('SELECT FirstName FROM EmployeeDetails WHERE UserId = %s', (employee_id,))
        EmployeeName = cursor.fetchone()

        if result:
            self.root.destroy()
            self.start_capture_window(employee_id, EmployeeName[0])
        else:
            messagebox.showerror("Error", "Employee ID not found!")

    def start_capture_window(self,employee_id, EmployeeName):
        self.root = tk.Tk()
        self.root.title("Face Capture")

        self.root.geometry("1000x500")
        self.root.configure(bg='#2E2E2E')
        self.employee_id = employee_id

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
        self.name_label = tk.Label(self.frame_right_info, text=f"Welcome, {EmployeeName}", font=('Helvetica', 12), bg='#2E2E2E', fg='#FFFFFF')
        self.name_label.pack(pady=10)
        
        # Fixing the command binding issue
        self.capture_button = tk.Button(self.frame_right_info, text="Capture Face", font=('Helvetica', 12), bg='#4A4A4A', fg='#FFFFFF', command=lambda: self.capture_face(employee_id))
        self.capture_button.pack(pady=10, ipadx=5, ipady=5)
        
        self.save_button = tk.Button(self.frame_right_info, text="Save Encodings", font=('Helvetica', 12), bg='#4A4A4A', fg='#FFFFFF', command=lambda: self.save_encodings(employee_id))
        self.save_button.pack(pady=10, ipadx=5, ipady=5)
        
        self.detect_button = tk.Button(self.frame_right_info, text="Detect Employees", font=('Helvetica', 12), bg='#4A4A4A', fg='#FFFFFF', command=self.detect_employees)
        self.detect_button.pack(pady=10, ipadx=5, ipady=5)

        self.captured_images = []
        self.show_frame()
      
    def capture_face(self, employee_id):
        self.employee_id = employee_id

        person_dir = os.path.join(self.images_path, str(employee_id))
        os.makedirs(person_dir, exist_ok=True)

        ret, frame = self.cap.read()
        if ret:
            # Save the captured image
            filename = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{random.randint(1,9999)}.jpg"
            img_path = os.path.join(person_dir, filename)
            cv2.imwrite(img_path, frame)
            print(f"Captured and saved {filename}")
            self.captured_images.append(img_path)
            # i=i+1
        else:
            messagebox.showerror("Capture Error", "Failed to capture image.")

    def save_encodings(self, employee_id):
        self.employee_id = employee_id

        def check_encodings_exist(conn, employee_id):
            cursor = conn.cursor()
            cursor.execute('SELECT FaceEncoding FROM EmployeeDetails WHERE UserId = %s', (employee_id,))
            result = cursor.fetchone()
            return result is not None and result[0] is not None

        def process_and_save_encodings(img_paths, employee_id, conn):
            def save_encoding_to_db(conn, employee_id, encoding):
                cursor = conn.cursor()
                cursor.execute('''
                    UPDATE EmployeeDetails
                    SET FaceEncoding = %s
                    WHERE UserId = %s
                ''', (pickle.dumps(encoding), employee_id))
                conn.commit()
                print(f"Encoding for {employee_id} updated in the database.")

            encodings = []
            for img_path in img_paths:
                print(img_path)
                img = cv2.imread(img_path)
                if img is not None:  # Ensure the image is loaded correctly
                    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    if len(face_recognition.face_encodings(rgb_img)) > 0:
                        img_encoding = face_recognition.face_encodings(rgb_img)[0]
                        encodings.append(img_encoding)
                        os.remove(img_path)
                        print(f"Image deleted {img_path}")
                else:
                    print(f"Failed to load image: {img_path}")
            if encodings:
                avg_encoding = np.mean(encodings, axis=0)
                save_encoding_to_db(conn, employee_id, avg_encoding)

        if self.captured_images:
            process_and_save_encodings(self.captured_images, self.employee_id, self.conn)
            messagebox.showinfo("Success", f"Encodings for {employee_id} updated successfully!")
        else:
            messagebox.showwarning("No Images", "No images captured to save.")

    def detect_employees(self):
        # self.employee_id = employee_id

        def load_encodings_from_db(conn):
            cursor = conn.cursor()
            cursor.execute('SELECT FirstName, FaceEncoding FROM EmployeeDetails')
            rows = cursor.fetchall()
            return [row[0] for row in rows], [row[0] for row in rows], [pickle.loads(row[1]) for row in rows]

        self.root.destroy()

        # Initialize SimpleFacerec and load encodings from MS SQL
        known_face_id, known_face_names, known_face_encodings = load_encodings_from_db(self.conn)
        cap = cv2.VideoCapture(0)
        last_attendance_time = {}

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
            def detect_known_faces(known_face_encodings, known_face_names, frame, frame_resizing, conn, attended_users):
                def mark_attendance(conn, name):
                    cursor = conn.cursor()
                    cursor.execute('SELECT UserId FROM EmployeeDetails WHERE FirstName = %s', (name,))
                    result = cursor.fetchone()
                    if result:
                        user_id = result[0]
                        attendance_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')  # Current time in the desired format
                        cursor.execute('''
                            INSERT INTO AttendanceLogs (UserId, AttendanceLogTime, CheckType) 
                            VALUES (%s, %s, 'IN')
                        ''', (user_id, attendance_time))
                        conn.commit()
                        print(f"Marked Attendance for {name}")

                small_frame = cv2.resize(frame, (0, 0), fx=frame_resizing, fy=frame_resizing)
                rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                face_locations = face_recognition.face_locations(rgb_small_frame)
                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
                face_names = []
                current_time = time.time()
                for face_encoding in face_encodings:
                    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                    name = "Unknown"

                    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index] and face_distances[best_match_index] < 0.45:
                        name = known_face_names[best_match_index]
                        if face_distances[best_match_index] < 0.3:
                            if name not in last_attendance_time or (current_time - last_attendance_time[name]) > config.waitTime:
                                last_attendance_time[name] = current_time
                                mark_attendance(conn, name)
                    face_names.append(name)
                face_locations = np.array(face_locations)
                face_locations = face_locations / frame_resizing
                return face_locations.astype(int), face_names

            ret, frame = cap.read()
            if ret:
                face_locations, face_names = detect_known_faces(known_face_encodings, known_face_names, frame, 0.25, self.conn, attended_users)
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
    face_capture = FaceCapture(server='DESKTOP-KUTTPQR', user='sa', password='User123', database='AttendanceTrackingSystem')
    face_capture.root.mainloop()
