import cv2
import os
import numpy as np
import face_recognition
import sqlite3
import pickle
from datetime import datetime

class FaceCapture:
    def __init__(self, db_path='face_encodings.db', images_path='images/'):
        self.db_path = db_path
        self.images_path = images_path
        self.create_db()

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

    def capture_faces(self):
        name = input("Enter the name of the person: ").strip()
        if not name:
            print("Name cannot be empty!")
            return

        person_dir = os.path.join(self.images_path, name)
        os.makedirs(person_dir, exist_ok=True)

        cap = cv2.VideoCapture(0)

        print("Press 'c' to capture a face, 's' to save encodings and quit, and 'q' to quit without saving.")
        captured_images = []

        while True:
            ret, frame = cap.read()
            cv2.imshow("Frame", frame)

            key = cv2.waitKey(1)
            if key & 0xFF == ord('c'):
                # Save the captured image
                filename = f"{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
                img_path = os.path.join(person_dir, filename)
                cv2.imwrite(img_path, frame)
                print(f"Captured and saved {filename}")
                captured_images.append(img_path)
            elif key & 0xFF == ord('s'):
                if captured_images:
                    self.process_and_save_encodings(captured_images, name)
                break
            elif key & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    def process_and_save_encodings(self, img_paths, name):
        encodings = []

        for img_path in img_paths:
            img = cv2.imread(img_path)
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Get encoding
            if len(face_recognition.face_encodings(rgb_img))> 0:
                    img_encoding = face_recognition.face_encodings(rgb_img)[0]
                    encodings.append(img_encoding)

        # Average the encodings
        avg_encoding = np.mean(encodings, axis=0)

        # Save averaged encoding to the database
        self.save_encoding_to_db(name, avg_encoding)

if __name__ == "__main__":
    face_capture = FaceCapture()
    face_capture.capture_faces()
