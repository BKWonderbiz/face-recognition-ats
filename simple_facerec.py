import face_recognition
import cv2
import numpy as np
import sqlite3
import pickle

class SimpleFacerec:
    def __init__(self, db_path='face_encodings.db'):
        self.known_face_encodings = []
        self.known_face_names = []
        self.db_path = db_path

        # Resize frame for faster speed
        self.frame_resizing = 1  # Reduce the resizing ratio for better performance

        self.load_encodings_from_db()

    def load_encodings_from_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT name, encoding FROM faces')
        rows = cursor.fetchall()
        conn.close()

        self.known_face_names = [row[0] for row in rows]
        self.known_face_encodings = [pickle.loads(row[1]) for row in rows]

    def save_encoding_to_db(self, name, encoding):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('INSERT INTO faces (name, encoding) VALUES (?, ?)', (name, pickle.dumps(encoding)))
        conn.commit()
        conn.close()

    def detect_known_faces(self, frame):
        small_frame = cv2.resize(frame, (0, 0), fx=self.frame_resizing, fy=self.frame_resizing)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            name = "Unknown"

            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index] and face_distances[best_match_index] < 0.45:
                name = self.known_face_names[best_match_index]
            face_names.append(name)

        face_locations = np.array(face_locations) / self.frame_resizing
        return face_locations.astype(int), face_names
