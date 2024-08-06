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

        # Resize frame for a faster speed
        self.frame_resizing = 0.5
        # 0.25 for faster speed

        self.load_encodings_from_db()

    def load_encodings_from_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT name, encoding FROM faces')
        rows = cursor.fetchall()
        conn.close()

        self.known_face_names = [row[0] for row in rows]
        self.known_face_encodings = [pickle.loads(row[1]) for row in rows]

    def detect_known_faces(self, frame):
        small_frame = cv2.resize(frame, (0, 0), fx=self.frame_resizing, fy=self.frame_resizing)
        # Find all the faces and face encodings in the current frame of video
        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            name = "Unknown"

            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            print(best_match_index,face_distances[best_match_index])
            if matches[best_match_index] and face_distances[best_match_index] < 0.45:
                name = self.known_face_names[best_match_index]
            face_names.append(name)

        # Convert to numpy array to adjust coordinates with frame resizing quickly
        face_locations = np.array(face_locations)
        face_locations = face_locations / self.frame_resizing
        return face_locations.astype(int), face_names

            # face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            # print(face_distances)
            # # if face_distances < 1.0:
            # best_match_index = np.argmin(face_distances)
            # print(face_distances[best_match_index])
            # if face_distances[best_match_index] < 0.6:
            #     if matches[best_match_index]:
            #         name = self.known_face_names[best_match_index]
            #     face_names.append(name)