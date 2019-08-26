import cv2
import face_recognition
import numpy as np
import sys
sys.path.insert(0, "sort")
from sort import Sort  # noqa


def main():
    video_capture = cv2.VideoCapture(0)
    mot_tracker = Sort(max_age=10)  # create instance of the SORT tracker

    while True:
        # Grab a single frame of video
        ret, frame = video_capture.read()
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Find all the faces and face encodings in the frame of video
        face_locations = face_recognition.face_locations(small_frame)

        if len(face_locations) != 0:
            # top right bottom left => left top right bottom
            dets_list = [[l, t, r, b, 1] for (t, r, b, l) in face_locations]
            dets = np.array(dets_list)

            trackers = mot_tracker.update(dets)
            ids = trackers[:, 4].flatten()

            for (top, right, bottom, left), id in zip(face_locations, ids):
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                # Draw a box around the face
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

                # Draw a label with a name below the face
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, str(id), (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        # Display the resulting image
        cv2.imshow('Video', frame)

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
