from audioop import avg
from glob import glob
from itertools import count
import cv2
import mediapipe as mp
import numpy as np
import threading as th
import sounddevice as sd
import audio

# global variables
x = 0                                       # X axis head pose (right/left)
y = 0                                       # Y axis head pose   (up/down)

X_AXIS_CHEAT = 0
Y_AXIS_CHEAT = 0

def pose():
    global VOLUME_NORM, x, y, X_AXIS_CHEAT, Y_AXIS_CHEAT
    
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    cap = cv2.VideoCapture(0)
    mp_drawing = mp.solutions.drawing_utils #drawing_utils module contains utility functions for drawing landmarks and connections on images.
    # mp_drawing_styles = mp.solutions

    while cap.isOpened():
        success, image = cap.read() #success indicates whether the frame was successfully read, and image contains the actual frame data.
       
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)  #flips the image horizontally and converts its color space from BGR to RGB. The flipping is done to make the display more intuitive in a later step.

        # To improve performance
        image.flags.writeable = False
        
        results = face_mesh.process(image) #performs the face mesh detection on the current frame using the face_mesh model provided by the mediapipe library. The result is stored in the results variable.
        
        # To improve performance
        image.flags.writeable = True
        
        # Convert the color space from RGB to BGR
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) #performs the face mesh detection on the current frame using the face_mesh model provided by the mediapipe library. The result is stored in the results variable.

        img_h, img_w, img_c = image.shape #number of channels (img_c) of the image.
        face_3d = []
        face_2d = []
        
        face_ids = [33, 263, 1, 61, 291, 199]
    #Left eye inner corner: ID 33 | Left eye inner corner: ID 33 | Nose tip: ID 1 | Left eyebrow inner end: ID 61 | Left eyebrow inner end: ID 61
        if results.multi_face_landmarks: #LIST of detected faces and their corresponding landmarks.
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=image, # specifies the image on which the landmarks will be drawn.
                    landmark_list=face_landmarks,# provides the landmarks for the current face to be drawn.
                    connections=mp_face_mesh.FACEMESH_CONTOURS, #specifies the connections or contours to be drawn between the landmarks.
                    landmark_drawing_spec=None) #specifies the connections or contours to be drawn between the landmarks.
                for idx, lm in enumerate(face_landmarks.landmark):
                    # print('represents the coordinates of the current landmark' lm) 
                    if idx in face_ids:
                        if idx == 1:
                            nose_2d = (lm.x * img_w, lm.y * img_h)
                            nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 8000)

                        x, y = int(lm.x * img_w), int(lm.y * img_h)

                        # Get the 2D Coordinates
                        face_2d.append([x, y])

                        # Get the 3D Coordinates
                        face_3d.append([x, y, lm.z])       
                
                # Convert it to the NumPy array
                face_2d = np.array(face_2d, dtype=np.float64)

                # Convert it to the NumPy array
                face_3d = np.array(face_3d, dtype=np.float64)

                # The camera matrix
                focal_length = 1 * img_w

                cam_matrix = np.array([ [focal_length, 0, img_h / 2],
                                        [0, focal_length, img_w / 2],
                                        [0, 0, 1]])

                # The Distance Matrix
                dist_matrix = np.zeros((4, 1), dtype=np.float64)

                # Solve PnP
                success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

                
                rmat, jac = cv2.Rodrigues(rot_vec) #converts the rotation vector (rot_vec) obtained from solvePnP into a rotation matrix (rmat) using the Rodrigues transformation. The rotation matrix represents the rotation of the head in 3D space.

                # Get angles
                angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat) #rotation matrix (rmat) into its constituent Euler angles (angles). Here, angles represents the three rotation angles around the x, y, and z axes, 

                # Get the y rotation degree
                x = angles[0] * 360
                y = angles[1] * 360

                # print(y)

                # See where the user's head tilting
                if y < -10:
                    text = "Looking Left"
                elif y > 10:
                    text = "Looking Right"
                elif x < -10:
                    text = "Looking Down"
                else:
                    text = "Forward"
                text = str(int(x)) + "::" + str(int(y)) + text
                # print(str(int(x)) + "::" + str(int(y)))
                # print("x: {x}   |   y: {y}  |   sound amplitude: {amp}".format(x=int(x), y=int(y), amp=audio.SOUND_AMPLITUDE))
                
                # Y is left / right
                # X is up / down
                if y < -10 or y > 10:
                    X_AXIS_CHEAT = 1
                else:
                    X_AXIS_CHEAT = 0

                if x < -5:
                    Y_AXIS_CHEAT = 1
                else:
                    Y_AXIS_CHEAT = 0

                # print(X_AXIS_CHEAT, Y_AXIS_CHEAT, text)
                # Display the nose direction
                nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)

                p1 = (int(nose_2d[0]), int(nose_2d[1]))
                p2 = (int(nose_3d_projection[0][0][0]), int(nose_3d_projection[0][0][1]))
                
                cv2.line(image, p1, p2, (255, 0, 0), 2)

                # Add the text on the image
                cv2.putText(image, text, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        cv2.imshow('Head Pose Estimation', image)

        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()

#############################
if __name__ == "__main__":
    t1 = th.Thread(target=pose)

    t1.start()

    t1.join()