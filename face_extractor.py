#######################################################
# Naive face extractor as training set
# @author: Christian Reichel
# Version: 0.1a
# -----------------------------------------------------
# This script makes a directory based on an entered
# name and a hash value and saves X faces into that
# folder as training set for face recognition.
#######################################################

# IMPORTS.
import os               # path finding
import cv2 as cv        # webcam image
import time             # sleep time
import datetime

# Settings.
path_to_trainingdata = os.getcwd() + '/data/training'
canny_lower_threshold = 40
canny_upper_threshold = 200

def build_training_set(detector):

    # Ask for users name.
    name = input("<face_extractor.py> How is your name?\n")

    new_path_str = path_to_trainingdata + '/' + name + '_' + str(hash(int))

    # Make new dir if user not exists.
    print('<face_extractor.py> Hello ' + name + '. The algorithm will now learn to recognise your face. Please rotate it slowly to make the recognition more stable. The recording start now.')
    time.sleep(0.100)
    print('-------------------------')
    print('<face_extractor.py> Make new folder at ' + new_path_str)
    os.makedirs(new_path_str, exist_ok=True)

    # Get camera image.
    camera = cv.VideoCapture(0)
    output, camera_image = camera.read()

    iterator = 0

    while output and iterator < 50:

        # Get face coordinates.
        faces, eyes, image = detector.detect_faces(camera_image, detect_eyes = False, draw_bounding_boxes = False)

        if len(faces) is 1:

            # Get the camera image.
            face_image = camera_image[faces[0][1]:faces[0][1]+faces[0][3], faces[0][0]:faces[0][0]+faces[0][2]]

            # Shows the image.
            cv.imshow('Face extractor', face_image)

            # Write the image.
            filename = name + "_" + str(hash(time.time())) + '.jpg'
            print('<face_extractor.py> Save image: ' + filename)
            cv.imwrite(new_path_str + '/' + filename, face_image)

            # Iterate iterator.
            iterator += 1

            time.sleep(0.2)

        # Grab next camera image.
        output, camera_image = camera.read()

    # End procedures.
    print('<face_extractor.py> Ending process. ' + str(iterator) + ' images are created and saved into ' + new_path_str + '.')
    cv.destroyAllWindows()
    camera.release()


# Method call. Uncomment for debugging.
# build_training_set()