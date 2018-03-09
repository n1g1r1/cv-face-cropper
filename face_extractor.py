#######################################################

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
path_to_validationdata = os.getcwd() + '/data/validation'
canny_lower_threshold = 40
canny_upper_threshold = 200


def build_training_set(detector, classifier = "lbp"):
    '''
    Builds a training set by asking the user for a name. The function makes a directory in `path_to_trainingdata` and `path_to_validationdata` which have the same names.

    :param detector: The detector module.
    :param str classifier: The classifier to detect the face.
    '''

    print('<face_extractor.py> Start building training set _____________________')

    # Ask for users name.
    name = input("How is your name?\n")

    folder_hash = str(hash(time.time()))
    abs_path_to_trainingdata = path_to_trainingdata + '/' + name + '.' + folder_hash
    abs_path_to_validationdata = path_to_validationdata + '/' + name + '.' + folder_hash

    # Make new dir if user not exists.
    print('Hello ' + name + '. The algorithm will now learn to recognise your face. The recording start now.')
    time.sleep(0.200)
    print('_____________________________________________________________________')
    print('Make new folder at ' + abs_path_to_trainingdata)
    os.makedirs(abs_path_to_trainingdata, exist_ok=True)
    print('Make new folder at ' + abs_path_to_validationdata)
    os.makedirs(abs_path_to_validationdata, exist_ok=True)
    print('----')

    # Get camera image.
    camera = cv.VideoCapture(0)
    output, camera_image = camera.read()

    iterator = 0

    while output and iterator < 100:

        # Get face coordinates.
        faces, eyes, image = detector.detect_faces(camera_image, detect_eyes = False, draw_bounding_boxes = False, classifier = classifier)

        if iterator is 50:
            input("Now the computer will save some variations of your face. Please tilt and rotate your face a bit, laugh or grin, make some grimaces in front of the camera to enhance the recognition even in unusual states. Hit enter when you are ready.\n")

        if len(faces) is 1:

            # Get the camera image.
            face_image = camera_image[faces[0][1]:faces[0][1]+faces[0][3], faces[0][0]:faces[0][0]+faces[0][2]]

            # Shows the image.
            cv.imshow('Face extractor', face_image)

            # Set filename.
            filename = name + "." + str(hash(time.time())) + '.jpg'
            print('Save face as ' + filename)

            # Toggle the path - either training or validation data.
            path = None

            if iterator < 50:
                path = abs_path_to_trainingdata
            else:
                path = abs_path_to_validationdata

            # Write the image.
            cv.imwrite(path + '/' + filename, face_image)

            # Iterate iterator.
            iterator += 1

            time.sleep(0.1)

        # Grab next camera image.
        output, camera_image = camera.read()

    # End procedures.
    print('Ending process. ' + str(iterator) + ' images are created and saved into ' + abs_path_to_trainingdata + ' and ' + abs_path_to_validationdata + '.')
    cv.destroyAllWindows()
    camera.release()

    print('</face_extractor.py> End ____________________________________________')
