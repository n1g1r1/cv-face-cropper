#######################################################
# Naive face extractor as training set
# @author: Christian Reichel
# Version: 0.1a
# -----------------------------------------------------
# This script makes a directory based on an entered
# name and a hash value and saves X faces into that
# folder as training set for face recognition.
#
# Algorithm description:
# 1) Asks for a name
# 2) Makes new dir
# 3) Saves a bunch of images into that folder
#    _Advanced way: Saves keypoints into that folder.
#
# Knowm challenges:
# [x] Record images
# [ ] Replace image face_extractor with keypoint face_extractor.
#######################################################

# IMPORTS
import os               # path finding
import cv2 as cv        # webcam image
import time             # sleep time
import json

import requests
import numpy as np
class NumPyArangeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist() # or map(int, obj)
        return json.JSONEncoder.default(self, obj)

def build_training_set():
    # Pretrained haar cascade classifiers.
    face_cascade = cv.CascadeClassifier(os.path.dirname(__file__) + 'data/cascades/haarcascade_frontalface_default.xml')

    # Make new dir if user not exists
    name = "tmp"
    new_path_str = os.getcwd() + '/data/training/' + name + '_' + str(hash(int))
    time.sleep(0.100)
    os.makedirs(new_path_str, exist_ok=True)

    # Get camera image.
    camera = cv.VideoCapture(0)
    output, camera_image = camera.read()

    iterator = 0

    while output and iterator < 1:

        # Save grayscale image.
        camera_image_gray = cv.cvtColor(camera_image, cv.COLOR_BGR2GRAY)

        # Get face coordinates.
        faces = face_cascade.detectMultiScale(camera_image_gray, 1.3, 5)

        if len(faces) is 1:

            print("I think you are ...")
            face_image = camera_image[faces[0][1]:faces[0][1]+faces[0][3], faces[0][0]:faces[0][0]+faces[0][2]]
            response = send_request(face_image)
            print("... ",response.json()["label"], "probably ... ", response.json()["probability"])
            # Write the image.
            filename = name + str(iterator) + '.jpg'
            cv.imwrite(new_path_str + '/' + filename, face_image)

            # Iterate iterator.
            iterator += 1


        # Grab next camera image.
        output, camera_image = camera.read()

    # End procedures.
    camera.release()

def send_request(image_np):
    url = "http://letsfaceit.quving.com/api/predict"
    data = {}
    data["image"]= json.dumps(image_np.tolist())
    data["mode"]= "predict"
    response = requests.post(url, json = data)
    return response

# Method call. Uncomment for debugging.
build_training_set()
