"""
Preparing data to next work with it.
In this block of code I open photos
and prepare them to detect faces on them.
Also I open this photos in windowed mode.
"""
import os
from detecting import detecting
import cv2


def prepare_data():
    directories = os.listdir('./Emotions')
    faces = []
    labels = []

    for image_path in directories:
        if image_path[0] == 'h':
            label = 1
        else:
            label = 2

        image_path_dub = './Emotions/' + image_path

        image = cv2.imread(image_path_dub, 0)

        cv2.imshow('Emotions', image)
        cv2.waitKey(100)

        face, rect = detecting(image)
        if face is not None:
            faces.append(face)
            labels.append(label)

    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.destroyAllWindows()

    return faces, labels
