"""
This is code which detected face on the photo
by special cascade map and BW mode of that.
Try to load photos which size are the same.
"""
import cv2


def detecting(image):
    grayImage = cv2.cvtColor(image, cv2.COLOR_BAYER_BG2GRAY)

    # This can be your path to your xml map
    cascadePath = r"C:\Users\matth\PycharmProjects\neuralNetwork\src\lbpcascade_frontalface.xml"

    faceCascade = cv2.CascadeClassifier(cascadePath)

    faces = faceCascade.detectMultiScale(
        grayImage,
        scaleFactor=1.2,
        minNeighbors=1
    )

    if len(faces) == 0:
        return None, None

    (x, y, w, h) = faces[0]
    return grayImage[y:y+w, x:x+h], faces[0]
