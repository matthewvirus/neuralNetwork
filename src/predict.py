from detecting import detecting
from draw_all import *


def predict(test_img, face_rec):
    subject = ['Angry', 'Happy', 'Fear']
    img = test_img.copy()

    face, rect = detecting(img)
    label = face_rec.predict(face)
    txt_label = subject[label[0]]

    draw_rectangle(img, rect)
    draw_text(img, txt_label, rect[0], rect[1] - 5)

    return img
