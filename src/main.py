from prepare_data import prepare_data
import cv2
import numpy as np
from predict import predict
from saving import study


if __name__ == '__main__':
    faces, labels = prepare_data()

    face_rec = cv2.face.LBPHFaceRecognizer_create()

    #face_rec.train(faces, np.array(labels))

    study(faces, labels, face_rec.train(faces, np.array(labels)))

    cap = cv2.VideoCapture(0)
    for i in range(15):
        cap.read()

    ret, frame = cap.read()
    cv2.imwrite(r'./Predict/cam.jpg', frame)

    test_img = cv2.imread(r"./Predict/cam.jpg", 0)

    predicted_img = predict(test_img, face_rec)
    print("Prediction complete")

    cv2.imshow('img', predicted_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
