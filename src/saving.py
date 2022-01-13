from tensorflow import keras
from keras.models import Sequential
import keras.callbacks
from keras.layers import Convolution2D, MaxPool2D, Dropout, Flatten, Dense


def study(faces, labels, face_rec):
    print("Preparing data to training...")
    print("Preparing completed!")

    print("Total faces: ", len(faces))
    print("Total labels: ", len(labels))

    print("Predicting images...")

    model = Sequential()
    model.add(Convolution2D(
        filters=32,
        kernel_size=(3, 3),
        padding='valid',
        input_shape=(690, 480, 1),
        activation='relu')
    )
    model.add(Convolution2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(len(faces), activation='softmax'))
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adadelta',
        metrics=['accuracy']
    )

    learning_rating = keras.callbacks.ReduceLROnPlateau(
        monitor='val_accuracy',
        patience=3,
        verbose=1,
        factor=0.5,
        min_lr=0.00001
    )

    model.fit(
        validation_data=face_rec,
        callbacks=[learning_rating],
        batch_size=64,
        epochs=124
    )

    model.save(r'./src/mnist_faces.h5')
