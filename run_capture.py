import cv2
import tensorflow as tf
import numpy as np

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, input_shape=(50, 50, 1), activation='relu', kernel_size=(2, 2)),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv2D(64, activation='relu', kernel_size=(2, 2)),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax'),
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

names = ['obama', 'omar', 'other']

model.load_weights('model.h5')
facecasc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
facesID = {0: 'Obama', 1: 'Omar', 2: 'other'}
cap = cv2.VideoCapture(0)
while True:
    # Find haar cascade to draw bounding box around face
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facecasc.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y - 50), (x + w, y + h + 10), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        img = np.array(roi_gray)
        img = img / 255
        img = np.resize(img, (1, 50, 50, 1))
        prediction = model.predict(img)
        maxindex = int(np.argmax(prediction))
        cv2.putText(frame, names[maxindex], (x + 20, y - 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                    cv2.LINE_AA)

    cv2.imshow('Video', cv2.resize(frame, (800, 400), interpolation=cv2.INTER_CUBIC))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



# img = cv2.imread('obama_faces/70.jpg', cv2.IMREAD_GRAYSCALE)
# img = np.array(img)
# img = img/255
# img = np.resize(img, (1, 50, 50, 1))
# prediction = model.predict(img)
# maxindex = int(np.argmax(prediction))
# print(names[maxindex])
