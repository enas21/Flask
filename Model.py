import cv2
from keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
import pickle
from collections import deque

model_path = "D:/My Drive/7wheat_deseases_Resent50_model.h5"
input = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSQTGGdC260U-qHLMkl9wN-nHlkAu-HJr1DvA&usqp=CAU"
# input = input("Uploud your image : ")
label = "label"
# load the trained model and label binarizer from disk
moodel = load_model(model_path)
lb = pickle.loads(open("label", "rb").read())

# initialize the image mean for mean subtraction along with the
# predictions queue
mean = np.array([123.68, 116.779, 103.939][::1], dtype="float32")
Q = deque(maxlen=128)
vs = cv2.VideoCapture(input)
(W, H) = (None, None)
while True:
    (grabbed, frame) = vs.read()
    if not grabbed:
        break
    if W is None or H is None:
        (H, W) = frame.shape[:2]
    output = frame.copy()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (224, 224)).astype("float32")
    frame -= mean
    preds = moodel.predict(np.expand_dims(frame, axis=0))[0]
    Q.append(preds)
    results = np.array(Q).mean(axis=0)
    i = np.argmax(results)
    label = lb.classes_[i]
    text = "PREDICTION: {}".format(label.upper())
    cv2.putText(output, text, (4, 4), cv2.FONT_HERSHEY_SIMPLEX,
                0.25, (200, 255, 155), 2)
    # show the output image
    image = cv2.resize(output, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    plt.imshow(image)
    plt.show()
    for i in range(0, 7):
        print(lb.classes_[i])
        print(results[i] * 100)
    # print(LABELS)
    # print(results)
    print(text)
    key = cv2.waitKey(10) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break
vs.release()