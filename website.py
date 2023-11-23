from flask import Flask, render_template, Response, request
import cv2
import datetime, time
import os, sys
import numpy as np
from threading import Thread

import DataLoader
import constances
import main

global prediction_text
prediction_text = ""

global model_age, model_gender, model_mood
model_age = None
model_gender = None
model_mood = None

global capture, rec_frame, grey, switch, neg, face, rec, out
capture = 0
grey = 0
neg = 0
face = 0
switch = 1
rec = 0

# make shots directory to save pics
try:
    os.mkdir('./shots')
except OSError as error:
    pass

# Load pretrained face detection model
net = cv2.dnn.readNetFromCaffe('./saved_model/deploy.prototxt.txt',
                               './saved_model/res10_300x300_ssd_iter_140000.caffemodel')

# instatiate flask app
app = Flask(__name__, template_folder='./templates')

camera = cv2.VideoCapture(0)


def run():
    app.run()


def record(out):
    global rec_frame
    while (rec):
        time.sleep(0.05)
        out.write(rec_frame)


def detect_face(frame):
    global net
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    confidence = detections[0, 0, 0, 2]

    if confidence < 0.5:
        return frame

    box = detections[0, 0, 0, 3:7] * np.array([w, h, w, h])
    (startX, startY, endX, endY) = box.astype("int")
    try:
        frame = frame[startY:endY, startX:endX]
        (h, w) = frame.shape[:2]
        r = 480 / float(h)
        dim = (int(w * r), 480)
        frame = cv2.resize(frame, dim)
    except Exception as e:
        pass
    return frame


def gen_frames():  # generate frame by frame from camera
    global out, capture, rec_frame
    while True:
        success, frame = camera.read()
        if success:
            if (face):
                frame = detect_face(frame)
            if (grey):
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if (neg):
                frame = cv2.bitwise_not(frame)
            if (capture):
                capture = 0
                now = datetime.datetime.now()
                image_name = "shot_{}.png".format(str(now).replace(":", ''));
                p = os.path.sep.join(['shots', image_name])

                resized_image = cv2.resize(frame, (DataLoader.WIDTH, DataLoader.HEIGHT))
                cv2.imwrite(p, resized_image)

                evaluate_shot(shot_filename=image_name)

            if (rec):
                rec_frame = frame
                frame = cv2.putText(cv2.flip(frame, 1), "Recording...", (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    (0, 0, 255), 4)
                frame = cv2.flip(frame, 1)

            try:
                ret, buffer = cv2.imencode('.jpg', cv2.flip(frame, 1))
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            except Exception as e:
                pass

        else:
            pass


def evaluate_shot(shot_filename):
    global model_age, model_gender, model_mood, prediction_text
    if model_age is None:
        model_age = DataLoader.load_current_model(main.AGE_EXTENSION)

    if model_gender is None:
        model_gender = DataLoader.load_current_model(main.GENDER_EXTENSION)

    if model_mood is None:
        model_mood = DataLoader.load_current_model(main.MOOD_EXTENSION)

    # Same with mood!

    pxls = np.array([0.0] * (DataLoader.WIDTH * DataLoader.HEIGHT * 3))
    pxls = pxls.reshape((-1, DataLoader.WIDTH, DataLoader.HEIGHT, 3))

    pxls[0] = DataLoader._preprocess_image("shots", shot_filename, False)

    pxls = pxls.reshape((-1, DataLoader.WIDTH, DataLoader.HEIGHT, 3))

    predictions = model_age.predict(x=pxls, batch_size=1)
    predictions = np.argmax(predictions, axis=1)

    print(f"----- Pred: {shot_filename} -----")


    if predictions[0] > 0:
        print(f"AGE: Between {constances.AGE_CATEGORIES[predictions[0] - 1]} and {constances.AGE_CATEGORIES[predictions[0]]}years old.")
        prediction_text = f"Between {constances.AGE_CATEGORIES[predictions[0] - 1]} and {constances.AGE_CATEGORIES[predictions[0]]}years old."
    else:
        print(f"AGE: Less than {constances.AGE_CATEGORIES[predictions[0]]} years old")
        prediction_text = f"Less than {constances.AGE_CATEGORIES[predictions[0]]} years old"

    predictions = model_gender.predict(x=pxls, batch_size=1)

    if int(predictions[0][0] + 0.5) >= 1:
        gender = "female"
    else:
        gender = "male"
    gender_prediction_text = f"Gender is {gender} ({predictions[0][0]:.2f})"
    print(gender_prediction_text)
    prediction_text = f"{prediction_text} {gender_prediction_text}."

    pxls = DataLoader.load_data_for_mood("shots", shot_filename)

    predictions = model_mood.predict(x=pxls, batch_size=1)


    pred_txt = ""
    for i in range(len(constances.MOOD_CATEGORIES)):
        pred_txt = f" {pred_txt} {constances.MOOD_CATEGORIES[i]} {predictions[0][i]:.2f}"

    print(f"MOOD PRED:{pred_txt}")
    prediction_text = f"{prediction_text}{pred_txt}."

    print("--- END ---")

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/predictions')
def predictions():
    return Response("hi <3")


@app.route('/requests', methods=['POST', 'GET'])
def tasks():
    global switch, camera
    if request.method == 'POST':
        if request.form.get('click') == 'Capture':
            global capture
            capture = 1
        elif request.form.get('grey') == 'Grey':
            global grey
            grey = not grey
        elif request.form.get('neg') == 'Negative':
            global neg
            neg = not neg
        elif request.form.get('face') == 'Face Only':
            global face
            face = not face
            if (face):
                time.sleep(4)
        elif request.form.get('stop') == 'Stop/Start':

            if (switch == 1):
                switch = 0
                camera.release()
                cv2.destroyAllWindows()

            else:
                camera = cv2.VideoCapture(0)
                switch = 1
        elif request.form.get('rec') == 'Start/Stop Recording':
            global rec, out
            rec = not rec
            if (rec):
                now = datetime.datetime.now()
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                out = cv2.VideoWriter('vid_{}.avi'.format(str(now).replace(":", '')), fourcc, 20.0, (640, 480))
                # Start new thread for recording the video
                thread = Thread(target=record, args=[out, ])
                thread.start()
            elif (rec == False):
                out.release()


    elif request.method == 'GET':
        return render_template('index.html', predictions=prediction_text)
    return render_template('index.html', predictions=prediction_text)
