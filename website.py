from flask import Flask, render_template, Response, request
import cv2
import datetime, time
import os, sys
import numpy as np

import CostumerValueModel
import DataLoader
import constances
import main

global prediction_age, prediction_gender, prediction_mood, prediction_cv, current_company
prediction_age = ""
prediction_gender = ""
prediction_mood = ""
prediction_cv = ""
current_company = "Electronics"

global model_age, model_gender, model_mood
model_age = None
model_gender = None
model_mood = None

global capture, switch, face, rec, out
capture = 0
face = 0
switch = 1

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
    global out, capture
    while True:
        success, frame = camera.read()
        if success:
            if (face):
                frame = detect_face(frame)
            if (capture):
                now = datetime.datetime.now()
                image_name = "shot_{}.png".format(str(now).replace(":", ''));
                p = os.path.sep.join(['shots', image_name])

                resized_image = cv2.resize(frame, (DataLoader.WIDTH, DataLoader.HEIGHT))
                cv2.imwrite(p, resized_image)

                evaluate_shot(shot_filename=image_name)
                capture = 0

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
    global model_age, model_gender, model_mood, prediction_age, prediction_mood, prediction_gender, prediction_cv
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
        prediction_age = f"Between {constances.AGE_CATEGORIES[predictions[0] - 1]} and {constances.AGE_CATEGORIES[predictions[0]]}years old."
    else:
        print(f"AGE: Less than {constances.AGE_CATEGORIES[predictions[0]]} years old")
        prediction_age = f"Less than {constances.AGE_CATEGORIES[predictions[0]]} years old"

    predictions_age_for_cv = predictions[0]

    predictions = model_gender.predict(x=pxls, batch_size=1)

    if int(predictions[0][0] + 0.5) >= 1:
        gender = "female"
    else:
        gender = "male"
    prediction_gender = f"Gender is {gender} ({predictions[0][0]:.2f})"
    print(prediction_gender)


    pxls = DataLoader.load_data_for_mood("shots", shot_filename)

    predictions = model_mood.predict(x=pxls, batch_size=1)


    prediction_mood = ""
    for i in range(len(constances.MOOD_CATEGORIES)):
        prediction_mood = f" {prediction_mood} {constances.MOOD_CATEGORIES[i]} {predictions[0][i]:.2f}"

    print(f"MOOD PRED:{prediction_mood}")

    prediction_cv = f'{(CostumerValueModel.calculateValue(current_company, gender == "female", predictions_age_for_cv, predictions[0]) * 100):.2f}'

    print(f"CV: {prediction_cv}")


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


@app.route('/company', methods = ['POST', 'GET'])
def customer_value():
    global current_company
    current_company = request.form.get('company')
    print(f"Company changed to {current_company}")


    return renderedTemplate()

@app.route('/requests', methods=['POST', 'GET'])
def tasks():
    global switch, camera
    if request.method == 'POST':
        if request.form.get('click') == 'Capture':
            global capture
            capture = 1

            while(capture):
                time.sleep(10)
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
    elif request.method == 'GET':
        return renderedTemplate()
    return renderedTemplate()

def renderedTemplate():
    return render_template('index.html', p_age=prediction_age, p_gender = prediction_gender, p_mood = prediction_mood, p_cv = prediction_cv, p_company = current_company)