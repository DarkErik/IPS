from flask import Flask, Response, render_template, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename

import cv2
import numpy as np
import os, sys
import datetime

import CostumerValueModel
import DataLoader
import main
import constances

# make shots and uploads directory to save pics
try:
    os.mkdir('./shots')
except OSError as error:
    pass

try:
    os.mkdir('./uploads')
except OSError as error:
    pass

# Load pretrained face detection model
net = cv2.dnn.readNetFromCaffe('./saved_model/deploy.prototxt.txt',
                               './saved_model/res10_300x300_ssd_iter_140000.caffemodel')

# instatiate flask app
app = Flask(__name__, template_folder='./templates')

prediction = {}
model = {}

video = None
camera_started = False

frame = None

face_only = False

model_age = None
model_gender = None
model_mood = None


def start_camera():
    global camera_started
    if camera_started:
        return

    global video
    video = cv2.VideoCapture(0)
    camera_started = True


def stop_camera():
    global camera_started
    if not camera_started:
        return

    global video
    video.release()
    cv2.destroyAllWindows()
    camera_started = False


def run():
    app.run()


def detect_face():
    global net
    global frame
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
    global video
    global frame
    global face_only

    while camera_started:
        success, frame = video.read()
        if success:
            if face_only:
                frame = detect_face()

            cv2.imwrite('t.jpg', frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + open('t.jpg', 'rb').read() + b'\r\n')


def evaluate_shot(shot_filename):
    global model_age, model_gender, model_mood
    if model_age is None:
        model_age = DataLoader.load_current_model(main.AGE_EXTENSION)

    if model_gender is None:
        model_gender = DataLoader.load_current_model(main.GENDER_EXTENSION)

    if model_mood is None:
        model_mood = DataLoader.load_current_model(main.MOOD_EXTENSION)

    # Same with mood!

    pxls = np.array([0.0] * (DataLoader.WIDTH * DataLoader.HEIGHT * 3))
    pxls = pxls.reshape((-1, DataLoader.WIDTH, DataLoader.HEIGHT, 3))

    pxls[0] = DataLoader._preprocess_image("uploads", shot_filename, False)

    pxls = pxls.reshape((-1, DataLoader.WIDTH, DataLoader.HEIGHT, 3))

    predictions = model_age.predict(x=pxls, batch_size=1)
    predictions = np.argmax(predictions, axis=1)

    print(f"----- Pred: {shot_filename} -----")

    if predictions[0] > 0:
        print(
            f"AGE: Between {constances.AGE_CATEGORIES[predictions[0] - 1]} and {constances.AGE_CATEGORIES[predictions[0]]}years old.")
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

    pxls = DataLoader.load_data_for_mood("uploads", shot_filename)

    predictions = model_mood.predict(x=pxls, batch_size=1)

    prediction_mood = ""
    for i in range(len(constances.MOOD_CATEGORIES)):
        prediction_mood = f" {prediction_mood} {constances.MOOD_CATEGORIES[i]} {predictions[0][i]:.2f}"

    print(f"MOOD PRED:{prediction_mood}")

    print("--- END ---")

    return {
        "age": prediction_age,
        "gender": prediction_gender,
        "mood": prediction_mood
    }


def take_shot():
    global frame

    now = datetime.datetime.now()
    image_name = "shot_{}.png".format(now.strftime("%Y-%m-%d_%H%M%S"))
    p = os.path.sep.join(['shots', image_name])

    resized_image = cv2.resize(frame, (DataLoader.WIDTH, DataLoader.HEIGHT))
    cv2.imwrite(p, resized_image)

    return p


@app.route('/analysis')
def analysis():
    stop_camera()
    return render_template('analysis.html', camera_started=camera_started)


@app.route('/camera')
def camera():
    start_camera()
    return render_template('camera.html', camera_started=camera_started)


@app.route('/')
def index():
    stop_camera()
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/company', methods=['POST'])
def customer_value():
    company = request.get_json()["company"]

    # Get last uploaded image
    files = os.listdir("uploads")
    shot_filename = files[len(files) - 1]

    global model_age, model_gender, model_mood
    if model_age is None:
        model_age = DataLoader.load_current_model(main.AGE_EXTENSION)

    if model_gender is None:
        model_gender = DataLoader.load_current_model(main.GENDER_EXTENSION)

    if model_mood is None:
        model_mood = DataLoader.load_current_model(main.MOOD_EXTENSION)

    # Same with mood!

    pxls = np.array([0.0] * (DataLoader.WIDTH * DataLoader.HEIGHT * 3))
    pxls = pxls.reshape((-1, DataLoader.WIDTH, DataLoader.HEIGHT, 3))

    pxls[0] = DataLoader._preprocess_image("uploads", shot_filename, False)

    pxls = pxls.reshape((-1, DataLoader.WIDTH, DataLoader.HEIGHT, 3))

    predictions = model_age.predict(x=pxls, batch_size=1)
    predictions = np.argmax(predictions, axis=1)

    print(f"----- Pred: {shot_filename} -----")

    if predictions[0] > 0:
        print(
            f"AGE: Between {constances.AGE_CATEGORIES[predictions[0] - 1]} and {constances.AGE_CATEGORIES[predictions[0]]}years old.")
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

    pxls = DataLoader.load_data_for_mood("uploads", shot_filename)

    predictions = model_mood.predict(x=pxls, batch_size=1)

    if company != "None":
        prediction_cv = f'{(CostumerValueModel.calculateValue(company, gender == "female", predictions_age_for_cv, predictions[0]) * 100):.2f}'
        print(f"CV: {prediction_cv}")
    else:
        prediction_cv = "0.0"

    return Response(prediction_cv, mimetype='text/plain')


@app.route('/requests', methods=['POST'])
def tasks():
    global face_only

    data = request.get_json()
    action = data.get('action')

    if action == 'capture':
        image_name = take_shot()

        return Response(image_name, mimetype='text/plain')
    elif action == 'faceOnly':
        face_only = True
    elif action == 'full':
        face_only = False

    return render_template('camera.html', face_only=face_only)


@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    filename = secure_filename(file.filename)
    filepath = os.path.join('uploads', filename)
    file.save(filepath)

    prediction = evaluate_shot(file.filename)

    return render_template('analysis.html', filename=filename, prediction=prediction)


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory('uploads', filename)
