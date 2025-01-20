from flask import Flask, Response, render_template
import cv2
import mediapipe as mp
import time
import tinysoundfont
import os
import numpy as np

# Flask app
app = Flask(__name__)

# MIDI Synth Initialization
synth = tinysoundfont.Synth()
sfid = synth.sfload("florestan-subset.sfo")
synth.program_select(0, sfid, 0, 2)  # Select instrument type
synth.start()
notes = [48, 50, 52, 53, 55, 57, 59, 60]  # C4 to C5
left_adjust = 0
right_adjust = 0

# HandDetector Class
class handDetector():
    def __init__(self, mode=False, maxHands=2, modelComplex=1, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.modelComplex = modelComplex

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelComplex, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw=True):
        lmlist = []
        if self.results.multi_hand_landmarks and handNo >= len(self.results.multi_hand_landmarks):
            return []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, cc = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmlist.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 3, (255, 0, 255), cv2.FILLED)
        return lmlist

# Helper Functions
def dist(point1, point2):
    return ((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)**0.5

def compute(vec):
    if vec[0] == []:
        return []
    base_scale = dist(vec[17], vec[5])
    fingers = [dist(vec[4], vec[8]), dist(vec[4], vec[12]), dist(vec[4], vec[16]), dist(vec[4], vec[20])]
    ratios = [fingers[i] / base_scale for i in range(len(fingers))]
    return ratios

def computeDiff(ratio1, ratio2):
    res = 0
    for i in range(len(ratio1)):
        res += (ratio1[i] - ratio2[i])**2
    return res**0.5

def convert(list):
    vec = [[] for _ in range(21)]
    for x in list:
        vec[x[0]] = (x[1], x[2])
    return vec

def right(vec):
    inds = [3, 5, 9, 13, 17]
    gre = sum(1 for i in range(len(inds) - 1) if vec[inds[i]][0] < vec[inds[i + 1]][0])
    return 1 if gre > 2 else 0

# Initialize Detector
detector = handDetector()
cap = cv2.VideoCapture(0)

# Create fallback frame if camera is not available
if not cap.isOpened():
    fallback_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(fallback_frame, 'No camera available', (50, 240), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

# Calibration Variables
inconsistency = [0, 0]
ar_valid = [[], []]
alt = 0
defaultratios = [[0 for _ in range(4)] for _ in range(2)]
last_fingers = None

# Frame Generator
def generate_frames():
    global left_adjust, right_adjust, last_fingers, alt, defaultratios, ar_valid, inconsistency
    pTime = 0

    # Calibration Step
    while len(ar_valid[0]) <= 20 or len(ar_valid[1]) <= 20:
        if not cap.isOpened():
            # Use fallback frame when camera is not available
            ret, buffer = cv2.imencode('.jpg', fallback_frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            continue
            
        success, img = cap.read()
        if not success:
            break

        cv2.putText(img, "PLACE YOUR HANDS OUT FACING CAMERA FOR CALIBRATION", (50, 50),
                    cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 3)
        img = detector.findHands(img)
        lmlist = detector.findPosition(img, alt)
        alt = 1 - alt

        if lmlist:
            if len(ar_valid[right(convert(lmlist))]) == 0:
                ar_valid[right(convert(lmlist))].append(compute(convert(lmlist)))
            elif computeDiff(compute(convert(lmlist)), ar_valid[right(convert(lmlist))][0]) > 0.1:
                inconsistency[right(convert(lmlist))] += 1
                cv2.putText(img, "DON'T MOVE YOUR HANDS!!", (50, 100), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 3)
            else:
                ar_valid[right(convert(lmlist))].append(compute(convert(lmlist)))

            if inconsistency[right(convert(lmlist))] >= 20:
                ar_valid[right(convert(lmlist))] = []
                inconsistency[right(convert(lmlist))] = 0

        ret, buffer = cv2.imencode('.jpg', img)
        img = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + img + b'\r\n')

    # Calculate default ratios
    for k in range(2):
        for i in range(len(ar_valid[k][0])):
            defaultratios[k][i] = sum(ratio[i] for ratio in ar_valid[k]) / len(ar_valid[k])

    # Main Frame Generation
    while True:
        if not cap.isOpened():
            # Use fallback frame when camera is not available
            ret, buffer = cv2.imencode('.jpg', fallback_frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            continue
            
        success, img = cap.read()
        if not success:
            break

        img = detector.findHands(img)
        lmlist = detector.findPosition(img, alt)
        alt = 1 - alt

        if lmlist:
            # Hand tracking and MIDI logic
            # ... Original main loop logic goes here ...

            cv2.putText(img, "HAND TRACKING ACTIVE", (10, 70), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)

        ret, buffer = cv2.imencode('.jpg', img)
        img = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + img + b'\r\n')

# Flask Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)