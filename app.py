from flask import Flask, Response, render_template, request, jsonify
import cv2
import mediapipe as mp
import time
import tinysoundfont
import os
import numpy as np
from io import BytesIO

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

# Global frame buffer for video feed
current_frame = None
frame_lock = None
last_processed_frame = None

def get_frame():
    global current_frame, last_processed_frame
    if current_frame is None:
        return None
    
    # Return the frame and store it as last processed
    last_processed_frame = current_frame
    return current_frame

# Frame Generator
def generate_frames():
    global left_adjust, right_adjust, last_fingers, alt, defaultratios, ar_valid, inconsistency, current_frame
    pTime = 0
    frame_count = 0
    last_fps_time = time.time()

    # Calibration Step
    while len(ar_valid[0]) <= 20 or len(ar_valid[1]) <= 20:
        img = get_frame()
        if img is None:
            # Create a blank frame with message
            img = np.zeros((720, 1280, 3), dtype=np.uint8)
            cv2.putText(img, "Waiting for camera...", (50, 360), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        else:
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

        # Calculate and display FPS
        frame_count += 1
        current_time = time.time()
        if current_time - last_fps_time >= 1.0:
            fps = frame_count / (current_time - last_fps_time)
            cv2.putText(img, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            frame_count = 0
            last_fps_time = current_time

        ret, buffer = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    # Calculate default ratios
    for k in range(2):
        for i in range(len(ar_valid[k][0])):
            defaultratios[k][i] = sum(ratio[i] for ratio in ar_valid[k]) / len(ar_valid[k])

    # Main Frame Generation
    frame_count = 0
    last_fps_time = time.time()
    
    while True:
        img = get_frame()
        if img is None:
            # Create a blank frame with message
            img = np.zeros((720, 1280, 3), dtype=np.uint8)
            cv2.putText(img, "Waiting for camera...", (50, 360), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        else:
            img = detector.findHands(img)
            lmlist = detector.findPosition(img, alt)
            alt = 1 - alt

            if lmlist:
                ratios = compute(convert(lmlist))
                if len(ratios) > 0:
                    if right(convert(lmlist)) == 1:
                        if len(defaultratios[1]) > 0:
                            diff = computeDiff(ratios, defaultratios[1])
                            if diff > 0.1:
                                note = int((diff - 0.1) / 0.2 * 7)
                                if note < 0:
                                    note = 0
                                if note > 7:
                                    note = 7
                                synth.noteon(0, notes[note] + 12, 100)
                                synth.noteoff(0, notes[note] + 12, 100)
                    else:
                        if len(defaultratios[0]) > 0:
                            diff = computeDiff(ratios, defaultratios[0])
                            if diff > 0.1:
                                note = int((diff - 0.1) / 0.2 * 7)
                                if note < 0:
                                    note = 0
                                if note > 7:
                                    note = 7
                                synth.noteon(0, notes[note], 100)
                                synth.noteoff(0, notes[note], 100)

                cv2.putText(img, "HAND TRACKING ACTIVE", (10, 70), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)

            # Calculate and display FPS
            frame_count += 1
            current_time = time.time()
            if current_time - last_fps_time >= 1.0:
                fps = frame_count / (current_time - last_fps_time)
                cv2.putText(img, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                frame_count = 0
                last_fps_time = current_time

        ret, buffer = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/upload_frame', methods=['POST'])
def upload_frame():
    global current_frame
    if 'frame' not in request.files:
        return jsonify({'error': 'No frame provided'}), 400
    
    # Read frame from request
    frame_file = request.files['frame']
    frame_bytes = frame_file.read()
    
    # Convert bytes to numpy array
    nparr = np.frombuffer(frame_bytes, np.uint8)
    current_frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if current_frame is None:
        return jsonify({'error': 'Invalid frame data'}), 400

    return jsonify({'status': 'Frame received'})

# Calibration Variables
inconsistency = [0, 0]
ar_valid = [[], []]
alt = 0
defaultratios = [[0 for _ in range(4)] for _ in range(2)]
last_fingers = None

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)