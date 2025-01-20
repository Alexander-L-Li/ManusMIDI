from flask import Flask, Response, render_template, request, jsonify
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

# Calibration Variables
inconsistency = [0, 0]
ar_valid = [[], []]
alt = 0
defaultratios = [[0 for _ in range(4)] for _ in range(2)]
last_fingers = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_frame', methods=['POST'])
def process_frame():
    if 'frame' not in request.files:
        return jsonify({'error': 'No frame provided'}), 400
    
    # Read frame from request
    frame_file = request.files['frame']
    frame_bytes = frame_file.read()
    
    # Convert bytes to numpy array
    nparr = np.frombuffer(frame_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        return jsonify({'error': 'Invalid frame data'}), 400

    # Process frame with hand detection
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)
    
    status_message = "Processing frame..."
    
    if len(lmList) != 0:
        # Your existing hand processing logic here
        # This is where you'd implement the MIDI control logic
        vec = lmList
        
        # Calculate finger positions and generate MIDI notes
        # (Your existing logic from generate_frames)
        ratios = compute(convert(vec))
        if len(ratios) > 0:
            if right(convert(vec)) == 1:
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
        
        status_message = "Hand detected"
    
    return jsonify({
        'status': status_message
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)