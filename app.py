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

# Calibration Variables
inconsistency = [0, 0]
ar_valid = [[], []]
alt = 0
defaultratios = [[0 for _ in range(4)] for _ in range(2)]
last_fingers = None

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

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_hand', methods=['POST'])
def process_hand():
    if not request.is_json:
        return jsonify({'error': 'Expected JSON data'}), 400
    
    data = request.get_json()
    if 'landmarks' not in data:
        return jsonify({'error': 'No landmarks provided'}), 400

    landmarks = data['landmarks']
    
    # Convert landmarks to the format expected by your existing code
    lmlist = []
    for i, landmark in enumerate(landmarks):
        x, y = landmark
        lmlist.append([i, int(x), int(y)])
    
    # Process hand landmarks for MIDI control
    if len(lmlist) > 0:
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

    return jsonify({'status': 'Processed'})

@app.route('/calibrate', methods=['POST'])
def calibrate():
    if not request.is_json:
        return jsonify({'error': 'Expected JSON data'}), 400
    
    data = request.get_json()
    if 'landmarks' not in data:
        return jsonify({'error': 'No landmarks provided'}), 400

    landmarks = data['landmarks']
    
    # Convert landmarks to the format expected by your existing code
    lmlist = []
    for i, landmark in enumerate(landmarks):
        x, y = landmark
        lmlist.append([i, int(x), int(y)])
    
    # Process calibration
    if lmlist:
        hand_side = right(convert(lmlist))
        if len(ar_valid[hand_side]) == 0:
            ar_valid[hand_side].append(compute(convert(lmlist)))
            return jsonify({'status': 'First calibration point recorded'})
        elif computeDiff(compute(convert(lmlist)), ar_valid[hand_side][0]) > 0.1:
            inconsistency[hand_side] += 1
            return jsonify({'status': 'Keep hands steady', 'error': True})
        else:
            ar_valid[hand_side].append(compute(convert(lmlist)))
            if len(ar_valid[hand_side]) > 20:
                # Calculate default ratios for this hand
                for i in range(len(ar_valid[hand_side][0])):
                    defaultratios[hand_side][i] = sum(ratio[i] for ratio in ar_valid[hand_side]) / len(ar_valid[hand_side])
                return jsonify({'status': 'Calibration complete for ' + ('right' if hand_side == 1 else 'left') + ' hand'})
            return jsonify({'status': 'Calibrating... Keep hands steady'})

    return jsonify({'status': 'No hand detected'})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)