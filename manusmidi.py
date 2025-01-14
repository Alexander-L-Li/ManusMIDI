from flask import Flask, render_template, Response
import cv2
from hand_tracking import handDetector  # Hand tracking logic from your script
import mediapipe as mp
import tinysoundfont
import time

app = Flask(__name__)

# Initialize video capture and hand tracking
cap = cv2.VideoCapture(0)
detector = handDetector()

# Initialize synth
synth = tinysoundfont.Synth()
sfid = synth.sfload("florestan-subset.sfo")
synth.program_select(0, sfid, 0, 2)  # Select instrument type
synth.start()
notes = [48, 50, 52, 53, 55, 57, 59, 60]  # C4 to C5
left_adjust = 0
right_adjust = 0

def generate_frames():
    global left_adjust, right_adjust
    pTime = 0

    while True:
        success, img = cap.read()
        if not success:
            break

        # Process frame with hand tracking
        img = detector.findHands(img)
        lmlist_right = detector.findPosition(img, 0)
        lmlist_left = detector.findPosition(img, 1)

        # Check hand positions and determine adjustments
        if lmlist_right and lmlist_left:
            right_hand_y = lmlist_right[0][2]
            left_hand_y = lmlist_left[0][2]
            if right_hand_y - left_hand_y >= 150:  # Right hand higher
                right_adjust = 1
                left_adjust = -1
            elif left_hand_y - right_hand_y >= 150:  # Left hand higher
                right_adjust = -1
                left_adjust = 1
            else:
                right_adjust = 0
                left_adjust = 0

        # Render FPS
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f"FPS: {int(fps)}", (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        # Encode and stream frame
        ret, buffer = cv2.imencode('.jpg', img)
        img_bytes = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + img_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)