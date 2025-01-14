# New version of the hand tracking module
import cv2
import mediapipe as mp
import time 

import tinysoundfont
import time
import threading

synth = tinysoundfont.Synth()
sfid = synth.sfload("florestan-subset.sfo")
synth.program_select(0, sfid, 0, 2) #select instrument type
synth.start()
notes = [48, 50, 52, 53, 55, 57, 59, 60] # C4 to C5
left_adjust = 0
right_adjust = 0

class handDetector():
  def __init__(self, mode = False, maxHands = 2, modelComplex = 1,detectionCon = .5, trackCon = .5):
    self.mode = mode
    self.maxHands = maxHands
    self.detectionCon = detectionCon
    self.trackCon = trackCon
    self.modelComplex = modelComplex

    self.mpHands =  mp.solutions.hands
    self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelComplex, self.detectionCon, self.trackCon)
    self.mpDraw = mp.solutions.drawing_utils
  
  def findHands(self, img, draw = True):
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    self.results = self.hands.process(imgRGB)

    if self.results.multi_hand_landmarks:
      for handLms in self.results.multi_hand_landmarks:
        if draw:
          self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
    return img
  def findPosition(self, img, handNo = 0, draw =  True):
    lmlist = []
    if (self.results.multi_hand_landmarks and handNo >= len(self.results.multi_hand_landmarks)):
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

def dist(point1, point2):
  return ((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)**0.5

def compute(vec):
  if (vec[0] == []):
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
  vec = [[] for i in range(21)]
  for x in list:
    vec[x[0]] = (x[1], x[2])
  return vec

def right(vec):
    inds = [3, 5, 9, 13, 17]
    gre = 0
    for i in range(len(inds) - 1):
        if (vec[inds[i]][0] < vec[inds[i+1]][0]):
            gre += 1
    if (gre <= 2):
        return 0
    else:
        return 1

def main():
  pTime = 0
  cTime = 0
  cap = cv2.VideoCapture(0)
  detector = handDetector()

  #CALIBRATION
  
  # last = []
  inconsistency = [0,0]
  ar_valid = [[],[]]
  alt = 0
  while (len(ar_valid[0]) <= 20 or len(ar_valid[1]) <= 20):
    success, img = cap.read()
    cv2.putText(img, "PLACE YOUR HANDS OUT FACING CAMERA FOR CALIBRATION", (200,500), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)
    img = detector.findHands(img)
    alt = 1-alt
    lmlist = detector.findPosition(img, alt)
    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)
    


    
    if (lmlist == []):
      cv2.imshow("Image", img)
      cv2.waitKey(1)
      continue
    if (len(ar_valid[right(convert(lmlist))]) == 21):
      cv2.imshow("Image", img)
      cv2.waitKey(1)
      continue

    if (len(ar_valid[right(convert(lmlist))]) == 0):
      ar_valid[right(convert(lmlist))].append(compute(convert(lmlist)))
    elif (computeDiff(compute(convert(lmlist)), ar_valid[right(convert(lmlist))][0]) > 0.1):
      inconsistency[right(convert(lmlist))] += 1
      cv2.putText(img, "DON'T MOVE YOUR HANDS!!", (200,600), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)
    else:
      ar_valid[right(convert(lmlist))].append(compute(convert(lmlist)))

    if (inconsistency[right(convert(lmlist))] >= 20):
      ar_valid[right(convert(lmlist))] = []
      inconsistency[right(convert(lmlist))] = 0

    cv2.imshow("Image", img)
    cv2.waitKey(1) 

    
  defaultratios = [[0 for i in range(len(ar_valid[0][0]))] for i in range(2)]
  for k in range(2):
    for i in range(len(ar_valid[k][0])):
        for j in range(len(ar_valid[k])):
          defaultratios[k][i] += ar_valid[k][j][i]
        defaultratios[k][i] /= len(ar_valid[k])

  last_fingers = None
  while (True):
    success, img = cap.read()
    img = detector.findHands(img)
    alt = 1-alt
    lmlist = detector.findPosition(img, alt)
    lmlist_right = detector.findPosition(img, 0)
    lmlist_left = detector.findPosition(img, 1)
    if (compute(convert(lmlist)) != []):
      right_hand_y = lmlist_right[0][2] if len(lmlist_right) > 0 else None
      left_hand_y = lmlist_left[0][2] if len(lmlist_left) > 0 else None
      if right_hand_y == None or left_hand_y == None:
        right_adjust = 0
        left_adjust = 0
        continue
      if right_hand_y-left_hand_y >= 150:  # Checks if right hand is higher
          right_adjust = 1
          left_adjust = -1
      if left_hand_y - right_hand_y >= 150:  # Checks if left hand is higher
          right_adjust = -1
          left_adjust = 1
      if abs(right_hand_y - left_hand_y) < 150:
          right_adjust = 0
          left_adjust = 0
      currentratios = compute(convert(lmlist))

      fingers = []
      flag = False
      for i in range(len(currentratios)):
        if currentratios[i]/defaultratios[right(convert(lmlist))][i] < 0.3:
          fingers.append(abs(3* (right(convert(lmlist)))-i) + 4*(1-right(convert(lmlist))))
      if (last_fingers != None):
        for finger in fingers:
          if (finger not in last_fingers[right(convert(lmlist))]):
            if finger >= 0 and finger <= 3:
              synth.noteon(0, notes[finger] + left_adjust, 100)

            if finger >= 4 and finger <= 7:
              synth.noteon(0, notes[finger] + right_adjust, 100)

        for finger in last_fingers[right(convert(lmlist))]:
          if finger not in fingers:
            if finger >= 0 and finger <= 3:
              synth.noteoff(0, notes[finger] + 1)
              synth.noteoff(0, notes[finger] - 1)
              synth.noteoff(0, notes[finger])

            if finger >= 4 and finger <= 7:
              synth.noteoff(0, notes[finger] + 1)
              synth.noteoff(0, notes[finger] - 1)
              synth.noteoff(0, notes[finger])

      if (last_fingers == None):
        last_fingers = [[],[]]
      last_fingers[right(convert(lmlist))] = fingers

      cv2.putText(img, str(fingers), (200,500), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)

    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)
    cv2.imshow("Image", img)
    cv2.waitKey(1)

if __name__ == "__main__":
  main()
