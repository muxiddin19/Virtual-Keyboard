#https://github.com/Titan-Spy/AI-Virtual-On-Screen-Keyboard-Using-Computer-Vision
import cv2
import mediapipe as mp
from time import sleep
import numpy as np
from pynput.keyboard import Controller
import pygame

# Initialize Pygame for sound
pygame.init()
pygame.mixer.init()
mc = pygame.mixer.Sound("kc.wav")

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam. Please check if the camera is connected and accessible.")
    exit()

cap.set(3, 1280)
cap.set(4, 720)

# Initialize MediaPipe Hands
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.8, min_tracking_confidence=0.5)
mpDraw = mp.solutions.drawing_utils

# Define keyboard layout
keys = [["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P"],
        ["A", "S", "D", "F", "G", "H", "J", "K", "L", ";"],
        ["Z", "X", "C", "V", "B", "N", "M", ",", ".", "/"]]
finalText = ""

# Initialize keyboard controller
keyboard = Controller()

# Button class with 2D and 3D positions
class Button:
    def __init__(self, pos_2d, pos_3d, text, size_2d=[85, 85], size_3d=[0.05, 0.05]):
        self.pos_2d = pos_2d
        self.pos_3d = pos_3d
        self.size_2d = size_2d
        self.size_3d = size_3d
        self.text = text

# Create button list with 2D and 3D positions
buttonList = []
for i in range(len(keys)):
    for j, key in enumerate(keys[i]):
        pos_2d = [100 * j + 50, 100 * i + 50]
        pos_3d = [(j - len(keys[i])/2) * 0.1, -i * 0.1, -0.5]  # For VR
        buttonList.append(Button(pos_2d, pos_3d, key))

# Function to draw buttons in 2D
def drawAll(img, buttonList):
    for button in buttonList:
        x, y = button.pos_2d
        w, h = button.size_2d
        cv2.rectangle(img, button.pos_2d, (x + w, y + h), (255, 0, 255), cv2.FILLED)
        cv2.putText(img, button.text, (x + 20, y + 65),
                    cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 4)
    return img

# Enhanced HandDetector class with occlusion handling
class EnhancedHandDetector:
    def __init__(self, detectionCon=0.8, maxHands=1):
        self.hands = mpHands.Hands(max_num_hands=maxHands, min_detection_confidence=detectionCon, min_tracking_confidence=0.5)
        self.mpDraw = mp.solutions.drawing_utils
        self.last_z_positions = {id: 0 for id in [8, 12, 16, 20]}  # For z-based typing detection

    def predict_occluded_finger(self, lmList, finger_tip_id):
        if finger_tip_id == 12:  # Middle finger
            base_id, mid_id = 9, 10
        elif finger_tip_id == 16:  # Ring finger
            base_id, mid_id = 13, 14
        elif finger_tip_id == 20:  # Pinky finger
            base_id, mid_id = 17, 18
        else:
            return lmList[finger_tip_id]

        if base_id >= len(lmList) or mid_id >= len(lmList):
            return lmList[finger_tip_id]

        base = np.array(lmList[base_id])
        mid = np.array(lmList[mid_id])
        vec = mid - base
        tip = mid + 1.5 * vec  # Extend vector to estimate tip
        return tip.tolist()

    def detect_key_press(self, landmark_id, z):
        current_z = z
        last_z = self.last_z_positions[landmark_id]
        self.last_z_positions[landmark_id] = current_z
        # Detect a downward motion (z decreases significantly)
        if last_z - current_z > 0.05:  # Adjusted for MediaPipe z-values (normalized)
            return True
        return False

    def find_hands_and_positions(self, img):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(imgRGB)
        lmList = []
        bboxInfo = None

        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                self.mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
                h, w, c = img.shape
                bbox = []
                for id, lm in enumerate(handLms.landmark):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cz = lm.z  # Normalized z-value
                    lmList.append([cx, cy, cz])
                    bbox.append([cx, cy])

                # Calculate bounding box
                if bbox:
                    x_min, y_min = np.min(bbox, axis=0)
                    x_max, y_max = np.max(bbox, axis=0)
                    bboxInfo = {"bbox": (x_min, y_min, x_max - x_min, y_max - y_min)}

                # Handle occlusion for middle, ring, and pinky fingers
                for finger_tip_id in [12, 16, 20]:
                    if lmList[finger_tip_id][2] > lmList[0][2]:  # Compare z with wrist
                        lmList[finger_tip_id] = self.predict_occluded_finger(lmList, finger_tip_id)

        return img, lmList, bboxInfo

# Initialize enhanced detector
enhanced_detector = EnhancedHandDetector(detectionCon=0.8)

# Main loop
while True:
    success, img = cap.read()
    if not success or img is None:
        print("Error: Failed to capture frame from webcam. Retrying...")
        continue

    # Detect hands and get landmarks
    img, lmList, bboxInfo = enhanced_detector.find_hands_and_positions(img)
    if img is None:
        print("Error: HandDetector failed to process the image. Skipping frame...")
        continue

    # Resize image
    img = cv2.resize(img, (1080, 720))
    img = drawAll(img, buttonList)

    if lmList:
        for button in buttonList:
            x, y = button.pos_2d
            w, h = button.size_2d

            # Check for each finger tip (index, middle, ring, pinky)
            for finger_tip_id in [8, 12, 16, 20]:
                tip_x, tip_y, tip_z = lmList[finger_tip_id]
                if x < tip_x < x + w and y < tip_y < y + h:
                    cv2.rectangle(img, (x - 5, y - 5), (x + w + 5, y + h + 5), (175, 0, 175), cv2.FILLED)
                    cv2.putText(img, button.text, (x + 20, y + 65),
                                cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 4)

                    # Use z-axis for key press detection
                    if enhanced_detector.detect_key_press(finger_tip_id, tip_z):
                        keyboard.press(button.text)
                        pygame.mixer.Sound.play(mc)
                        cv2.rectangle(img, button.pos_2d, (x + w, y + h), (0, 255, 0), cv2.FILLED)
                        cv2.putText(img, button.text, (x + 20, y + 65),
                                    cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 4)
                        finalText += button.text
                        sleep(0.15)
                        break

    # Display typed text
    cv2.rectangle(img, (50, 350), (700, 450), (175, 0, 175), cv2.FILLED)
    cv2.putText(img, finalText, (60, 430),
                cv2.FONT_HERSHEY_PLAIN, 5, (255, 255, 255), 5)

    # Show the image
    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()