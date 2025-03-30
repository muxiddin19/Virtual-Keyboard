#https://github.com/Titan-Spy/AI-Virtual-On-Screen-Keyboard-Using-Computer-Vision
import cv2
import mediapipe as mp
from time import sleep
import numpy as np
from pynput.keyboard import Controller, Key
import pygame
import time

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

# VirtualKeyboard class to manage state and logic
class VirtualKeyboard:
    def __init__(self):
        self.keys = [["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P"],
                     ["A", "S", "D", "F", "G", "H", "J", "K", "L", ";"],
                     ["Shift", "Z", "X", "C", "V", "B", "N", "M", ",", ".", "/"],
                     ["\u232b", " ", "Enter"]]
        self.finalText = ""
        self.is_shift_active = False
        self.keyboard = Controller()
        self.buttonList = self.create_buttons()
        self.detector = EnhancedHandDetector(detectionCon=0.8)

    def create_buttons(self):
        buttonList = []
        for i in range(len(self.keys)):
            for j, key in enumerate(self.keys[i]):
                # Adjusted spacing: 80 pixels
                if key == "\u232b":
                    pos_2d = [80 * j + 30, 80 * i + 30]
                    size_2d = [120, 60]
                elif key == " ":
                    pos_2d = [80 * j + 30, 80 * i + 30]
                    size_2d = [240, 60]
                elif key in ["Shift", "Enter"]:
                    pos_2d = [80 * j + 30, 80 * i + 30]
                    size_2d = [120, 60]
                else:
                    pos_2d = [80 * j + 30, 80 * i + 30]
                    size_2d = [60, 60]
                pos_3d = [(j - len(self.keys[i])/2) * 0.1, -i * 0.1, -0.5]
                buttonList.append(Button(pos_2d, pos_3d, key, size_2d=size_2d))
        return buttonList

    def draw_all(self, img):
        for button in self.buttonList:
            x, y = button.pos_2d
            w, h = button.size_2d
            if button.text in ["\u232b", " ", "Enter", "Shift"]:
                color = (255, 144, 0)
                if button.text == "Shift" and self.is_shift_active:
                    color = (0, 255, 0)
            else:
                color = (255, 0, 255)
            cv2.rectangle(img, button.pos_2d, (x + w, y + h), color, cv2.FILLED)
            cv2.putText(img, button.text, (x + 10, y + 45),
                        cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 3)
        shift_text = "Caps: ON" if self.is_shift_active else "Caps: OFF"
        cv2.putText(img, shift_text, (800, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
        return img

    def run(self):
        while True:
            success, img = cap.read()
            if not success or img is None:
                print("Error: Failed to capture frame from webcam. Retrying...")
                continue

            img, lmList, bboxInfo = self.detector.find_hands_and_positions(img)
            if img is None:
                print("Error: HandDetector failed to process the image. Skipping frame...")
                continue

            img = cv2.resize(img, (1080, 720))
            img = self.draw_all(img)

            if lmList:
                for button in self.buttonList:
                    x, y = button.pos_2d
                    w, h = button.size_2d

                    for finger_tip_id in [8, 12, 16, 20]:
                        tip_x, tip_y, tip_z = lmList[finger_tip_id]
                        # Debug: Print finger tip and button positions
                        print(f"Finger {finger_tip_id}: ({tip_x}, {tip_y}, {tip_z}), Button {button.text}: ({x}, {y}, {w}, {h})")
                        if x < tip_x < x + w and y < tip_y < y + h:
                            cv2.rectangle(img, (x - 5, y - 5), (x + w + 5, y + h + 5), (175, 0, 175), cv2.FILLED)
                            cv2.putText(img, button.text, (x + 10, y + 45),
                                        cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 3)

                            if self.detector.detect_key_press(finger_tip_id, tip_z):
                                print(f"Key pressed: {button.text}")
                                if button.text == "\u232b":
                                    if self.finalText:
                                        self.finalText = self.finalText[:-1]
                                        self.keyboard.press(Key.backspace)
                                        self.keyboard.release(Key.backspace)
                                elif button.text == " ":
                                    self.finalText += " "
                                    self.keyboard.press(Key.space)
                                    self.keyboard.release(Key.space)
                                elif button.text == "Enter":
                                    self.finalText += "\n"
                                    self.keyboard.press(Key.enter)
                                    self.keyboard.release(Key.enter)
                                elif button.text == "Shift":
                                    self.is_shift_active = not self.is_shift_active
                                    for btn in self.buttonList:
                                        if btn.text.isalpha():
                                            btn.text = btn.text.upper() if self.is_shift_active else btn.text.lower()
                                else:
                                    key_to_press = button.text.lower() if not self.is_shift_active else button.text.upper()
                                    self.keyboard.press(key_to_press)
                                    self.finalText += key_to_press
                                pygame.mixer.Sound.play(mc)
                                cv2.rectangle(img, button.pos_2d, (x + w, y + h), (0, 255, 0), cv2.FILLED)
                                cv2.putText(img, button.text, (x + 10, y + 45),
                                            cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 3)
                                break

            # Adjusted text display area to avoid overlap
            cv2.rectangle(img, (50, 500), (700, 600), (175, 0, 175), cv2.FILLED)
            lines = self.finalText.split("\n")
            for i, line in enumerate(lines[-3:]):
                cv2.putText(img, line, (60, 550 + i * 30),
                            cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)

            cv2.imshow("Image", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

# Button class with 2D and 3D positions
class Button:
    def __init__(self, pos_2d, pos_3d, text, size_2d=[60, 60], size_3d=[0.05, 0.05]):
        self.pos_2d = pos_2d
        self.pos_3d = pos_3d
        self.size_2d = size_2d
        self.size_3d = size_3d
        self.text = text

# Enhanced HandDetector class with occlusion handling
class EnhancedHandDetector:
    def __init__(self, detectionCon=0.8, maxHands=1):
        self.hands = mpHands.Hands(max_num_hands=maxHands, min_detection_confidence=detectionCon, min_tracking_confidence=0.5)
        self.mpDraw = mp.solutions.drawing_utils
        self.last_z_positions = {id: 0 for id in [8, 12, 16, 20]}
        self.last_press_time = {id: 0 for id in [8, 12, 16, 20]}

    def predict_occluded_finger(self, lmList, finger_tip_id):
        if finger_tip_id == 12:
            base_id, mid_id = 9, 10
        elif finger_tip_id == 16:
            base_id, mid_id = 13, 14
        elif finger_tip_id == 20:
            base_id, mid_id = 17, 18
        else:
            return lmList[finger_tip_id]

        if base_id >= len(lmList) or mid_id >= len(lmList):
            return lmList[finger_tip_id]

        base = np.array(lmList[base_id])
        mid = np.array(lmList[mid_id])
        vec = mid - base
        tip = mid + 1.5 * vec
        return tip.tolist()

    def detect_key_press(self, landmark_id, z):
        current_z = z
        last_z = self.last_z_positions[landmark_id]
        self.last_z_positions[landmark_id] = current_z
        # Lowered threshold for more sensitivity
        if last_z - current_z > 0.02:
            current_time = time.time()
            # Reduced debounce time for faster typing
            if current_time - self.last_press_time[landmark_id] > 0.1:
                self.last_press_time[landmark_id] = current_time
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
                    cz = lm.z
                    lmList.append([cx, cy, cz])
                    bbox.append([cx, cy])

                if bbox:
                    x_min, y_min = np.min(bbox, axis=0)
                    x_max, y_max = np.max(bbox, axis=0)
                    bboxInfo = {"bbox": (x_min, y_min, x_max - x_min, y_max - y_min)}

                for finger_tip_id in [12, 16, 20]:
                    if lmList[finger_tip_id][2] > lmList[0][2]:
                        lmList[finger_tip_id] = self.predict_occluded_finger(lmList, finger_tip_id)

        return img, lmList, bboxInfo

# Run the virtual keyboard
vk = VirtualKeyboard()
vk.run()

# Release resources
cap.release()
cv2.destroyAllWindows()