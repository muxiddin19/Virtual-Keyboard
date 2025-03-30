import cv2
from cvzone.HandTrackingModule import HandDetector
from time import sleep
import numpy as np
import cvzone
from pynput.keyboard import Controller
import math


def detect_key_press(lmList, button, prev_finger_height):
    """Detect if a key is pressed by tracking vertical finger movement"""
    x, y = button.pos
    w, h = button.size
    
    # Check if finger is above the key
    if x < lmList[8][0] < x + w and y < lmList[8][1] < y + h:
        # Get current finger height (z coordinate is normalized in MediaPipe)
        # Lower values mean finger is closer to camera
        current_height = lmList[8][2]
        
        # Calculate the height change (positive value means pressing down)
        height_change = prev_finger_height - current_height
        
        # Return if a press occurred and the new height
        return height_change > 0.05, current_height
    
    return False, prev_finger_height

def get_active_fingertip(lmList, buttonList):
    """Find which fingertip is actively pressing a key"""
    # Fingertip indices: thumb(4), index(8), middle(12), ring(16), pinky(20)
    fingertips = [4, 8, 12, 16, 20]
    
    for finger_id in fingertips:
        # Skip thumb for typing (uncommon to type with thumb except for space)
        if finger_id == 4:
            continue
            
        finger_pos = lmList[finger_id]
        
        for button in buttonList:
            x, y = button.pos
            w, h = button.size
            
            if x < finger_pos[0] < x + w and y < finger_pos[1] < y + h:
                return finger_id, button
    
    return None, None

def visualize_finger_height(img, lmList, finger_id):
    """Add visual indicator of finger height above keyboard"""
    if finger_id is None:
        return img
        
    # Get finger position
    x, y = lmList[finger_id][0], lmList[finger_id][1]
    z = lmList[finger_id][2]
    
    # Normalize z value for visualization (lower value = closer to camera)
    indicator_size = int(30 * (1.0 - z))  # Size increases as finger gets closer
    
    # Draw indicator circle
    color = (0, 255, 0) if z < 0.5 else (0, 255, 255)  # Green when close to pressing
    cv2.circle(img, (x, y-30), indicator_size, color, cv2.FILLED)
    
    return img

def detect_key_press(lmList, button, prev_finger_height):
    """Detect if a key is pressed by tracking vertical finger movement"""
    x, y = button.pos
    w, h = button.size
    
    # Check if finger is above the key
    if x < lmList[8][0] < x + w and y < lmList[8][1] < y + h:
        # Get current finger height (z coordinate is normalized in MediaPipe)
        # Lower values mean finger is closer to camera
        current_height = lmList[8][2]
        
        # Calculate the height change (positive value means pressing down)
        height_change = prev_finger_height - current_height
        
        # Return if a press occurred and the new height
        return height_change > 0.05, current_height
    
    return False, prev_finger_height



cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# Increase detection confidence for more stability
detector = HandDetector(detectionCon=0.8, maxHands=2)

keys = [
    ["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P"],
    ["A", "S", "D", "F", "G", "H", "J", "K", "L", ";"],
    ["Z", "X", "C", "V", "B", "N", "M", ",", ".", "/"],
    ["Space", "Backspace"]
]
finalText = ""

keyboard = Controller()

class Button():
    def __init__(self, pos, text, size=[85, 85]):
        self.pos = pos
        self.size = size
        self.text = text
        self.clicked = False  # Track if a button is currently pressed

buttonList = []
for i in range(len(keys)):
    for j, key in enumerate(keys[i]):
        if key == "Space":
            buttonList.append(Button([100 * j + 50, 100 * i + 50], key, size=[500, 85]))
        elif key == "Backspace":
            buttonList.append(Button([700, 350], key, size=[150, 85]))
        else:
            buttonList.append(Button([100 * j + 50, 100 * i + 50], key))

def drawAll(img, buttonList):
    for button in buttonList:
        x, y = button.pos
        w, h = button.size
        cvzone.cornerRect(img, (x, y, w, h), 20, rt=0)
        
        # Different color for clicked buttons
        if button.clicked:
            cv2.rectangle(img, button.pos, (x + w, y + h), (0, 255, 0), cv2.FILLED)
        else:
            cv2.rectangle(img, button.pos, (x + w, y + h), (255, 0, 255), cv2.FILLED)
            
        cv2.putText(img, button.text, (x + 20, y + 65), cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 4)
    return img

# Track timing for debouncing
last_key_press_time = 0
key_press_delay = 0.3  # seconds between key presses

while True:
    success, img = cap.read()
    
    if not success or img is None:
        print("Error: Failed to capture image. Check your camera.")
        continue
    
    # Mirror the image for more intuitive interaction
    img = cv2.flip(img, 1)
    
    # Find hands
    hands, img = detector.findHands(img)
    
    # Draw the keyboard
    img = drawAll(img, buttonList)
    
    # Reset all button clicked states
    for button in buttonList:
        button.clicked = False
    
    # Current time for debouncing
    current_time = cv2.getTickCount() / cv2.getTickFrequency()
    
    if hands:
        # Process each detected hand
        for hand in hands:
            # Get landmarks for this hand
            lmList = hand["lmList"]
            
            # Process for typing - check index and middle fingers (landmarks 8 and 12)
            if len(lmList) >= 21:  # Ensure we have all landmarks
                # Check index finger position
                index_finger_tip = lmList[8][:2]  # x, y coordinates
                middle_finger_tip = lmList[12][:2]
                
                # Calculate distance between index and middle finger
                distance = math.dist(index_finger_tip, middle_finger_tip)
                
                # Check if index finger is over any button
                for button in buttonList:
                    x, y = button.pos
                    w, h = button.size
                    
                    # Check if index finger is over this button
                    if x < index_finger_tip[0] < x + w and y < index_finger_tip[1] < y + h:
                        # Highlight button when finger is over it
                        cv2.rectangle(img, (x - 5, y - 5), (x + w + 5, y + h + 5), 
                                     (175, 0, 175), cv2.FILLED)
                        cv2.putText(img, button.text, (x + 20, y + 65), 
                                   cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 4)
                        
                        # Check if fingers are pinched (typing gesture)
                        if distance < 40 and current_time - last_key_press_time > key_press_delay:
                            # Mark this button as clicked
                            button.clicked = True
                            
                            # Process the key
                            if button.text == "Space":
                                finalText += " "
                                keyboard.press(" ")
                                keyboard.release(" ")
                            elif button.text == "Backspace":
                                if len(finalText) > 0:
                                    finalText = finalText[:-1]
                                    keyboard.press("\b")
                                    keyboard.release("\b")
                            else:
                                # Handle all other keys
                                key_char = button.text.lower() if len(button.text) == 1 else button.text
                                finalText += key_char
                                keyboard.press(key_char)
                                keyboard.release(key_char)
                            
                            # Update last key press time for debouncing
                            last_key_press_time = current_time
                            
                            # Visual feedback - flash green when pressed
                            cv2.rectangle(img, button.pos, (x + w, y + h), 
                                         (0, 255, 0), cv2.FILLED)
                            cv2.putText(img, button.text, (x + 20, y + 65), 
                                       cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 4)
    
    # Display the typed text
    cv2.rectangle(img, (50, 500), (1200, 600), (175, 0, 175), cv2.FILLED)
    cv2.putText(img, finalText, (60, 570), cv2.FONT_HERSHEY_PLAIN, 5, (255, 255, 255), 4)
    
    # Add instructions overlay
    cv2.rectangle(img, (50, 20), (1230, 100), (0, 0, 0), cv2.FILLED)
    cv2.putText(img, "Virtual Keyboard: Pinch index and middle fingers to type", 
               (60, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 3)
    
    cv2.imshow("Virtual AI Keyboard", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()