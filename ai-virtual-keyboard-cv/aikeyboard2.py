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

# Define a virtual keyboard surface height
KEYBOARD_HEIGHT = 100  # Arbitrary value representing distance from camera

class Button():
    def __init__(self, pos, text, size=[85, 85]):
        self.pos = pos
        self.size = size
        self.text = text
        self.pressed = False  # Track if a button is currently pressed
        self.last_press_time = 0  # Track when the button was last pressed

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
    # Draw virtual keyboard surface plane
    cv2.rectangle(img, (40, 40), (1240, 400), (50, 50, 50), 2)
    
    for button in buttonList:
        x, y = button.pos
        w, h = button.size
        
        # Different color for pressed buttons
        if button.pressed:
            cv2.rectangle(img, button.pos, (x + w, y + h), (0, 255, 0), cv2.FILLED)
            # Add 3D effect for pressed key (move down slightly)
            y_offset = 5
            cv2.rectangle(img, (x, y+y_offset), (x + w, y + h+y_offset), (0, 150, 0), 2)
        else:
            cv2.rectangle(img, button.pos, (x + w, y + h), (255, 0, 255), cv2.FILLED)
            # Add 3D effect for key
            y_offset = 10
            cv2.rectangle(img, (x, y+y_offset), (x + w, y + h+y_offset), (150, 0, 150), 2)
            
        cv2.putText(img, button.text, (x + 20, y + 65), cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 4)
    return img

# Dictionary to store the previous y-position of each fingertip
prev_finger_y = {}
# Dictionary to store whether each fingertip is in a "pressing" state
finger_pressing = {}
# List of fingertip IDs to track
fingertips = [4, 8, 12, 16, 20]  # thumb, index, middle, ring, pinky

# Initialize fingertip tracking dictionaries
for finger_id in fingertips:
    prev_finger_y[finger_id] = 0
    finger_pressing[finger_id] = False

# Debounce time for key presses (seconds)
debounce_time = 0.2

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
    
    # Current time for debouncing
    current_time = cv2.getTickCount() / cv2.getTickFrequency()
    
    # Reset all pressed buttons if enough time has passed
    for button in buttonList:
        if button.pressed and current_time - button.last_press_time > 0.1:
            button.pressed = False
    
    if hands:
        # Process each detected hand
        for hand in hands:
            # Get landmarks for this hand
            lmList = hand["lmList"]
            
            if len(lmList) >= 21:  # Ensure we have all landmarks
                # Process each fingertip
                for finger_id in fingertips:
                    # Get current fingertip position
                    finger_x, finger_y = lmList[finger_id][:2]
                    
                    # Visual indicator for fingertip
                    cv2.circle(img, (finger_x, finger_y), 10, (0, 255, 255), cv2.FILLED)
                    
                    # Check if fingertip is over any button
                    for button in buttonList:
                        x, y = button.pos
                        w, h = button.size
                        
                        # Check if fingertip is over this button
                        if x < finger_x < x + w and y < finger_y < y + h:
                            # Highlight button when finger is over it
                            cv2.rectangle(img, (x - 5, y - 5), (x + w + 5, y + h + 5), 
                                         (175, 0, 175), cv2.FILLED)
                            cv2.putText(img, button.text, (x + 20, y + 65), 
                                       cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 4)
                            
                            # Check for vertical movement (key press)
                            if prev_finger_y[finger_id] > 0:  # Make sure we have a previous position
                                # Calculate vertical movement (positive = moving down)
                                y_movement = finger_y - prev_finger_y[finger_id]
                                
                                # Check if finger is moving down (pressing) and not already pressing
                                if y_movement > 8 and not finger_pressing[finger_id]:
                                    finger_pressing[finger_id] = True
                                    
                                    # Only trigger if debounce time has passed since last press
                                    if current_time - button.last_press_time > debounce_time:
                                        # Mark button as pressed
                                        button.pressed = True
                                        button.last_press_time = current_time
                                        
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
                                
                                # Check if finger is moving up (releasing)
                                elif y_movement < -5:
                                    finger_pressing[finger_id] = False
                    
                    # Update previous y position for this finger
                    prev_finger_y[finger_id] = finger_y
    
    # Display the typed text with a nicer UI
    cv2.rectangle(img, (50, 500), (1200, 600), (175, 0, 175), cv2.FILLED)
    cv2.putText(img, finalText, (60, 570), cv2.FONT_HERSHEY_PLAIN, 5, (255, 255, 255), 4)
    
    # Add instructions overlay
    cv2.rectangle(img, (50, 20), (1230, 100), (0, 0, 0), cv2.FILLED)
    cv2.putText(img, "Virtual Keyboard: Press with your fingers like a real keyboard", 
               (60, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 3)
    
    cv2.imshow("Virtual AI Keyboard", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()