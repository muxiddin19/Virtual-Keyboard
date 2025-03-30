import cv2
import mediapipe as mp
import numpy as np
import time

class VirtualKeyboard:
    def __init__(self):
        # Video capture setup
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, 1280)
        self.cap.set(4, 720)

        # Hand tracking setup
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        # Keyboard layout
        self.keys = [
            ["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P"],
            ["A", "S", "D", "F", "G", "H", "J", "K", "L", "⌫"],
            ["Z", "X", "C", "V", "B", "N", "M", ",", ".", "Space"]
        ]

        # Key properties
        self.key_width = 100
        self.key_height = 80
        self.key_padding = 10
        self.start_x = 100
        self.start_y = 100

        # State tracking
        self.current_text = ""
        self.typed_text_history = []
        
        # Interaction tracking
        self.hand_states = {
            'left': {
                'last_click_time': 0,
                'last_key': None,
                'click_cooldown': 0.3,
                'deletion_mode': False,
                'deletion_start_time': 0,
                'deletion_count': 0
            },
            'right': {
                'last_click_time': 0,
                'last_key': None,
                'click_cooldown': 0.3,
                'deletion_mode': False,
                'deletion_start_time': 0,
                'deletion_count': 0
            }
        }

        # Create button list
        self.create_buttons()

    def create_buttons(self):
        self.buttons = []
        for i, row in enumerate(self.keys):
            for j, key in enumerate(row):
                x = self.start_x + j * (self.key_width + self.key_padding)
                y = self.start_y + i * (self.key_height + self.key_padding)
                
                # Adjust width for special keys
                width = self.key_width
                if key in ["Space", "BackSpace"]:
                    width = self.key_width * 1.5

                self.buttons.append({
                    'text': key,
                    'pos': (x, y),
                    'size': (width, self.key_height)
                })

    def get_hand_type(self, hand_landmarks):
        # Determine hand type based on thumb position
        wrist = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
        thumb_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
        
        # If thumb is on the left side of the wrist, it's likely a left hand
        return 'left' if thumb_tip.x < wrist.x else 'right'

    def is_hovering(self, landmark, button):
        x, y = button['pos']
        w, h = button['size']
        
        # Convert landmark to pixel coordinates
        lm_x = int(landmark.x * 1280)
        lm_y = int(landmark.y * 720)
        
        return (x < lm_x < x + w and y < lm_y < y + h)

    def check_click_gesture(self, hand_landmarks):
        # Advanced click detection
        index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        index_pip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_PIP]
        
        # Thumb tip
        thumb_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
        
        # Multiple click detection methods
        bent_index = index_tip.y < index_pip.y
        pinch_gesture = (abs(index_tip.x - thumb_tip.x) < 0.05 and 
                         abs(index_tip.y - thumb_tip.y) < 0.05)
        
        return bent_index or pinch_gesture

    def check_deletion_gesture(self, hand_landmarks):
        # Improved closed fist detection
        finger_joints = [
            (self.mp_hands.HandLandmark.INDEX_FINGER_TIP, self.mp_hands.HandLandmark.INDEX_FINGER_PIP),
            (self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP, self.mp_hands.HandLandmark.MIDDLE_FINGER_PIP),
            (self.mp_hands.HandLandmark.RING_FINGER_TIP, self.mp_hands.HandLandmark.RING_FINGER_PIP),
            (self.mp_hands.HandLandmark.PINKY_TIP, self.mp_hands.HandLandmark.PINKY_PIP)
        ]
        
        # Check if all finger tips are below their respective PIP joints
        all_fingers_bent = all(
            hand_landmarks.landmark[tip].y > hand_landmarks.landmark[pip].y
            for tip, pip in finger_joints
        )
        
        return all_fingers_bent

    def draw_keyboard(self, frame, current_landmarks=None):
        for button in self.buttons:
            x, y = button['pos']
            w, h = button['size']
            
            # Default button color
            color = (64, 64, 64)
            
            # Highlight hovering button
            if current_landmarks and current_landmarks.multi_hand_landmarks:
                for hand_landmarks in current_landmarks.multi_hand_landmarks:
                    if self.is_hovering(hand_landmarks.landmark[8], button):
                        color = (100, 100, 100)
                        break
            
            # Draw button rectangle
            cv2.rectangle(frame, (int(x), int(y)), 
                         (int(x + w), int(y + h)), 
                         color, cv2.FILLED)
            
            # Draw button text
            cv2.putText(frame, button['text'], 
                        (int(x + 20), int(y + h - 20)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, 
                        (255, 255, 255), 2)
        
        return frame

    def process_hand_interaction(self, hand_landmarks):
        # Determine hand type
        hand_type = self.get_hand_type(hand_landmarks)
        current_time = time.time()
        hand_state = self.hand_states[hand_type]

        # Deletion gesture check with improved behavior
        if self.check_deletion_gesture(hand_landmarks):
            if not hand_state['deletion_mode']:
                hand_state['deletion_mode'] = True
                hand_state['deletion_start_time'] = current_time
                hand_state['deletion_count'] = 0
            
            # Progressive deletion with timing
            if current_time - hand_state['deletion_start_time'] > 0.3:
                # Increase deletion speed over time
                deletion_interval = max(0.1, 0.3 - (hand_state['deletion_count'] * 0.05))
                
                if current_time - hand_state.get('last_deletion_time', 0) > deletion_interval:
                    if self.current_text:
                        self.current_text = self.current_text[:-1]
                        if self.typed_text_history:
                            self.typed_text_history.pop()
                        
                        hand_state['last_deletion_time'] = current_time
                        hand_state['deletion_count'] += 1
        else:
            hand_state['deletion_mode'] = False
            hand_state['deletion_count'] = 0

        # Normal key press detection
        for button in self.buttons:
            # Check if index finger tip is hovering over a button
            if self.is_hovering(hand_landmarks.landmark[8], button):
                # Check if index finger is bent (clicking)
                if self.check_click_gesture(hand_landmarks):
                    # Cooldown to prevent rapid typing
                    if (current_time - hand_state['last_click_time'] > hand_state['click_cooldown'] or 
                        button['text'] != hand_state['last_key']):
                        
                        key = button['text']
                        if key == "Space":
                            self.current_text += " "
                        elif key == "⌫":
                            if self.current_text:
                                self.current_text = self.current_text[:-1]
                                if self.typed_text_history:
                                    self.typed_text_history.pop()
                        else:
                            self.current_text += key
                            self.typed_text_history.append(key)
                        
                        # Update hand state
                        hand_state['last_click_time'] = current_time
                        hand_state['last_key'] = key

    def run(self):
        while True:
            # Read frame
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Flip frame horizontally
            frame = cv2.flip(frame, 1)
            
            # Detect hand landmarks
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            current_landmarks = self.hands.process(rgb_frame)
            
            # Draw keyboard first (in case no hands are detected)
            frame = self.draw_keyboard(frame)
            
            # Draw hand tracking and process interactions
            if current_landmarks.multi_hand_landmarks:
                # Draw hand landmarks
                for hand_landmarks in current_landmarks.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        frame, 
                        hand_landmarks, 
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing_styles.get_default_hand_landmarks_style(),
                        self.mp_drawing_styles.get_default_hand_connections_style()
                    )
                
                # Process each hand's interaction
                for hand_landmarks in current_landmarks.multi_hand_landmarks:
                    self.process_hand_interaction(hand_landmarks)
            
            # Display typed text
            cv2.rectangle(frame, (50, 600), (1230, 700), (64, 64, 64), cv2.FILLED)
            cv2.putText(frame, self.current_text, (60, 670), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, 
                        (255, 255, 255), 3)
            
            # Add instruction text
            instruction_text = "Hover & bend index finger to type. Closed fist to delete. Press 'q' to quit."
            cv2.putText(frame, instruction_text, (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                        (255, 255, 255), 2)
            
            # Show frame
            cv2.imshow('Virtual Keyboard', frame)
            
            # Exit condition
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()

# Run the virtual keyboard
if __name__ == "__main__":
    keyboard = VirtualKeyboard()
    keyboard.run()