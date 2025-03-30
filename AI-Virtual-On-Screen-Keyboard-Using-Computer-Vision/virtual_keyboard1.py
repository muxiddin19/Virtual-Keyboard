import cv2
import mediapipe as mp
import numpy as np

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
            max_num_hands=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
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
        self.click_delay = 0
        self.last_key = None
        self.current_landmarks = None

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
                if key in ["Space", "⌫"]:
                    width = self.key_width * 1.5

                self.buttons.append({
                    'text': key,
                    'pos': (x, y),
                    'size': (width, self.key_height)
                })

    def get_hand_landmarks(self, frame):
        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame and find hand landmarks
        results = self.hands.process(rgb_frame)
        
        return results

    def draw_hand_tracking(self, frame, results):
        # If hand landmarks are detected, draw them
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks
                self.mp_drawing.draw_landmarks(
                    frame, 
                    hand_landmarks, 
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
        return frame

    def is_hovering(self, landmark, button):
        # More precise hovering detection
        x, y = button['pos']
        w, h = button['size']
        
        # Convert landmark to pixel coordinates
        lm_x = int(landmark.x * 1280)
        lm_y = int(landmark.y * 720)
        
        return (x < lm_x < x + w and y < lm_y < y + h)

    def check_click(self, landmarks):
        # More robust click detection
        # Check if index finger is bent and other fingers are extended
        index_tip = landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        index_pip = landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_PIP]
        
        # More precise bent finger detection
        return index_tip.y < index_pip.y

    def draw_keyboard(self, frame):
        for button in self.buttons:
            x, y = button['pos']
            w, h = button['size']
            
            # Default button color
            color = (64, 64, 64)
            
            # Highlight hovering button if hand is detected
            if self.current_landmarks and self.current_landmarks.multi_hand_landmarks:
                if self.is_hovering(
                    self.current_landmarks.multi_hand_landmarks[0].landmark[8], 
                    button
                ):
                    color = (100, 100, 100)
            
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

    def process_key_press(self):
        # Check if hand landmarks are detected
        if not self.current_landmarks or not self.current_landmarks.multi_hand_landmarks:
            return

        # Get the first hand's landmarks
        hand_landmarks = self.current_landmarks.multi_hand_landmarks[0]

        for button in self.buttons:
            # Check if index finger tip is hovering over a button
            if self.is_hovering(hand_landmarks.landmark[8], button):
                # Check if index finger is bent (clicking)
                if self.check_click(hand_landmarks):
                    key = button['text']
                    
                    # Add click delay to prevent rapid typing
                    if key != self.last_key or self.click_delay > 10:
                        if key == "Space":
                            self.current_text += " "
                        elif key == "⌫":
                            self.current_text = self.current_text[:-1]
                        else:
                            self.current_text += key
                        
                        self.last_key = key
                        self.click_delay = 0
                    
                    self.click_delay += 1

    def run(self):
        while True:
            # Read frame
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Flip frame horizontally
            frame = cv2.flip(frame, 1)
            
            # Detect hand landmarks
            self.current_landmarks = self.get_hand_landmarks(frame)
            
            # Draw hand tracking
            if self.current_landmarks.multi_hand_landmarks:
                frame = self.draw_hand_tracking(frame, self.current_landmarks)
            
            # Draw keyboard
            frame = self.draw_keyboard(frame)
            
            # Process key press
            self.process_key_press()
            
            # Display typed text
            cv2.rectangle(frame, (50, 600), (1230, 700), (64, 64, 64), cv2.FILLED)
            cv2.putText(frame, self.current_text, (60, 670), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, 
                        (255, 255, 255), 3)
            
            # Add instruction text
            instruction_text = "Hover over keys, bend index finger to type. Press 'q' to quit."
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