"""
Hand Tracking Module
-------------------
Uses MediaPipe to detect and track hands in camera input.
"""

import cv2
import mediapipe as mp
import numpy as np

class HandTracker:
    """Handles hand detection and tracking using MediaPipe"""
    
    def __init__(self, static_image_mode=False, max_num_hands=2, 
                 min_detection_confidence=0.5, min_tracking_confidence=0.5):
        """Initialize the hand tracker with MediaPipe Hands"""
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Set up MediaPipe Hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        # Initialize tracking state
        self.results = None
        self.hand_landmarks = []
        self.handedness = []
        
        # Index of important landmarks
        self.LANDMARK_INDICES = {
            'WRIST': 0,
            'THUMB_TIP': 4,
            'INDEX_FINGER_MCP': 5,
            'INDEX_FINGER_TIP': 8,
            'MIDDLE_FINGER_TIP': 12,
            'RING_FINGER_TIP': 16,
            'PINKY_TIP': 20
        }
    
    def process_frame(self, frame):
        """
        Process a video frame to detect and track hands.
        
        Args:
            frame: RGB image frame
            
        Returns:
            List of processed hand landmarks or None if no hands detected
        """
        if frame is None:
            return None
        
        # Convert to RGB (MediaPipe requires RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame with MediaPipe
        self.results = self.hands.process(rgb_frame)
        
        # Reset state
        self.hand_landmarks = []
        self.handedness = []
        
        # Check if hands were detected
        if self.results.multi_hand_landmarks:
            frame_height, frame_width, _ = frame.shape
            
            # Process each detected hand
            for idx, hand_landmarks in enumerate(self.results.multi_hand_landmarks):
                # Get handedness (left/right)
                handedness = self.results.multi_handedness[idx].classification[0].label
                
                # Normalize landmarks and add to our list
                normalized_landmarks = self._normalize_landmarks(
                    hand_landmarks, frame_width, frame_height)
                
                self.hand_landmarks.append(normalized_landmarks)
                self.handedness.append(handedness)
            
            return {
                'landmarks': self.hand_landmarks,
                'handedness': self.handedness
            }
        
        return None
    
    def get_finger_positions(self, landmarks, finger_name=None):
        """Get specific finger positions from landmarks"""
        if not landmarks:
            return {}
        
        positions = {}
        for name, idx in self.LANDMARK_INDICES.items():
            if finger_name is None or finger_name in name:
                positions[name] = landmarks[idx]
        
        return positions
    
    def get_index_finger_tip(self, hand_idx=0):
        """Get the position of the index finger tip"""
        if not self.hand_landmarks or hand_idx >= len(self.hand_landmarks):
            return None
        
        landmarks = self.hand_landmarks[hand_idx]
        idx = self.LANDMARK_INDICES['INDEX_FINGER_TIP']
        return landmarks[idx]
    
    def draw_landmarks(self, frame):
        """Draw the hand landmarks on the frame"""
        if self.results and self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
        return frame
    
    def _normalize_landmarks(self, hand_landmarks, frame_width, frame_height):
        """
        Convert landmarks from MediaPipe format to normalized 3D coordinates
        
        Returns:
            List of 21 [x, y, z] coordinates (normalized)
        """
        normalized_landmarks = []
        
        for landmark in hand_landmarks.landmark:
            # Convert to pixel coordinates
            px = landmark.x * frame_width
            py = landmark.y * frame_height
            pz = landmark.z * frame_width  # Use width as depth scale
            
            # Normalize to 0-1 range
            x = landmark.x  # Already normalized by MediaPipe
            y = landmark.y  # Already normalized by MediaPipe
            z = landmark.z  # Depth, negative is toward camera
            
            normalized_landmarks.append([x, y, z, px, py, pz])
            
        return normalized_landmarks
    
    def get_hand_direction(self, hand_idx=0):
        """
        Determine if the hand is facing up, down, left or right
        
        Returns:
            Direction as string: "up", "down", "left", "right", or None
        """
        if not self.hand_landmarks or hand_idx >= len(self.hand_landmarks):
            return None
            
        # Get landmarks
        landmarks = self.hand_landmarks[hand_idx]
        
        # Get wrist and middle finger positions
        wrist = landmarks[self.LANDMARK_INDICES['WRIST']]
        middle_tip = landmarks[self.LANDMARK_INDICES['MIDDLE_FINGER_TIP']]
        
        # Calculate direction
        dx = middle_tip[0] - wrist[0]
        dy = middle_tip[1] - wrist[1]
        
        # Determine primary direction
        if abs(dx) > abs(dy):
            # Horizontal movement dominates
            return "right" if dx > 0 else "left"
        else:
            # Vertical movement dominates
            return "down" if dy > 0 else "up"
    
    def close(self):
        """Clean up resources"""
        self.hands.close()

# Example usage
if __name__ == "__main__":
    # Simple test to verify the module works
    cap = cv2.VideoCapture(0)
    tracker = HandTracker()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Flip the frame horizontally
        frame = cv2.flip(frame, 1)
        
        # Track hands
        hand_data = tracker.process_frame(frame)
        
        # Draw landmarks
        frame = tracker.draw_landmarks(frame)
        
        # Display additional info
        if hand_data:
            cv2.putText(frame, f"Hands: {len(hand_data['landmarks'])}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            for i, handedness in enumerate(hand_data['handedness']):
                cv2.putText(frame, f"{handedness} hand", 
                           (10, 70 + 40*i), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Show the frame
        cv2.imshow('Hand Tracking Test', frame)
        
        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    tracker.close()