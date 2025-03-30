"""
Key Detection Module
-------------------
Implements logic for detecting and processing key presses.
"""

import numpy as np
import time

class KeyDetector: #  KeyPressDetecter:
    """Detects and processes key presses based on finger position"""
    
    def __init__(self, keyboard=None, config=None):
        """
        Initialize key press detector
        
        Args:
            keyboard: KeyboardPlane object
            config: Configuration dictionary
        """
        self.keyboard = keyboard
        self.config = config or {}
        
        # Detection parameters
        self.hover_threshold = self.config.get('hover_threshold', 0.03)  # Distance to consider hovering
        self.press_threshold = self.config.get('press_threshold', 0.01)  # Distance to consider pressing
        self.press_duration = self.config.get('press_duration', 0.5)  # Seconds to hold for a press
        
        # State tracking
        self.current_key = None
        self.hover_key = None
        self.hover_start_time = 0
        self.pressed_key = None
        self.last_press_time = 0
        self.press_cooldown = self.config.get('press_cooldown', 0.5)  # Seconds between presses
        
        # For tracking key changes
        self.key_events = []
        self.typing_buffer = ""
    
    def update(self, finger_position, finger_direction):
        """
        Update key detection state based on finger position
        
        Args:
            finger_position: 3D position of the finger tip
            finger_direction: 3D direction vector of the finger
        
        Returns:
            Dict containing detection state and any key events
        """
        if self.keyboard is None:
            return {'state': 'no_keyboard', 'events': []}
        
        # Clear previous events
        self.key_events = []
        
        # Get key at current finger position
        intersected_key = self.keyboard.get_key_at_intersection(finger_position, finger_direction)
        
        # Calculate distance from finger to keyboard plane
        distance = self._calculate_distance_to_plane(finger_position)
        
        # Process hover state
        if intersected_key is not None:
            # Check if we're hovering over a new key
            if self.hover_key is None or intersected_key['label'] != self.hover_key['label']:
                self.hover_key = intersected_key
                self.hover_start_time = time.time()
                self.key_events.append(('hover', intersected_key['label']))
            
            # Check for press conditions
            if distance <= self.press_threshold:
                press_time = time.time()
                if (press_time - self.last_press_time) > self.press_cooldown:
                    # Register a new key press
                    self.pressed_key = intersected_key
                    self.last_press_time = press_time
                    self._process_key_press(intersected_key['label'])
            
            self.current_key = intersected_key
        else:
            # Not over any key
            if self.hover_key is not None:
                self.key_events.append(('hover_end', self.hover_key['label']))
                self.hover_key = None
            self.current_key = None
        
        # Return the current state
        return {
            'state': 'pressing' if distance <= self.press_threshold else 
                     'hovering' if self.hover_key is not None else 'idle',
            'current_key': self.current_key['label'] if self.current_key else None,
            'distance': distance,
            'events': self.key_events,
            'buffer': self.typing_buffer
        }
    
    def _calculate_distance_to_plane(self, finger_position):
        """Calculate the distance from finger to keyboard plane"""
        finger_position = np.array(finger_position)
        plane_normal = self.keyboard.plane_normal
        plane_point = self.keyboard.plane_center
        
        # Calculate signed distance
        v = finger_position - plane_point
        distance = abs(np.dot(v, plane_normal))
        return distance
    
    def _process_key_press(self, key_label):
        """Process a detected key press"""
        self.key_events.append(('press', key_label))
        
        # Update typing buffer
        if key_label == 'BACKSPACE':
            self.typing_buffer = self.typing_buffer[:-1]
        elif key_label == 'SPACE':
            self.typing_buffer += ' '
        elif key_label == 'ENTER':
            self.typing_buffer += '\n'
        elif len(key_label) == 1:  # Regular character
            self.typing_buffer += key_label
    
    def reset(self):
        """Reset the detector state"""
        self.current_key = None
        self.hover_key = None
        self.hover_start_time = 0
        self.pressed_key = None
        self.key_events = []
        
    def clear_buffer(self):
        """Clear the typing buffer"""
        previous_buffer = self.typing_buffer
        self.typing_buffer = ""
        return previous_buffer