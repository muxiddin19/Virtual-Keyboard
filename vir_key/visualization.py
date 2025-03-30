"""
Visualization Module
------------------
Provides utilities for visualization and feedback.
"""

import cv2
import numpy as np
import time

class Visualizer:
    """Handles visualization for the virtual keyboard system"""
    
    def __init__(self, config=None):
        """
        Initialize visualizer
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Visualization settings
        self.show_debug = self.config.get('show_debug', True)
        self.show_fps = self.config.get('show_fps', True)
        self.show_hand = self.config.get('show_hand', True)
        self.show_keyboard = self.config.get('show_keyboard', True)
        self.show_depth = self.config.get('show_depth', True)
        
        # FPS calculation
        self.prev_frame_time = 0
        self.fps = 0
        
        # Colors
        self.colors = {
            'idle': (200, 200, 200),       # Gray
            'hovering': (255, 255, 0),     # Yellow
            'pressing': (0, 255, 0),       # Green
            'debug_text': (50, 220, 250),  # Light orange
            'hand': (0, 255, 255),         # Yellow
            'finger': (0, 255, 0),         # Green
            'projection': (255, 0, 255),   # Magenta
            'keyboard': (200, 200, 200),   # Gray
            'special_key': (100, 100, 255) # Light blue
        }
        
        # Debug overlay
        self.active_text_overlay = []
        self.text_overlay_timeout = time.time()
    
    def calculate_fps(self):
        """Calculate and update FPS"""
        current_time = time.time()
        time_diff = current_time - self.prev_frame_time
        
        if time_diff > 0:
            self.fps = 1 / time_diff
        
        self.prev_frame_time = current_time
    
    def add_text_overlay(self, text, duration=2.0):
        """
        Add text to temporary overlay
        
        Args:
            text: Text to display
            duration: How long to display text in seconds
        """
        self.active_text_overlay.append(text)
        self.text_overlay_timeout = time.time() + duration
    
    def render_frame(self, frame, keyboard=None, key_detector=None, 
                    hand_landmarks=None, depth_data=None):
        """
        Render the main visualization frame
        
        Args:
            frame: Original camera frame
            keyboard: KeyboardPlane object
            key_detector: KeyPressDetecter object
            hand_landmarks: Dictionary of hand landmark points
            depth_data: Depth estimation data
        
        Returns:
            Rendered frame
        """
        # Create a copy of the frame
        vis_frame = frame.copy()
        
        # Calculate FPS
        if self.show_fps:
            self.calculate_fps()
        
        # Draw keyboard if available
        if self.show_keyboard and keyboard:
            vis_frame = keyboard.render(vis_frame)
        
        # Draw hand landmarks if available
        if self.show_hand and hand_landmarks:
            self._draw_hand(vis_frame, hand_landmarks)
            
            # Draw projection ray from index finger if we have a keyboard
            if keyboard and 'index_tip' in hand_landmarks and 'index_direction' in hand_landmarks:
                self._draw_projection(vis_frame, hand_landmarks['index_tip'], 
                                     hand_landmarks['index_direction'], keyboard)
        
        # Draw key detection info
        if key_detector and key_detector.current_key:
            self._highlight_key(vis_frame, key_detector)
        
        # Draw depth visualization
        if self.show_depth and depth_data:
            self._visualize_depth(vis_frame, depth_data)
        
        # Add debug info
        if self.show_debug:
            self._add_debug_info(vis_frame, keyboard, key_detector, hand_landmarks)
        
        # Add text overlay if active
        current_time = time.time()
        if current_time < self.text_overlay_timeout:
            self._draw_text_overlay(vis_frame)
        else:
            self.active_text_overlay = []
        
        return vis_frame
    
    def _draw_hand(self, frame, hand_landmarks):
        """Draw hand landmarks and connections"""
        height, width = frame.shape[:2]
        
        # Draw landmark points
        for label, point in hand_landmarks.items():
            if isinstance(point, (list, tuple, np.ndarray)) and len(point) >= 2:
                # Convert normalized coordinates to pixel coordinates
                x, y = int(point[0] * width), int(point[1] * height)
                cv2.circle(frame, (x, y), 5, self.colors['hand'], -1)
                
                # Add label if in debug mode
                if self.show_debug:
                    cv2.putText(frame, label, (x + 10, y), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors['debug_text'], 1)
        
        # Draw connections (if connection data available)
        if 'connections' in hand_landmarks:
            for connection in hand_landmarks['connections']:
                if len(connection) == 2:
                    p1_label, p2_label = connection
                    if p1_label in hand_landmarks and p2_label in hand_landmarks:
                        p1 = hand_landmarks[p1_label]
                        p2 = hand_landmarks[p2_label]
                        
                        x1, y1 = int(p1[0] * width), int(p1[1] * height)
                        x2, y2 = int(p2[0] * width), int(p2[1] * height)
                        
                        cv2.line(frame, (x1, y1), (x2, y2), self.colors['hand'], 2)
    
    def _draw_projection(self, frame, finger_tip, finger_direction, keyboard):
        """Draw projection ray from finger to keyboard"""
        height, width = frame.shape[:2]
        
        # Convert normalized coordinates to pixel coordinates
        x1 = int(finger_tip[0] * width)
        y1 = int(finger_tip[1] * height)
        
        # Calculate the intersection with the keyboard plane
        intersection = keyboard._ray_plane_intersection(
            np.array(finger_tip), np.array(finger_direction)
        )
        
        if intersection is not None:
            # Convert intersection to pixel coordinates
            x2 = int(intersection[0] * width)
            y2 = int(intersection[1] * height)
            
            # Draw line from finger to intersection
            cv2.line(frame, (x1, y1), (x2, y2), self.colors['projection'], 2)
            
            # Draw intersection point
            cv2.circle(frame, (x2, y2), 8, self.colors['projection'], -1)
    
    def _highlight_key(self, frame, key_detector):
        """Highlight the currently active key"""
        height, width = frame.shape[:2]
        
        if key_detector.current_key:
            key = key_detector.current_key
            x1, y1, x2, y2 = key['rect']
            
            # Convert normalized coordinates to pixel coordinates
            pt1 = (int(x1 * width), int(y1 * height))
            pt2 = (int(x2 * width), int(y2 * height))
            
            # Select color based on state
            if key_detector.pressed_key and key['label'] == key_detector.pressed_key['label']:
                color = self.colors['pressing']
            elif key_detector.hover_key and key['label'] == key_detector.hover_key['label']:
                color = self.colors['hovering']
            else:
                color = self.colors['idle']
            
            # Draw highlighted key
            cv2.rectangle(frame, pt1, pt2, color, 2)
    
    def _visualize_depth(self, frame, depth_data):
        """Visualize depth estimation data"""
        # Implement based on the format of your depth data
        # Example: Draw a depth map in a corner of the frame
        if isinstance(depth_data, np.ndarray) and depth_data.ndim == 2:
            # Normalize depth map for visualization
            depth_vis = cv2.normalize(depth_data, None, 0, 255, cv2.NORM_MINMAX)
            depth_vis = depth_vis.astype(np.uint8)
            depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
            
            # Resize for visualization
            depth_vis = cv2.resize(depth_vis, (160, 120))
            
            # Place in the top-right corner
            h, w = depth_vis.shape[:2]
            frame[10:10+h, frame.shape[1]-w-10:frame.shape[1]-10] = depth_vis
    
    def _add_debug_info(self, frame, keyboard, key_detector, hand_landmarks):
        """Add debug information to the frame"""
        h, w = frame.shape[:2]
        
        # Add FPS
        if self.show_fps:
            cv2.putText(frame, f"FPS: {self.fps:.1f}", (10, 30), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['debug_text'], 2)
        
        # Add key detection info
        if key_detector:
            y_pos = 60
            
            # Current state
            state_text = "State: " + (key_detector.update(np.zeros(3), np.zeros(3))['state'] 
                                     if hand_landmarks else "No hand")
            cv2.putText(frame, state_text, (10, y_pos), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['debug_text'], 1)
            y_pos += 25
            
            # Current key
            key_text = "Key: " + (key_detector.current_key['label'] 
                                 if key_detector.current_key else "None")
            cv2.putText(frame, key_text, (10, y_pos), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['debug_text'], 1)
            y_pos += 25
            
            # Typing buffer (last few characters)
            buffer = key_detector.typing_buffer[-20:] if key_detector.typing_buffer else ""
            cv2.putText(frame, f"Buffer: {buffer}", (10, y_pos), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['debug_text'], 1)
        
        # Add keyboard info
        if keyboard:
            # Draw keyboard plane normal vector
            center = keyboard.plane_center
            normal = keyboard.plane_normal
            
            cx = int(center[0] * w)
            cy = int(center[1] * h)
            nx = int(cx + normal[0] * 50)
            ny = int(cy + normal[1] * 50)
            
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
            cv2.line(frame, (cx, cy), (nx, ny), (0, 0, 255), 2)
    
    def _draw_text_overlay(self, frame):
        """Draw the active text overlay"""
        h = frame.shape[0]
        
        y_pos = h - 50
        for text in reversed(self.active_text_overlay):
            cv2.putText(frame, text, (10, y_pos), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            y_pos -= 30
            
    def draw_typing_display(self, frame, text, cursor_pos=-1):
        """
        Draw a text input display
        
        Args:
            frame: Frame to draw on
            text: Text to display
            cursor_pos: Optional cursor position
        """
        h, w = frame.shape[:2]
        
        # Draw input box
        box_h = 60
        box_y = h - box_h - 20
        cv2.rectangle(frame, (20, box_y), (w - 20, box_y + box_h), (50, 50, 50), -1)
        cv2.rectangle(frame, (20, box_y), (w - 20, box_y + box_h), (150, 150, 150), 2)
        
        # Draw text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2
        
        # Get text size
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        
        # Draw text and cursor
        if cursor_pos < 0:
            # No cursor, just draw text
            text_x = 30
            text_y = box_y + (box_h + text_size[1]) // 2
            cv2.putText(frame, text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)
        else:
            # Draw text before cursor
            before_text = text[:cursor_pos]
            before_size = cv2.getTextSize(before_text, font, font_scale, thickness)[0][0] if before_text else 0
            
            text_x = 30
            text_y = box_y + (box_h + text_size[1]) // 2
            
            # Draw text
            cv2.putText(frame, text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)
            
            # Draw cursor
            cursor_x = text_x + before_size
            cursor_start = (cursor_x, text_y - text_size[1] - 5)
            cursor_end = (cursor_x, text_y + 5)
            cv2.line(frame, cursor_start, cursor_end, (255, 255, 255), 2)
        
        return frame