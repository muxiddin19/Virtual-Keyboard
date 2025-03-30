"""
Calibration Module
----------------
Implements calibration procedures for the virtual keyboard.
"""

import cv2
import numpy as np
import time


class Calibrator: #CalibrationManager:
    """Manages the calibration process for the virtual keyboard system"""
    
    def __init__(self, config=None):
        """
        Initialize calibration manager
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # State
        self.is_calibrating = False
        self.current_step = 0
        self.steps_completed = 0
        self.calibration_points = []
        self.step_start_time = 0
        self.current_point = None
        
        # Calibration settings
        self.total_steps = self.config.get('calibration_steps', 4)
        self.point_collect_time = self.config.get('point_collect_time', 2.0)  # Seconds
        self.calibration_timeout = self.config.get('calibration_timeout', 30.0)  # Seconds
        self.calibration_start_time = 0
        
        # Reference points for keyboard corners (normalized screen coordinates)
        self.reference_points = [
            (0.3, 0.3),  # Top-left
            (0.7, 0.3),  # Top-right
            (0.7, 0.7),  # Bottom-right
            (0.3, 0.7)   # Bottom-left
        ]
        
        # Depth values for different calibration variations
        self.depth_variations = {
            'flat': [0.5, 0.5, 0.5, 0.5],  # Same depth for all corners
            'angled_front': [0.45, 0.45, 0.55, 0.55],  # Front edge closer
            'angled_side': [0.45, 0.55, 0.55, 0.45]    # Left edge closer
        }
        
        # Default to flat
        self.current_depth_config = 'flat'
        
        # Results
        self.calibration_data = None
    
    def start_calibration(self, depth_config='flat'):
        """
        Start the calibration process
        
        Args:
            depth_config: Keyboard depth configuration ('flat', 'angled_front', 'angled_side')
        """
        self.is_calibrating = True
        self.current_step = 0
        self.steps_completed = 0
        self.calibration_points = []
        self.current_depth_config = depth_config
        self.calibration_data = None
        self.calibration_start_time = time.time()
        self.step_start_time = time.time()
        self.current_point = self.reference_points[0]
    
    def update(self, hand_landmarks=None):
        """
        Update calibration process
        
        Args:
            hand_landmarks: Hand landmarks from hand tracking
            
        Returns:
            Dict with calibration state and progress
        """
        if not self.is_calibrating:
            return {
                'status': 'inactive',
                'progress': 0,
                'message': 'Calibration inactive'
            }
        
        # Check for timeout
        current_time = time.time()
        if current_time - self.calibration_start_time > self.calibration_timeout:
            self.is_calibrating = False
            return {
                'status': 'failed',
                'progress': 0,
                'message': 'Calibration timed out'
            }
        
        # Process current step
        if current_time - self.step_start_time >= self.point_collect_time:
            # Step time completed
            self.steps_completed += 1
            
            # Move to next step or finish
            if self.steps_completed >= self.total_steps:
                # Calibration complete
                self._finalize_calibration()
                self.is_calibrating = False
                return {
                    'status': 'complete',
                    'progress': 1.0,
                    'message': 'Calibration complete',
                    'data': self.calibration_data
                }
            else:
                # Next step
                self.current_step = self.steps_completed
                self.step_start_time = current_time
                self.current_point = self.reference_points[self.current_step % len(self.reference_points)]
        
        # Collect hand position if available
        if hand_landmarks and 'index_tip' in hand_landmarks:
            index_tip = hand_landmarks['index_tip']
            
            # Add depth information from our configuration
            depth = self.depth_variations[self.current_depth_config][self.current_step % len(self.reference_points)]
            point_3d = [index_tip[0], index_tip[1], depth]
            
            # Update current collection point
            self.calibration_points.append(point_3d)
        
        # Calculate progress
        step_progress = (current_time - self.step_start_time) / self.point_collect_time
        overall_progress = (self.steps_completed + step_progress) / self.total_steps
        
        return {
            'status': 'in_progress',
            'progress': overall_progress,
            'current_step': self.current_step,
            'steps_completed': self.steps_completed,
            'total_steps': self.total_steps,
            'current_point': self.current_point,
            'message': f'Calibrating point {self.current_step+1} of {self.total_steps}',
            'time_remaining': self.point_collect_time - (current_time - self.step_start_time)
        }
    
    def _finalize_calibration(self):
        """Process collected calibration data"""
        if len(self.calibration_points) < self.total_steps:
            # Not enough points
            self.calibration_data = None
            return False
        
        # Average the collected points for each step
        processed_points = []
        points_per_step = len(self.calibration_points) // self.total_steps
        
        for i in range(self.total_steps):
            start_idx = i * points_per_step
            end_idx = start_idx + points_per_step
            step_points = self.calibration_points[start_idx:end_idx]
            
            if step_points:
                # Calculate average position for this step
                avg_point = np.mean(step_points, axis=0)
                processed_points.append(avg_point)
        
        # Create calibration data
        self.calibration_data = {
            'points': processed_points,
            'reference_points': self.reference_points[:len(processed_points)],
            'depth_config': self.current_depth_config,
            'timestamp': time.time()
        }
        
        return True
    
    def render_calibration_guide(self, frame):
        """
        Render calibration guide on the frame
        
        Args:
            frame: Frame to render on
            
        Returns:
            Rendered frame with calibration guide
        """
        if not self.is_calibrating:
            return frame
        
        h, w = frame.shape[:2]
        
        # Overlay semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        alpha = 0.5
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        
        # Draw calibration points
        for i, point in enumerate(self.reference_points):
            px = int(point[0] * w)
            py = int(point[1] * h)
            
            # Different colors for current, completed, and pending points
            if i == self.current_step % len(self.reference_points):
                color = (0, 255, 0)  # Green for current
                size = 20
            elif i < self.steps_completed % len(self.reference_points):
                color = (0, 255, 255)  # Yellow for completed
                size = 15
            else:
                color = (100, 100, 100)  # Gray for pending
                size = 15
                
            cv2.circle(frame, (px, py), size, color, -1)
            cv2.circle(frame, (px, py), size+2, (255, 255, 255), 2)
            
            # Add label
            labels = ["Top Left", "Top Right", "Bottom Right", "Bottom Left"]
            cv2.putText(frame, labels[i], (px - 40, py - 25), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Add progress bar
        progress = (self.steps_completed + 
                   (time.time() - self.step_start_time) / self.point_collect_time) / self.total_steps
        progress = max(0, min(1, progress))
        
        bar_width = int(w * 0.6)
        bar_height = 20
        bar_x = (w - bar_width) // 2
        bar_y = h - 100
        
        # Draw background
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                     (100, 100, 100), -1)
        
        # Draw progress
        progress_width = int(bar_width * progress)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + progress_width, bar_y + bar_height), 
                     (0, 255, 0), -1)
        
        # Draw border
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                     (255, 255, 255), 2)
        
        # Add instructions
        cv2.putText(frame, "Calibration in Progress", (w//2 - 150, 50), 
                   cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)
        
        instruction = "Place your index finger at the green point"
        cv2.putText(frame, instruction, (w//2 - 200, h - 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        return frame
    
    def apply_calibration_to_keyboard(self, keyboard):
        """
        Apply calibration data to keyboard
        
        Args:
            keyboard: KeyboardPlane object
            
        Returns:
            True if calibration applied, False otherwise
        """
        if not self.calibration_data or not keyboard:
            return False
        
        # Extract calibration points
        points = self.calibration_data['points']
        if len(points) < 3:  # Need at least 3 points to define a plane
            return False
        
        try:
            # Calculate plane using first three points
            v1 = np.array(points[1]) - np.array(points[0])
            v2 = np.array(points[2]) - np.array(points[0])
            
            # Calculate normal vector using cross product
            normal = np.cross(v1, v2)
            normal = normal / np.linalg.norm(normal)
            
            # Set plane parameters
            keyboard.plane_normal = normal
            
            # Calculate center of the plane
            center = np.mean(points, axis=0)
            keyboard.plane_center = center
            keyboard.plane_depth = center[2]
            
            # Update keyboard corners
            keyboard.corners_3d = keyboard._calculate_corners()
            
            return True
        except Exception as e:
            print(f"Error applying calibration: {e}")
            return False


class CalibrationUI:
    """User interface for calibration"""
    
    def __init__(self, calibration_manager):
        """
        Initialize calibration UI
        
        Args:
            calibration_manager: Instance of CalibrationManager
        """
        self.calibration_manager = calibration_manager
        
        # UI state
        self.show_ui = False
        self.selected_option = 'flat'
        self.button_regions = {}
        
    def toggle_ui(self):
        """Toggle calibration UI visibility"""
        self.show_ui = not self.show_ui
        
    def render(self, frame):
        """
        Render calibration UI on frame
        
        Args:
            frame: Frame to render on
            
        Returns:
            Rendered frame
        """
        if not self.show_ui:
            return frame
        
        if self.calibration_manager.is_calibrating:
            return self.calibration_manager.render_calibration_guide(frame)
        
        h, w = frame.shape[:2]
        
        # Semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        alpha = 0.4
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        
        # Title
        cv2.putText(frame, "Keyboard Calibration", (w//2 - 150, 50), 
                   cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)
        
        # Instructions
        instructions = [
            "Select a calibration profile:",
            "- Flat: All corners at same depth",
            "- Angled Front: Front edge closer to camera",
            "- Angled Side: Left edge closer to camera",
            "",
            "Click Start when ready."
        ]
        
        for i, text in enumerate(instructions):
            y = 100 + i * 30
            cv2.putText(frame, text, (w//2 - 200, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        
        # Option buttons
        button_height = 40
        button_width = 150
        button_margin = 20
        button_y = h // 2
        
        options = [
            ('flat', "Flat"),
            ('angled_front', "Angled Front"),
            ('angled_side', "Angled Side")
        ]
        
        self.button_regions = {}
        
        for i, (option_id, option_name) in enumerate(options):
            button_x = w//2 - button_width - button_margin + i * (button_width + button_margin)
            button_rect = (button_x, button_y, button_width, button_height)
            self.button_regions[option_id] = button_rect
            
            # Draw button
            color = (0, 200, 0) if option_id == self.selected_option else (100, 100, 100)
            cv2.rectangle(frame, (button_x, button_y), 
                         (button_x + button_width, button_y + button_height), color, -1)
            cv2.rectangle(frame, (button_x, button_y), 
                         (button_x + button_width, button_y + button_height), (255, 255, 255), 2)
            
            # Button text
            text_size = cv2.getTextSize(option_name, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            text_x = button_x + (button_width - text_size[0]) // 2
            text_y = button_y + (button_height + text_size[1]) // 2
            cv2.putText(frame, option_name, (text_x, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Start button
        start_button_width = 200
        start_button_x = (w - start_button_width) // 2
        start_button_y = button_y + button_height + 40
        start_button_height = 50
        
        start_button = (start_button_x, start_button_y, start_button_width, start_button_height)
        self.button_regions['start'] = start_button
        
        cv2.rectangle(frame, (start_button_x, start_button_y), 
                     (start_button_x + start_button_width, start_button_y + start_button_height), 
                     (0, 120, 255), -1)
        cv2.rectangle(frame, (start_button_x, start_button_y), 
                     (start_button_x + start_button_width, start_button_y + start_button_height), 
                     (255, 255, 255), 2)
        
        # Start button text
        text_size = cv2.getTextSize("Start Calibration", cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        text_x = start_button_x + (start_button_width - text_size[0]) // 2
        text_y = start_button_y + (start_button_height + text_size[1]) // 2
        cv2.putText(frame, "Start Calibration", (text_x, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Close button
        close_button_size = 30
        close_button_x = w - 50
        close_button_y = 50
        
        self.button_regions['close'] = (close_button_x - close_button_size//2, 
                                      close_button_y - close_button_size//2,
                                      close_button_size, close_button_size)
        
        cv2.circle(frame, (close_button_x, close_button_y), close_button_size//2, (0, 0, 200), -1)
        cv2.circle(frame, (close_button_x, close_button_y), close_button_size//2, (255, 255, 255), 2)
        cv2.line(frame, (close_button_x - 10, close_button_y - 10), 
                (close_button_x + 10, close_button_y + 10), (255, 255, 255), 2)
        cv2.line(frame, (close_button_x + 10, close_button_y - 10), 
                (close_button_x - 10, close_button_y + 10), (255, 255, 255), 2)
        
        return frame
    
    def handle_click(self, x, y):
        """
        Handle click events for the UI
        
        Args:
            x: X-coordinate of click
            y: Y-coordinate of click
            
        Returns:
            Dict with action result
        """
        if not self.show_ui:
            return {'action': None}
        
        # Check button clicks
        for button_id, (bx, by, bw, bh) in self.button_regions.items():
            if bx <= x <= bx + bw and by <= y <= by + bh:
                if button_id == 'close':
                    self.show_ui = False
                    return {'action': 'close_ui'}
                
                elif button_id == 'start':
                    self.calibration_manager.start_calibration(self.selected_option)
                    return {'action': 'start_calibration', 'option': self.selected_option}
                
                elif button_id in ['flat', 'angled_front', 'angled_side']:
                    self.selected_option = button_id
                    return {'action': 'select_option', 'option': button_id}
        
        return {'action': None}


def save_calibration(calibration_data, filename="calibration.npz"):
    """
    Save calibration data to file
    
    Args:
        calibration_data: Calibration data dictionary
        filename: Output filename
        
    Returns:
        True if saved successfully, False otherwise
    """
    try:
        if not calibration_data:
            return False
        
        np.savez(filename, 
                points=np.array(calibration_data['points']),
                reference_points=np.array(calibration_data['reference_points']),
                depth_config=calibration_data['depth_config'],
                timestamp=calibration_data['timestamp'])
        
        return True
    except Exception as e:
        print(f"Error saving calibration: {e}")
        return False


def load_calibration(filename="calibration.npz"):
    """
    Load calibration data from file
    
    Args:
        filename: Input filename
        
    Returns:
        Calibration data dictionary or None if failed
    """
    try:
        data = np.load(filename, allow_pickle=True)
        
        calibration_data = {
            'points': data['points'].tolist(),
            'reference_points': data['reference_points'].tolist(),
            'depth_config': str(data['depth_config']),
            'timestamp': float(data['timestamp'])
        }
        
        return calibration_data
    except Exception as e:
        print(f"Error loading calibration: {e}")
        return None


def check_calibration_quality(calibration_data):
    """
    Check quality of calibration data
    
    Args:
        calibration_data: Calibration data dictionary
        
    Returns:
        Dict with quality metrics and status
    """
    if not calibration_data or 'points' not in calibration_data:
        return {
            'status': 'invalid',
            'message': 'Invalid calibration data',
            'quality': 0.0
        }
    
    points = calibration_data['points']
    
    # Check number of points
    if len(points) < 3:
        return {
            'status': 'insufficient',
            'message': 'Not enough calibration points',
            'quality': 0.0
        }
    
    try:
        # Check if points form a reasonable plane
        points_array = np.array(points)
        
        # Calculate vectors between points
        v1 = points_array[1] - points_array[0]
        v2 = points_array[2] - points_array[0]
        
        # Calculate normal vector
        normal = np.cross(v1, v2)
        normal_length = np.linalg.norm(normal)
        
        if normal_length < 0.01:
            return {
                'status': 'colinear',
                'message': 'Calibration points are too close to a line',
                'quality': 0.3
            }
        
        # Check if remaining points are close to the plane
        if len(points) > 3:
            # Normalize normal vector
            normal = normal / normal_length
            
            # Calculate plane equation: ax + by + cz + d = 0
            a, b, c = normal
            d = -np.dot(normal, points_array[0])
            
            # Check distance of all points to plane
            max_distance = 0
            for point in points_array:
                distance = abs(np.dot(normal, point) + d)
                max_distance = max(max_distance, distance)
            
            if max_distance > 0.1:  # Threshold for "good" calibration
                return {
                    'status': 'inconsistent',
                    'message': 'Calibration points do not form a consistent plane',
                    'quality': 0.5
                }
            
            quality = max(0.0, min(1.0, 1.0 - max_distance * 5))
        else:
            quality = 0.7  # Default quality for minimal points
        
        # Calculate timestamp age
        age_hours = (time.time() - calibration_data['timestamp']) / 3600
        
        return {
            'status': 'valid',
            'message': 'Calibration valid',
            'quality': quality,
            'age_hours': age_hours,
            'normal_vector': normal.tolist()
        }
        
    except Exception as e:
        print(f"Error checking calibration quality: {e}")
        return {
            'status': 'error',
            'message': f'Error analyzing calibration: {str(e)}',
            'quality': 0.0
        }