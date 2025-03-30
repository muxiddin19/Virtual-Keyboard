"""
Calibration Module
----------------
Implements calibration procedures for the virtual keyboard.
"""

import cv2
import numpy as np
import time

class CalibrationManager:
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
            print(f"Error applying calibration