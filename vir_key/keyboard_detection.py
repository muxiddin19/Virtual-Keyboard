"""
Keyboard Detection Module
------------------------
Defines and manages the virtual keyboard plane in 3D space.
"""

import cv2
import numpy as np

class KeyboardPlane:
    """Manages the virtual keyboard plane and key definitions"""
    
    # Standard keyboard layouts
    LAYOUTS = {
        'qwerty': [
            ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0'],
            ['Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P'],
            ['A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L'],
            ['Z', 'X', 'C', 'V', 'B', 'N', 'M', 'BACKSPACE'],
            ['SPACE']
        ],
        'compact': [
            ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0'],
            ['Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P'],
            ['A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L'],
            ['Z', 'X', 'C', 'V', 'B', 'N', 'M'],
            ['SPACE', 'BACKSPACE', 'ENTER']
        ]
    }
    
    def __init__(self, layout='qwerty', size=(0.6, 0.4)):
        """
        Initialize the virtual keyboard plane
        
        Args:
            layout: Keyboard layout name or a custom layout
            size: (width, height) of keyboard as proportion of screen
        """
        self.size = size
        self.layout = layout
        self.width, self.height = size
        self.position = position
        
        # Add the missing plane_depth attribute
        self.plane_depth = position[2]  # Z-coordinate from position
        
        # Initialize other attributes
        self.keys = {}
        self.hover_key = None
        self.pressed_key = None
        
        # Calculate key positions
        self._calculate_key_positions()
        # Set layout
        if isinstance(layout, str) and layout in self.LAYOUTS:
            self.layout = self.LAYOUTS[layout]
        elif isinstance(layout, list):
            self.layout = layout
        else:
            self.layout = self.LAYOUTS['qwerty']
        
        # Calculate key dimensions and positions
        self.keys = []
        self._calculate_key_positions()
        
        # Keyboard plane parameters in 3D space
        self.plane_depth = 0.5  # Normalized depth (z-coordinate)
        self.plane_normal = np.array([0, 0, 1])  # Points towards the camera
        self.plane_center = np.array([0.5, 0.5, self.plane_depth])
        
        # Corners of the keyboard in 3D space (for visualization)
        self.corners_3d = self._calculate_corners()
    
    def _calculate_key_positions(self):
        """Calculate positions and dimensions for each key"""
        # Clear existing keys
        self.keys = []
        
        # Get maximum keys per row
        max_keys = max(len(row) for row in self.layout)
        
        # Calculate key dimensions
        key_width = self.size[0] / max_keys
        key_height = self.size[1] / len(self.layout)
        
        # Keyboard top-left corner position
        kb_x = (1.0 - self.size[0]) / 2
        kb_y = (1.0 - self.size[1]) / 2
        
        # Calculate each key's position
        for row_idx, row in enumerate(self.layout):
            row_width = len(row) * key_width
            row_start_x = kb_x + (self.size[0] - row_width) / 2
            
            for key_idx, key_label in enumerate(row):
                # Special key sizes
                is_special = key_label in ['SPACE', 'BACKSPACE', 'ENTER']
                key_w = key_width * (3 if key_label == 'SPACE' else 
                                    1.5 if is_special else 1)
                
                # Calculate key rectangle
                x1 = row_start_x + key_idx * key_width
                y1 = kb_y + row_idx * key_height
                x2 = x1 + key_w
                y2 = y1 + key_height
                
                # Create key object
                key = {
                    'label': key_label,
                    'rect': [x1, y1, x2, y2],  # [x1, y1, x2, y2]
                    'center': [(x1 + x2) / 2, (y1 + y2) / 2, self.plane_depth],
                    'is_special': is_special
                }
                
                self.keys.append(key)
                
                # Adjust for special key widths
                if is_special:
                    row_start_x += key_w - key_width
    
    def _calculate_corners(self):
        """Calculate the corners of the keyboard in 3D space"""
        kb_x = (1.0 - self.size[0]) / 2
        kb_y = (1.0 - self.size[1]) / 2
        
        corners = [
            [kb_x, kb_y, self.plane_depth],  # Top-left
            [kb_x + self.size[0], kb_y, self.plane_depth],  # Top-right
            [kb_x + self.size[0], kb_y + self.size[1], self.plane_depth],  # Bottom-right
            [kb_x, kb_y + self.size[1], self.plane_depth]  # Bottom-left
        ]
        
        return np.array(corners)
    
    def set_depth(self, depth):
        """Set the depth of the keyboard plane"""
        self.plane_depth = depth
        self.plane_center[2] = depth
        
        # Update the corners with the new depth
        self.corners_3d = self._calculate_corners()
        
        # Update the depth for all key centers
        for key in self.keys:
            key['center'][2] = depth

    
    def get_key_at_position(self, position):
        """
        Find the key at the given normalized position (x, y)
        
        Args:
            position: Tuple (x, y) in normalized coordinates
            
        Returns:
            Key dict if position is over a key, None otherwise
        """
        for key in self.keys:
            x1, y1, x2, y2 = key['rect']
            if (x1 <= position[0] <= x2) and (y1 <= position[1] <= y2):
                return key
        return None
    
    def get_key_at_intersection(self, ray_origin, ray_direction):
        """
        Find the key at the intersection of a ray with the keyboard plane
        
        Args:
            ray_origin: 3D point where ray originates (finger tip)
            ray_direction: Direction vector of the ray
            
        Returns:
            Key dict if ray intersects a key, None otherwise
        """
        # Calculate ray-plane intersection
        intersection = self._ray_plane_intersection(ray_origin, ray_direction)
        
        # If no intersection with keyboard plane, return None
        if intersection is None:
            return None
        
        # Check if intersection point is within any key
        x, y, _ = intersection
        
        # Convert 3D coordinates to normalized 2D coordinates
        position = (x, y)
        
        return self.get_key_at_position(position)
    
    def _ray_plane_intersection(self, ray_origin, ray_direction):
        """
        Calculate the intersection of a ray with the keyboard plane
        
        Args:
            ray_origin: 3D point where ray originates
            ray_direction: Direction vector of the ray
            
        Returns:
            3D intersection point or None if no intersection
        """
        # Normalize the ray direction
        ray_direction = ray_direction / np.linalg.norm(ray_direction)
        
        # Calculate the denominator (dot product of ray and plane normal)
        denom = np.dot(ray_direction, self.plane_normal)
        
        # If denominator is zero, ray is parallel to plane
        if abs(denom) < 1e-6:
            return None
        
        # Calculate distance along ray to intersection
        t = np.dot(self.plane_center - ray_origin, self.plane_normal) / denom
        
        # If t is negative, intersection is behind ray origin
        if t < 0:
            return None
        
        # Calculate the intersection point
        intersection = ray_origin + t * ray_direction
        
        # Check if intersection is within keyboard bounds
        kb_x = (1.0 - self.size[0]) / 2
        kb_y = (1.0 - self.size[1]) / 2
        
        if not (kb_x <= intersection[0] <= kb_x + self.size[0] and
                kb_y <= intersection[1] <= kb_y + self.size[1]):
            return None
        
        return intersection
    
    def adjust_position(self, new_position):
        """
        Adjust the position of the keyboard plane
        
        Args:
            new_position: New center position (x, y, z)
        """
        self.plane_center = np.array(new_position)
        self.corners_3d = self._calculate_corners()
        
        # Update key positions
        for key in self.keys:
            # Keep x,y relative positions but update z
            key['center'][2] = self.plane_depth
    
    def render(self, frame, projection_matrix=None, camera_matrix=None):
        """
        Render the keyboard on the given frame
        
        Args:
            frame: Image to render on
            projection_matrix: Optional 3D-to-2D projection matrix
            camera_matrix: Optional camera intrinsic matrix
            
        Returns:
            Frame with rendered keyboard
        """
        height, width = frame.shape[:2]
        
        # Default simple projection if matrices not provided
        if projection_matrix is None or camera_matrix is None:
            # Simple rendering based on normalized coordinates
            for key in self.keys:
                x1, y1, x2, y2 = key['rect']
                
                # Convert normalized coordinates to pixel coordinates
                pt1 = (int(x1 * width), int(y1 * height))
                pt2 = (int(x2 * width), int(y2 * height))
                
                # Draw key rectangle
                color = (100, 100, 255) if key['is_special'] else (200, 200, 200)
                cv2.rectangle(frame, pt1, pt2, color, 1)
                
                # Draw key label
                label_pt = (int((x1 + x2) * width / 2 - 10), int((y1 + y2) * height / 2 + 5))
                font_scale = 0.4 if key['is_special'] else 0.5
                cv2.putText(frame, key['label'], label_pt, 
                          cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA)
        else:
            # Use projection and camera matrices for accurate 3D rendering
            for key in self.keys:
                x1, y1, x2, y2 = key['rect']
                
                # Create 3D points for key corners
                corners_3d = np.array([
                    [x1, y1, self.plane_depth],
                    [x2, y1, self.plane_depth],
                    [x2, y2, self.plane_depth],
                    [x1, y2, self.plane_depth]
                ])
                
                # Project 3D points to 2D image plane
                corners_2d, _ = cv2.projectPoints(
                    corners_3d, projection_matrix[:3, :3], projection_matrix[:3, 3],
                    camera_matrix, None
                )
                corners_2d = corners_2d.reshape(-1, 2).astype(int)
                
                # Draw key polygon
                color = (100, 100, 255) if key['is_special'] else (200, 200, 200)
                cv2.polylines(frame, [corners_2d], True, color, 1)
                
                # Calculate center for label
                center_2d = np.mean(corners_2d, axis=0).astype(int)
                
                # Draw key label
                font_scale = 0.4 if key['is_special'] else 0.5
                cv2.putText(frame, key['label'], tuple(center_2d), 
                          cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA)
                          
        return frame


class KeyboardDetector:
    """Detects and manages the virtual keyboard in the scene"""
    
    def __init__(self, config=None):
        """
        Initialize the keyboard detector
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        
        # Create default keyboard
        self.keyboard = KeyboardPlane(
            layout=self.config.get('layout', 'qwerty'),
            size=self.config.get('size', (0.6, 0.4))
        )
        
        # Calibration state
        self.is_calibrated = False
        self.calibration_points = []
        
    def detect_keyboard_plane(self, frame, hand_landmarks=None):
        """
        Detect or update the keyboard plane position based on the frame
        and optional hand landmarks
        
        Args:
            frame: Current camera frame
            hand_landmarks: Optional hand landmarks from hand tracking
            
        Returns:
            Updated keyboard plane
        """
        # If not calibrated and hand landmarks provided, use them
        # for automatic keyboard placement
        if not self.is_calibrated and hand_landmarks is not None:
            self._auto_place_keyboard(frame, hand_landmarks)
            
        return self.keyboard
    
    def _auto_place_keyboard(self, frame, hand_landmarks):
        """
        Automatically place the keyboard relative to hand position
        
        Args:
            frame: Current camera frame
            hand_landmarks: Hand landmarks from hand tracking
        """
        # Get wrist position (assuming hand_landmarks has a wrist point)
        if 'wrist' in hand_landmarks:
            wrist = hand_landmarks['wrist']
            
            # Calculate a position for keyboard below the hand
            # This is a simple heuristic - adjust as needed
            height, width = frame.shape[:2]
            
            # Normalize coordinates
            wrist_x = wrist[0] / width
            wrist_y = wrist[1] / height
            
            # Place keyboard below hand
            kb_center_x = 0.5  # Center horizontally
            kb_center_y = wrist_y + 0.2  # Below the wrist
            kb_depth = self.keyboard.plane_depth
            
            # Update keyboard position
            self.keyboard.adjust_position([kb_center_x, kb_center_y, kb_depth])
            
    def start_calibration(self):
        """Start the keyboard calibration process"""
        self.is_calibrated = False
        self.calibration_points = []
        
    def add_calibration_point(self, point_3d):
        """
        Add a calibration point to define the keyboard plane
        
        Args:
            point_3d: 3D point to add
        """
        self.calibration_points.append(point_3d)
        
        # If we have enough points, calibrate the plane
        if len(self.calibration_points) >= 3:
            self._calibrate_from_points()
            
    def _calibrate_from_points(self):
        """Calibrate keyboard plane using collected points"""
        if len(self.calibration_points) < 3:
            return False
            
        # Convert points to numpy array
        points = np.array(self.calibration_points)
        
        # Calculate plane using first three points
        v1 = points[1] - points[0]
        v2 = points[2] - points[0]
        
        # Calculate normal vector using cross product
        normal = np.cross(v1, v2)
        normal = normal / np.linalg.norm(normal)
        
        # Set plane parameters
        self.keyboard.plane_normal = normal
        
        # Calculate center of the plane
        center = np.mean(points, axis=0)
        self.keyboard.plane_center = center
        self.keyboard.plane_depth = center[2]
        
        # Update keyboard corners
        self.keyboard.corners_3d = self.keyboard._calculate_corners()
        
        self.is_calibrated = True
        return True
        
    def render_debug(self, frame):
        """
        Render debug visualizations
        
        Args:
            frame: Image to render on
            
        Returns:
            Frame with debug visualizations
        """
        # Render keyboard
        frame = self.keyboard.render(frame)
        
        # Render calibration points if available
        height, width = frame.shape[:2]
        for i, point in enumerate(self.calibration_points):
            # Project 3D point to 2D (simple projection)
            pt_x = int(point[0] * width)
            pt_y = int(point[1] * height)
            
            # Draw point
            cv2.circle(frame, (pt_x, pt_y), 5, (0, 255, 0), -1)
            cv2.putText(frame, f"P{i}", (pt_x + 10, pt_y), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
        return frame
'''
This continuation adds several important features to your `keyboard_detection.py` module:

1. Methods to detect key intersections with finger rays for detecting keypresses
2. Rendering functionality to visualize the keyboard
3. A `KeyboardDetector` class that manages keyboard plane detection and calibration
4. Support for both simple 2D and full 3D projection rendering
5. Automatic keyboard placement based on hand position
6. A calibration system allowing users to define the keyboard plane

The code maintains the architecture and style of your existing implementation while extending functionality in a modular way. It handles the core responsibility of this module: defining and managing the virtual keyboard plane in 3D space.
'''