"""
Utilities Module
--------------
Utility functions for the virtual keyboard system.
"""

import os
import time
import math
import numpy as np
import cv2


def timestamp():
    """
    Get current timestamp
    
    Returns:
        Current timestamp as string
    """
    return time.strftime("%Y-%m-%d %H:%M:%S")


def format_time(seconds):
    """
    Format time in seconds to human-readable format
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds // 60
        secs = seconds % 60
        return f"{int(minutes)}m {int(secs)}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        return f"{int(hours)}h {int(minutes)}m"


def calculate_fps(start_time, frame_count, smoothing=0.9):
    """
    Calculate frames per second
    
    Args:
        start_time: Start time in seconds
        frame_count: Frame count
        smoothing: Smoothing factor (0-1)
        
    Returns:
        FPS value
    """
    current_time = time.time()
    elapsed = current_time - start_time
    
    if elapsed == 0:
        return 0
    
    return frame_count / elapsed


class FPSCounter:
    """FPS counter for tracking performance"""
    
    def __init__(self, smoothing=0.9, window_size=30):
        """
        Initialize FPS counter
        
        Args:
            smoothing: Smoothing factor (0-1)
            window_size: Window size for rolling average
        """
        self.smoothing = smoothing
        self.window_size = window_size
        self.prev_time = time.time()
        self.fps = 0
        self.fps_history = []
    
    def update(self):
        """
        Update FPS counter
        
        Returns:
            Current FPS
        """
        current_time = time.time()
        elapsed = current_time - self.prev_time
        
        if elapsed == 0:
            return self.fps
        
        current_fps = 1.0 / elapsed
        
        # Update rolling history
        self.fps_history.append(current_fps)
        if len(self.fps_history) > self.window_size:
            self.fps_history.pop(0)
        
        # Calculate smoothed FPS
        if not self.fps:
            self.fps = current_fps
        else:
            # Use either smoothing or rolling average
            if self.smoothing > 0:
                self.fps = self.fps * self.smoothing + current_fps * (1 - self.smoothing)
            else:
                self.fps = sum(self.fps_history) / len(self.fps_history)
        
        self.prev_time = current_time
        return self.fps
    
    def get(self):
        """
        Get current FPS
        
        Returns:
            Current FPS value
        """
        return self.fps
    
    def draw(self, frame, position=(30, 30), color=(0, 255, 0), size=0.6, thickness=2):
        """
        Draw FPS counter on frame
        
        Args:
            frame: Frame to draw on
            position: Position (x, y)
            color: Text color
            size: Text size
            thickness: Text thickness
            
        Returns:
            Frame with FPS counter
        """
        fps_text = f"FPS: {self.fps:.1f}"
        cv2.putText(frame, fps_text, position, cv2.FONT_HERSHEY_SIMPLEX, 
                   size, color, thickness)
        return frame


def smooth_value(new_value, old_value, smoothing=0.8):
    """
    Apply smoothing to a numeric value
    
    Args:
        new_value: New value
        old_value: Previous value
        smoothing: Smoothing factor (0-1)
        
    Returns:
        Smoothed value
    """
    if old_value is None:
        return new_value
    
    return old_value * smoothing + new_value * (1 - smoothing)


def smooth_point(new_point, old_point, smoothing=0.8):
    """
    Apply smoothing to a point (x, y) or (x, y, z)
    
    Args:
        new_point: New point coordinates
        old_point: Previous point coordinates
        smoothing: Smoothing factor (0-1)
        
    Returns:
        Smoothed point
    """
    if old_point is None:
        return new_point
    
    result = []
    for i in range(min(len(new_point), len(old_point))):
        result.append(old_point[i] * smoothing + new_point[i] * (1 - smoothing))
    
    return result


def euclidean_distance(point1, point2):
    """
    Calculate Euclidean distance between two points
    
    Args:
        point1: First point (x, y) or (x, y, z)
        point2: Second point (x, y) or (x, y, z)
        
    Returns:
        Distance between points
    """
    sum_sq = 0
    for i in range(min(len(point1), len(point2))):
        sum_sq += (point1[i] - point2[i]) ** 2
    
    return math.sqrt(sum_sq)


def distance_to_plane(point, plane_normal, plane_point):
    """
    Calculate distance from point to plane
    
    Args:
        point: Point coordinates (x, y, z)
        plane_normal: Plane normal vector
        plane_point: Point on the plane
        
    Returns:
        Signed distance to plane
    """
    point = np.array(point)
    plane_normal = np.array(plane_normal)
    plane_point = np.array(plane_point)
    
    # Normalize plane normal
    normal_length = np.linalg.norm(plane_normal)
    if normal_length < 1e-6:
        return 0
    
    plane_normal = plane_normal / normal_length
    
    # Calculate signed distance
    v = point - plane_point
    distance = np.dot(v, plane_normal)
    
    return distance


def project_point_to_plane(point, plane_normal, plane_point):
    """
    Project a point onto a plane
    
    Args:
        point: Point coordinates (x, y, z)
        plane_normal: Plane normal vector
        plane_point: Point on the plane
        
    Returns:
        Projected point on the plane
    """
    point = np.array(point)
    plane_normal = np.array(plane_normal)
    plane_point = np.array(plane_point)
    
    # Normalize plane normal
    normal_length = np.linalg.norm(plane_normal)
    if normal_length < 1e-6:
        return point
    
    plane_normal = plane_normal / normal_length
    
    # Calculate signed distance
    v = point - plane_point
    distance = np.dot(v, plane_normal)
    
    # Project point onto plane
    projected_point = point - distance * plane_normal
    
    return projected_point


def point_in_rect(point, rect):
    """
    Check if point is inside rectangle
    
    Args:
        point: Point coordinates (x, y)
        rect: Rectangle (x, y, width, height)
        
    Returns:
        True if point is inside rectangle, False otherwise
    """
    x, y = point
    rx, ry, rw, rh = rect
    
    return rx <= x <= rx + rw and ry <= y <= ry + rh


def normalize_coordinates(point, frame_shape):
    """
    Normalize coordinates to range [0, 1]
    
    Args:
        point: Point coordinates (x, y)
        frame_shape: Frame shape (height, width)
        
    Returns:
        Normalized coordinates
    """
    h, w = frame_shape[:2]
    x, y = point
    
    return (x / w, y / h)


def denormalize_coordinates(point, frame_shape):
    """
    Denormalize coordinates from range [0, 1]
    
    Args:
        point: Normalized coordinates (x, y)
        frame_shape: Frame shape (height, width)
        
    Returns:
        Denormalized coordinates
    """
    h, w = frame_shape[:2]
    x, y = point
    
    return (int(x * w), int(y * h))


def resize_frame(frame, target_width=None, target_height=None):
    """
    Resize frame to target width or height
    
    Args:
        frame: Frame to resize
        target_width: Target width (None to calculate from height)
        target_height: Target height (None to calculate from width)
        
    Returns:
        Resized frame
    """
    h, w = frame.shape[:2]
    
    if target_width is None and target_height is None:
        return frame
    
    if target_width is None:
        aspect_ratio = w / h
        target_width = int(target_height * aspect_ratio)
    elif target_height is None:
        aspect_ratio = h / w
        target_height = int(target_width * aspect_ratio)
    
    return cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)


def draw_text_with_background(frame, text, position, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.6, 
                             color=(255, 255, 255), thickness=1, bg_color=(0, 0, 0), 
                             bg_opacity=0.7, padding=5):
    """
    Draw text with semi-transparent background
    
    Args:
        frame: Frame to draw on
        text: Text to draw
        position: Position (x, y)
        font: Font type
        font_scale: Font scale
        color: Text color
        thickness: Text thickness
        bg_color: Background color
        bg_opacity: Background opacity (0-1)
        padding: Background padding
        
    Returns:
        Frame with text
    """
    x, y = position
    
    # Get text size
    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
    text_w, text_h = text_size
    
    # Create background rectangle
    rect_x = x - padding
    rect_y = y - text_h - padding
    rect_w = text_w + padding * 2
    rect_h = text_h + padding * 2
    
    # Draw semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (rect_x, rect_y), (rect_x + rect_w, rect_y + rect_h), bg_color, -1)
    cv2.addWeighted(overlay, bg_opacity, frame, 1 - bg_opacity, 0, frame)
    
    # Draw text
    cv2.putText(frame, text, (x, y), font, font_scale, color, thickness)
    
    return frame


def draw_progress_bar(frame, progress, position, width=100, height=20, 
                     fg_color=(0, 255, 0), bg_color=(100, 100, 100), 
                     border_color=(255, 255, 255), border_width=2):
    """
    Draw progress bar on frame
    
    Args:
        frame: Frame to draw on
        progress: Progress value (0-1)
        position: Position (x, y)
        width: Bar width
        height: Bar height
        fg_color: Foreground color
        bg_color: Background color
        border_color: Border color
        border_width: Border width
        
    Returns:
        Frame with progress bar
    """
    x, y = position
    progress = max(0, min(1, progress))  # Clamp to 0-1
    
    # Draw background
    cv2.rectangle(frame, (x, y), (x + width, y + height), bg_color, -1)
    
    # Draw progress
    progress_width = int(width * progress)
    cv2.rectangle(frame, (x, y), (x + progress_width, y + height), fg_color, -1)
    
    # Draw border
    if border_width > 0:
        cv2.rectangle(frame, (x, y), (x + width, y + height), border_color, border_width)
    
    return frame


def create_directory(directory):
    """
    Create directory if it doesn't exist
    
    Args:
        directory: Directory path
        
    Returns:
        True if directory exists or was created, False otherwise
    """
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
        return True
    except Exception as e:
        print(f"Error creating directory {directory}: {e}")
        return False


def save_frame(frame, output_dir="screenshots", prefix="frame", file_format="jpg"):
    """
    Save frame to file
    
    Args:
        frame: Frame to save
        output_dir: Output directory
        prefix: Filename prefix
        file_format: File format
        
    Returns:
        Path to saved file or None if failed
    """
    try:
        # Create directory if needed
        if not create_directory(output_dir):
            return None
        
        # Generate filename
        timestamp_str = time.strftime("%Y%m%d_%H%M%S")
        ms = int(time.time() * 1000) % 1000
        filename = f"{prefix}_{timestamp_str}_{ms}.{file_format}"
        filepath = os.path.join(output_dir, filename)
        
        # Save frame
        cv2.imwrite(filepath, frame)
        
        return filepath
    except Exception as e:
        print(f"Error saving frame: {e}")
        return None


class MovingAverage:
    """Moving average calculator for smoothing values"""
    
    def __init__(self, window_size=10):
        """
        Initialize moving average
        
        Args:
            window_size: Window size
        """
        self.window_size = window_size
        self.values = []
        self.sum = 0
    
    def update(self, value):
        """
        Update moving average with new value
        
        Args:
            value: New value
            
        Returns:
            Current moving average
        """
        self.values.append(value)
        self.sum += value
        
        if len(self.values) > self.window_size:
            self.sum -= self.values.pop(0)
        
        return self.get()
    
    def get(self):
        """
        Get current moving average
        
        Returns:
            Current moving average or None if no values
        """
        if not self.values:
            return None
        
        return self.sum / len(self.values)
    
    def reset(self):
        """Reset moving average"""
        self.values = []
        self.sum = 0


class KeyPressTracker:
    """Tracks key press events with timing"""
    
    def __init__(self, timeout=0.5):
        """
        Initialize key press tracker
        
        Args:
            timeout: Key press timeout in seconds
        """
        self.pressed_keys = {}
        self.last_press = None
        self.timeout = timeout
    
    def press(self, key):
        """
        Register key press
        
        Args:
            key: Key identifier
            
        Returns:
            True if this is a new press, False if key is already pressed
        """
        current_time = time.time()
        
        # Check if key is already pressed
        if key in self.pressed_keys:
            # Check if press has timed out
            if current_time - self.pressed_keys[key] > self.timeout:
                self.pressed_keys[key] = current_time
                self.last_press = key
                return True
            return False
        
        # New key press
        self.pressed_keys[key] = current_time
        self.last_press = key
        return True
    
    def release(self, key):
        """
        Register key release
        
        Args:
            key: Key identifier
            
        Returns:
            True if key was pressed, False otherwise
        """
        if key in self.pressed_keys:
            del self.pressed_keys[key]
            return True
        
        return False
    
    def is_pressed(self, key):
        """
        Check if key is pressed
        
        Args:
            key: Key identifier
            
        Returns:
            True if key is pressed, False otherwise
        """
        if key not in self.pressed_keys:
            return False
        
        # Check for timeout
        current_time = time.time()
        if current_time - self.pressed_keys[key] > self.timeout:
            del self.pressed_keys[key]
            return False
        
        return True
    
    def get_pressed_keys(self):
        """
        Get list of currently pressed keys
        
        Returns:
            List of pressed keys
        """
        # Check for timeouts
        current_time = time.time()
        expired_keys = []
        
        for key, press_time in self.pressed_keys.items():
            if current_time - press_time > self.timeout:
                expired_keys.append(key)
        
        # Remove expired keys
        for key in expired_keys:
            del self.pressed_keys[key]
        
        return list(self.pressed_keys.keys())
    
    def reset(self):
        """Reset all pressed keys"""
        self.pressed_keys = {}
        self.last_press = None