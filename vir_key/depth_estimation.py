"""
Depth Estimation Module
----------------------
Estimates the depth (z-coordinate) of hand landmarks using a single camera.
"""

import cv2
import numpy as np
import math

class DepthEstimator:
    """Estimates depth using various single-camera methods"""
    
    def __init__(self, method='heuristic', model_path=None):
        """
        Initialize the depth estimator
        
        Args:
            method: Depth estimation method to use. Options:
                   'heuristic' - Use hand size heuristics
                   'ml' - Use machine learning model
                   'geometric' - Use geometric constraints
            model_path: Path to ML model if using 'ml' method
        """
        self.method = method
        self.model = None
        self.history = []  # For temporal smoothing
        self.history_size = 5
        
        # Base sizes for heuristic method (calibration values)
        self.base_hand_size = None
        self.base_depth = 0.5  # Normalized reference depth
        
        # Load ML model if specified
        if method == 'ml' and model_path:
            self._load_model(model_path)
    
    def estimate(self, hand_data, frame):
        """
        Estimate depth for hand landmarks
        
        Args:
            hand_data: Dictionary with hand landmarks from HandTracker
            frame: Original image frame for reference
            
        Returns:
            Dictionary with depth information for each hand
        """
        if not hand_data or 'landmarks' not in hand_data:
            return None
            
        depths = []
        
        for i, landmarks in enumerate(hand_data['landmarks']):
            if self.method == 'heuristic':
                depth = self._estimate_by_heuristic(landmarks, frame)
            elif self.method == 'ml':
                depth = self._estimate_by_ml(landmarks, frame)
            elif self.method == 'geometric':
                depth = self._estimate_by_geometric(landmarks, frame)
            else:
                # Default fallback
                depth = self._estimate_by_heuristic(landmarks, frame)
                
            # Add temporal smoothing
            depth = self._apply_temporal_smoothing(depth, i)
            depths.append(depth)
            
        return depths
    
    def _estimate_by_heuristic(self, landmarks, frame):
        """
        Estimate depth using hand size heuristic
        
        The idea: Hand appears smaller as it moves away from camera.
        We can use the hand width or distance between landmarks
        to estimate relative depth.
        """
        # Get dimensions
        frame_height, frame_width = frame.shape[:2]
        
        # Calculate hand size by measuring distance between thumb and pinky
        thumb_tip = landmarks[4][:2]  # x,y of thumb tip
        pinky_tip = landmarks[20][:2]  # x,y of pinky tip
        
        # Convert to pixel coordinates
        thumb_px = (thumb_tip[0] * frame_width, thumb_tip[1] * frame_height)
        pinky_px = (pinky_tip[0] * frame_width, pinky_tip[1] * frame_height)
        
        # Calculate Euclidean distance
        hand_size = np.sqrt((thumb_px[0] - pinky_px[0])**2 + 
                           (thumb_px[1] - pinky_px[1])**2)
        
        # If this is the first measurement, use it as baseline
        if self.base_hand_size is None:
            self.base_hand_size = hand_size
            return self.base_depth
        
        # Calculate relative depth
        # As hand_size decreases (moves away), depth increases
        if hand_size > 0:
            relative_depth = self.base_hand_size / hand_size
            
            # Normalize to reasonable range (0.1 to 1.0)
            normalized_depth = np.clip(relative_depth * self.base_depth, 0.1, 1.0)
            return normalized_depth
        
        return self.base_depth
    
    def _estimate_by_ml(self, landmarks, frame):
        """
        Estimate depth using a machine learning model
        
        This would use a pre-trained model to predict depth from 2D landmarks.
        For actual implementation, you'd need a trained model.
        """
        if self.model is None:
            # Fallback to heuristic if no model
            return self._estimate_by_heuristic(landmarks, frame)
        
        # Prepare input for model (would depend on your model architecture)
        model_input = self._prepare_landmarks_for_model(landmarks)
        
        # Get prediction
        # In real implementation, this would call your model's predict method
        depth = 0.5  # Placeholder for model prediction
        
        return depth
    
    def _estimate_by_geometric(self, landmarks, frame):
        """
        Estimate depth using geometric constraints of the hand
        
        Uses the known proportions of a human hand to estimate depth.
        """
        # Get key points
        wrist = landmarks[0]
        index_mcp = landmarks[5]  # Knuckle of index finger
        index_tip = landmarks[8]  # Tip of index finger
        
        # Known proportions (these are approximations)
        # Average adult index finger length is about 7.5-8cm
        # Average hand width at knuckles is about 8.5-9cm
        REAL_FINGER_LENGTH = 8.0  # cm
        
        # Calculate observed finger length in pixels
        finger_px_length = np.sqrt(
            (index_tip[3] - index_mcp[3])**2 + 
            (index_tip[4] - index_mcp[4])**2
        )
        
        # Estimate focal length of the camera (simplified)
        # In a real implementation, you would calibrate this
        focal_length = frame.shape[1]  # Approximate focal length
        
        # Estimate depth using similar triangles
        if finger_px_length > 0:
            depth_cm = REAL_FINGER_LENGTH * focal_length / finger_px_length
            
            # Normalize to 0-1 range (assuming max depth of 100cm)
            normalized_depth = np.clip(depth_cm / 100.0, 0.1, 1.0)
            return normalized_depth
        
        return 0.5  # Default depth
    
    def _apply_temporal_smoothing(self, depth, hand_idx):
        """Apply temporal smoothing to depth values to reduce jitter"""
        # Ensure we have a history list for this hand
        while len(self.history) <= hand_idx:
            self.history.append([])
            
        # Add current depth to history
        self.history[hand_idx].append(depth)
        
        # Keep history at fixed size
        if len(self.history[hand_idx]) > self.history_size:
            self.history[hand_idx].pop(0)
            
        # Calculate smoothed depth
        smoothed_depth = np.mean(self.history[hand_idx])
        
        return smoothed_depth
    
    def _load_model(self, model_path):
        """Load the ML model for depth estimation"""
        try:
            # This is a placeholder - you would use the appropriate
            # loading method for your model type (TensorFlow, PyTorch, etc.)
            print(f"Loading depth estimation model from {model_path}")
            # self.model = YourModelLibrary.load(model_path)
        except Exception as e:
            print(f"Error loading depth model: {e}")
    
    def _prepare_landmarks_for_model(self, landmarks):
        """Prepare landmarks data for input to ML model"""
        # This would depend on what your model expects
        # Typically you'd flatten the landmarks into a 1D array
        flattened = []
        for lm in landmarks:
            # Use only x, y coordinates for the model
            flattened.extend([lm[0], lm[1]])
        
        return np.array(flattened).reshape(1, -1)
    
    def calibrate(self, frame, hand_landmarks):
        """
        Calibrate the depth estimator with a reference hand position
        
        This should be called when the hand is at a known distance
        """
        if hand_landmarks:
            if self.method == 'heuristic':
                # Reset base values
                landmarks = hand_landmarks['landmarks'][0]
                self.base_hand_size = None
                depth = self._estimate_by_heuristic(landmarks, frame)
                print(f"Depth estimator calibrated. Base depth: {depth}")
                return True
        
        return False

# Simple test code
if __name__ == "__main__":
    from hand_tracking import HandTracker
    
    # Test with webcam
    cap = cv2.VideoCapture(0)
    tracker = HandTracker()
    depth_estimator = DepthEstimator()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Flip frame horizontally
        frame = cv2.flip(frame, 1)
        
        # Track hands
        hand_data = tracker.process_frame(frame)
        
        # Estimate depth
        depths = None
        if hand_data:
            depths = depth_estimator.estimate(hand_data, frame)
            
        # Draw landmarks
        frame = tracker.draw_landmarks(frame)
        
        # Show depth information
        if depths:
            for i, depth in enumerate(depths):
                color = (0, int(255 * (1 - depth)), int(255 * depth))
                cv2.putText(frame, f"Depth {i}: {depth:.2f}", 
                           (10, 70 + 40*i), cv2.FONT_HERSHEY_SIMPLEX, 
                           1, color, 2)
        
        # Show the frame
        cv2.imshow('Depth Estimation Test', frame)
        
        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()