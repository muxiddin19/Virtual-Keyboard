#!/usr/bin/env python3
"""
Virtual Keyboard System - Main Entry Point
-----------------------------------------
A webcam-based virtual keyboard system without specialized hardware.
"""

import cv2
import time
import numpy as np
import argparse

# Import project modules
from hand_tracking import HandTracker
from depth_estimation import DepthEstimator
from keyboard_detection import KeyboardPlane
from key_detection import KeyDetector
from visualization import Visualizer
from calibration import Calibrator
from config import Config

class VirtualKeyboardSystem:
    """Main system class that orchestrates all components"""
 # In main.py, update the __init__ method
    def __init__(self, config_file=None):
        # Load configuration
        self.config = Config()
        if config_file:
            self.config.load_from_file(config_file)
        
        # Set default values if attributes are missing
        if not hasattr(self.config, 'depth_method'):
            self.config.depth_method = "relative_size"  # default value
        
        # Initialize depth estimation
        self.depth_estimator = DepthEstimator(
            method=getattr(self.config, 'depth_method', "relative_size")
        )   
    
        #self.config = Config(config_file)
        
        # Initialize components
        self.hand_tracker = HandTracker(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        # self.depth_estimator = DepthEstimator(
        #     method=self.config.depth_method
        # )
        keyboard_position = config.get('keyboard_position', (0, 0, 0.3))
    
        self.keyboard = KeyboardPlane(
        layout=config.get('keyboard_layout', 'qwerty'),
        size=config.get('keyboard_size', (0.3, 0.2)),
        position_param=keyboard_position  # Make sure this matches the parameter name in KeyboardPlane
    ) 
        # self.keyboard = KeyboardPlane(
        #     layout=self.config.keyboard_layout,
        #     size=self.config.keyboard_size
        # )
        
        self.key_detector = KeyDetector(
            activation_threshold=self.config.activation_threshold,
            dwell_time=self.config.dwell_time
        )
        
        self.visualizer = Visualizer(
            show_hands=self.config.show_hands,
            show_keyboard=self.config.show_keyboard,
            show_depth=self.config.show_depth
        )
        
        self.calibrator = Calibrator(
            num_points=4,
            timeout=30
        )
        
        # State variables
        self.is_running = False
        self.is_calibrated = False
        self.text_buffer = ""
        self.fps = 0
        self.frame_count = 0
        self.start_time = 0
        
    def start(self):
        """Main loop to start the system"""
        print("Starting Virtual Keyboard System...")
        
        # Open webcam
        self.cap = cv2.VideoCapture(self.config.camera_id)
        if not self.cap.isOpened():
            raise IOError("Cannot open webcam")
            
        self.is_running = True
        self.start_time = time.time()
        
        # Check if calibration is needed
        if not self.is_calibrated and self.config.require_calibration:
            self.run_calibration()
        
        print("System initialized. Press 'q' to quit.")
        
        # Main loop
        while self.is_running:
            # Read frame
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to grab frame")
                break
                
            # Flip horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Track hands
            hand_landmarks = self.hand_tracker.process_frame(frame)
            
            if hand_landmarks:
                # Estimate depths
                depths = self.depth_estimator.estimate(hand_landmarks, frame)
                
                # Detect key presses
                pressed_key = self.key_detector.detect(
                    hand_landmarks, 
                    depths,
                    self.keyboard
                )
                
                # Handle key press
                if pressed_key:
                    self.handle_key_press(pressed_key)
            
            # Update FPS calculation
            self.frame_count += 1
            elapsed_time = time.time() - self.start_time
            if elapsed_time > 1:
                self.fps = self.frame_count / elapsed_time
                self.frame_count = 0
                self.start_time = time.time()
            
            # Visualize
            display_frame = self.visualizer.draw(
                frame, 
                hand_landmarks, 
                depths if hand_landmarks else None,
                self.keyboard,
                self.text_buffer,
                self.fps
            )
            
            # Show frame
            cv2.imshow('Virtual Keyboard', display_frame)
            
            # Check for quit
            key = cv2.waitKey(1)
            if key == ord('q'):
                self.is_running = False
            elif key == ord('c'):
                self.run_calibration()
        
        # Clean up
        self.cap.release()
        cv2.destroyAllWindows()
        print("System stopped.")
    
    def run_calibration(self):
        """Run the calibration procedure"""
        print("Starting calibration...")
        self.is_calibrated = self.calibrator.calibrate(
            self.cap, 
            self.hand_tracker,
            self.keyboard
        )
        if self.is_calibrated:
            print("Calibration successful!")
        else:
            print("Calibration failed or was cancelled.")
    
    def handle_key_press(self, key):
        """Handle a key press event"""
        if key == "BACKSPACE":
            if self.text_buffer:
                self.text_buffer = self.text_buffer[:-1]
        elif key == "SPACE":
            self.text_buffer += " "
        elif key == "ENTER":
            self.text_buffer += "\n"
        elif key == "CLEAR":
            self.text_buffer = ""
        else:
            self.text_buffer += key

# def parse_args():
#     """Parse command line arguments"""
#     parser = argparse.ArgumentParser(description='Virtual Keyboard System')
#     parser.add_argument('--config', type=str, help='Path to config file')
#     parser.add_argument('--camera', type=int, default=0, help='Camera ID')
#     return parser.parse_args()



def parse_arguments():
    parser = argparse.ArgumentParser(description='Virtual Keyboard using computer vision')
    parser.add_argument('--config', type=str, help='Path to custom configuration file')
    parser.add_argument('--camera', type=int, default=0, help='Camera device ID (default: 0)')
    parser.add_argument('--resolution', type=str, default='1280x720', 
                        help='Camera resolution as WIDTHxHEIGHT (default: 1280x720)')
    parser.add_argument('--layout', type=str, default='qwerty', 
                        choices=['qwerty', 'azerty', 'numeric'], 
                        help='Keyboard layout (qwerty, azerty, numeric)')
    parser.add_argument('--calibrate', action='store_true', 
                        help='Start with calibration (default: auto-detect)')
    parser.add_argument('--debug', action='store_true', help='Enable debug visualization')
    parser.add_argument('--fullscreen', action='store_true', help='Enable fullscreen mode')
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    
    # Override config with command line args
    config_file = args.config if args.config else None
    
    # Create and start the system
    system = VirtualKeyboardSystem(config_file)
    system.start()



# def main():
#     args = parse_arguments()
    
#     # Parse resolution string into width and height
#     if args.resolution:
#         try:
#             width, height = map(int, args.resolution.split('x'))
#             # Use width and height here for camera setup
#         except ValueError:
#             print(f"Invalid resolution format: {args.resolution}. Using default.")
#             width, height = 1280, 720
    
#     # Rest of your code...
    
# if __name__ == "__main__":
#     main()