"""
Configuration Module
------------------
Configuration settings for the virtual keyboard system.
"""

import os
import json
from pathlib import Path

# Default configuration
DEFAULT_CONFIG = {
    # Camera settings
    "camera": {
        "device_id": 0,                # Camera device ID
        "width": 1280,                 # Camera frame width
        "height": 720,                 # Camera frame height
        "fps": 30,                     # Camera frame rate
        "vertical_flip": False,        # Flip camera vertically
        "horizontal_flip": False,      # Flip camera horizontally
    },
    
    # Hand tracking settings
    "hand_tracking": {
        "max_num_hands": 1,            # Maximum number of hands to track
        "min_detection_confidence": 0.7,  # Minimum confidence for hand detection
        "min_tracking_confidence": 0.5,   # Minimum confidence for hand tracking
        "use_static_image_mode": False,   # Static image mode (slower but more accurate)
        "landmark_smoothing": 0.7,        # Smoothing factor for landmarks (0-1)
        "feature_smoothing": 0.5,         # Smoothing factor for features (0-1)
    },
    
    # Depth estimation settings
    "depth": {
        "method": "hand_size",         # Depth estimation method (hand_size, stereovision)
        "min_depth": 0.2,              # Minimum depth in meters
        "max_depth": 1.0,              # Maximum depth in meters
        "reference_hand_width": 0.08,  # Reference hand width in meters
        "depth_smoothing": 0.8,        # Smoothing factor for depth values
    },
    
    # Virtual keyboard settings
    "keyboard": {
        "layout": "qwerty",            # Keyboard layout
        "width": 0.30,                 # Keyboard width in meters
        "height": 0.15,                # Keyboard height in meters
        "key_spacing": 0.005,          # Space between keys in meters
        "default_depth": 0.5,          # Default depth for keyboard plane
        "hover_threshold": 0.02,       # Distance threshold for hover in meters
        "press_threshold": 0.01,       # Distance threshold for press in meters
        "press_duration": 0.2,         # Duration threshold for key press in seconds
        "feedback_duration": 0.3,      # Duration for visual feedback in seconds
    },
    
    # Calibration settings
    "calibration": {
        "calibration_steps": 4,        # Number of calibration steps
        "point_collect_time": 2.0,     # Time to collect points at each step
        "calibration_timeout": 30.0,   # Calibration timeout in seconds
        "auto_calibrate": True,        # Auto-calibrate on startup
        "save_calibration": True,      # Save calibration to file
        "calibration_file": "calibration.npz",  # Calibration file path
    },
    
    # Visualization settings
    "visualization": {
        "show_fps": True,              # Show FPS counter
        "show_hand_landmarks": True,   # Show hand landmarks
        "show_keyboard": True,         # Show virtual keyboard
        "show_depth": True,            # Show depth information
        "show_press_point": True,      # Show press point
        "keyboard_opacity": 0.7,       # Keyboard opacity (0-1)
        "debug_view": False,           # Enable debug visualization
    },
    
    # Application settings
    "app": {
        "window_name": "Virtual Keyboard",  # Window name
        "fullscreen": False,           # Fullscreen mode
        "output_to_system": True,      # Output keystrokes to system
        "text_preview": True,          # Show text preview
        "max_preview_chars": 50,       # Maximum preview characters
        "auto_save_log": True,         # Auto-save typing log
        "log_file": "typing_log.txt",  # Log file path
    }
}

# Keyboard layouts
KEYBOARD_LAYOUTS = {
    "qwerty": [
        ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0", "-", "=", "⌫"],
        ["q", "w", "e", "r", "t", "y", "u", "i", "o", "p", "[", "]", "\\"],
        ["a", "s", "d", "f", "g", "h", "j", "k", "l", ";", "'", "⏎"],
        ["⇧", "z", "x", "c", "v", "b", "n", "m", ",", ".", "/", "⇧"],
        ["ctrl", "alt", " ", "alt", "ctrl"]
    ],
    "azerty": [
        ["&", "é", "\"", "'", "(", "-", "è", "_", "ç", "à", ")", "=", "⌫"],
        ["a", "z", "e", "r", "t", "y", "u", "i", "o", "p", "^", "$", "*"],
        ["q", "s", "d", "f", "g", "h", "j", "k", "l", "m", "ù", "⏎"],
        ["⇧", "w", "x", "c", "v", "b", "n", ",", ";", ":", "!", "⇧"],
        ["ctrl", "alt", " ", "alt", "ctrl"]
    ],
    "numeric": [
        ["1", "2", "3"],
        ["4", "5", "6"],
        ["7", "8", "9"],
        ["0", ".", "⌫"]
    ]
}

# Special key mappings
SPECIAL_KEYS = {
    "⌫": "backspace",
    "⏎": "return",
    "⇧": "shift",
    " ": "space",
    "ctrl": "control",
    "alt": "alt"
}

class Config:
    """Configuration manager for the virtual keyboard system"""
    
    def __init__(self, config_file=None):
        """
        Initialize configuration
        
        Args:
            config_file: Path to configuration file
        """
        self.config = DEFAULT_CONFIG.copy()
        self.config_file = config_file or "keyboard_config.json"
        self.depth_method = "relative_size"  # or another default value
        
        # You might want to add other related settings too
        self.depth_reference_hand_width = 0.08  # in meters
        self.depth_min_threshold = 0.1
        self.depth_max_threshold = 0.5
        self.keyboard_layout = "qwerty"  # Default value
        
        # You might need additional keyboard settings
        self.keyboard_hover_threshold = 0.02  # in meters
        self.keyboard_press_threshold = 0.01  # in meters
        self.keyboard_width = 0.3  # in meters
        self.keyboard_height = 0.2  # in meters
        self.camera_id = 0
        self.camera_width = 1280
        self.camera_height = 720
        self.camera_fps = 30
        
        # Hand tracking settings
        self.hand_detection_confidence = 0.7
        self.hand_tracking_confidence = 0.5
        self.hand_max_num_hands = 1
        self.hand_smoothing = True
        
        # Depth estimation settings
        self.depth_method = "relative_size"
        self.depth_reference_hand_width = 0.08  # in meters
        self.depth_min_threshold = 0.1
        self.depth_max_threshold = 0.5
        
        # Keyboard settings
        self.keyboard_layout = "qwerty"
        self.keyboard_size = (0.3, 0.2)  # width and height in meters
        self.keyboard_hover_threshold = 0.02  # in meters
        self.keyboard_press_threshold = 0.01  # in meters
        
        # Calibration settings
        self.start_with_calibration = False
        self.calibration_points = 5
        self.calibration_time = 3  # seconds per point
        
        # Visualization settings
        self.debug_mode = False
        self.fullscreen = False
        self.show_hand_landmarks = True
        self.show_depth_values = True
        
        # Application settings
        self.audio_feedback = True
        # Load configuration if exists
        if os.path.exists(self.config_file):
            self.load()
    

    def get(self, section, key=None, default=None):
        """
        Get configuration value
        
        Args:
            section: Configuration section
            key: Configuration key (None for entire section)
            default: Default value if not found
            
        Returns:
            Configuration value or default
        """
        if key is None:
            return self.config.get(section, default)
        
        return self.config.get(section, {}).get(key, default)
    
    def set(self, section, key, value):
        """
        Set configuration value
        
        Args:
            section: Configuration section
            key: Configuration key
            value: Configuration value
        """
        if section not in self.config:
            self.config[section] = {}
        
        self.config[section][key] = value
    
    def load(self):
        """Load configuration from file"""
        try:
            with open(self.config_file, 'r') as f:
                loaded_config = json.load(f)
                
                # Update config with loaded values
                for section, values in loaded_config.items():
                    if section in self.config:
                        if isinstance(values, dict):
                            self.config[section].update(values)
                        else:
                            self.config[section] = values
                    else:
                        self.config[section] = values
                        
            return True
        except Exception as e:
            print(f"Error loading configuration: {e}")
            return False
    
    def save(self):
        """Save configuration to file"""
        try:
            # Create directory if needed
            config_dir = os.path.dirname(self.config_file)
            if config_dir and not os.path.exists(config_dir):
                os.makedirs(config_dir)
                
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=4)
                
            return True
        except Exception as e:
            print(f"Error saving configuration: {e}")
            return False
    
    def get_keyboard_layout(self):
        """
        Get keyboard layout
        
        Returns:
            List of keyboard rows and keys
        """
        layout_name = self.get("keyboard", "layout", "qwerty")
        return KEYBOARD_LAYOUTS.get(layout_name, KEYBOARD_LAYOUTS["qwerty"])
    
    def get_special_key_mapping(self, key):
        """
        Get special key mapping
        
        Args:
            key: Special key symbol
            
        Returns:
            Special key name or original key
        """
        return SPECIAL_KEYS.get(key, key)


# Singleton config instance
config = Config()