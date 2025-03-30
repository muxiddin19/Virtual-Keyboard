# Virtual Keyboard Project

A computer vision-based virtual keyboard system that allows users to type in the air using hand gestures. The system uses a standard webcam to track hand movements, estimate depth, and detect virtual key presses.

## Features

- Hand tracking using MediaPipe
- Single-camera depth estimation
- Virtual keyboard plane detection and interaction
- Key press detection with visual and audio feedback
- Customizable keyboard layout and settings
- Calibration system for different environments
- Real-time visualization with debugging options

## Requirements

- Python 3.8+
- OpenCV
- NumPy
- MediaPipe
- PyAutoGUI (for system keystroke output)
- SciPy (optional, for advanced depth estimation)

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/virtual_keyboard.git
   cd virtual_keyboard
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Run the main application:
   ```
   python main.py
   ```

2. Calibrate the system by following the on-screen instructions (place your index finger at each calibration point).

3. Start typing by moving your hand in the air and pressing virtual keys with your index finger.

### Command Line Arguments

- `--config`: Path to custom configuration file
- `--camera`: Camera device ID (default: 0)
- `--resolution`: Camera resolution as WIDTHxHEIGHT (default: 1280x720)
- `--layout`: Keyboard layout (qwerty, azerty, numeric)
- `--calibrate`: Start with calibration (default: auto-detect)
- `--debug`: Enable debug visualization
- `--fullscreen`: Enable fullscreen mode

Example:
```
python main.py --camera 1 --resolution 1920x1080 --layout qwerty --calibrate
```

## Project Structure

```
virtual_keyboard/
├── main.py                # Entry point
├── hand_tracking.py       # Hand tracking module
├── depth_estimation.py    # Depth estimation algorithms
├── keyboard_detection.py  # Virtual keyboard plane detection 
├── key_detection.py       # Key press detection logic
├── visualization.py       # Visualization utilities
├── calibration.py         # Calibration procedures
├── config.py              # Configuration settings
├── utils.py               # Utility functions
└── README.md              # Project documentation
```

## Module Responsibilities

- **main.py**: Orchestrates the entire system, minimal logic
- **hand_tracking.py**: Interfaces with MediaPipe for hand detection
- **depth_estimation.py**: Algorithms for single-camera depth estimation
- **keyboard_detection.py**: Defines and manages the virtual keyboard plane
- **key_detection.py**: Logic for determining key presses
- **visualization.py**: Display functions for debugging and user interface
- **calibration.py**: User calibration procedures
- **config.py**: Configuration constants and settings
- **utils.py**: Shared utility functions

## Calibration

The system requires calibration to work effectively in different environments. The calibration process:

1. Places the virtual keyboard in the correct position relative to the user
2. Adjusts for different lighting conditions and camera setups
3. Establishes depth references for accurate key press detection

Calibration options:
- **Flat**: Standard keyboard at uniform depth
- **Angled Front**: Keyboard with the front edge closer to the user
- **Angled Side**: Keyboard with one side closer to the user

## Configuration

Configuration settings are stored in a JSON file and can be modified either through the provided UI or by directly editing the file.

Key configuration sections:
- **Camera settings**: Device, resolution, FPS
- **Hand tracking settings**: Detection confidence, smoothing
- **Depth estimation settings**: Method, reference values
- **Keyboard settings**: Layout, dimensions, thresholds
- **Calibration settings**: Steps, timing, auto-calibration
- **Visualization settings**: Debug options, display toggles
- **Application settings**: UI options, system integration

## Custom Keyboard Layouts

You can create custom keyboard layouts by modifying the `KEYBOARD_LAYOUTS` dictionary in `config.py`.

## Troubleshooting

If you encounter issues:

1. **Calibration problems**: Try adjusting lighting or camera position, and use the "Recalibrate" option.
2. **Hand detection issues**: Ensure good lighting and a clear background.
3. **Depth estimation problems**: Adjust the `depth.reference_hand_width` setting in the configuration.
4. **Key press sensitivity**: Modify the `keyboard.hover_threshold` and `keyboard.press_threshold` values.
5. **Performance issues**: Lower the camera resolution or disable debug visualizations.

## How It Works

1. **Hand tracking**: The system uses MediaPipe to detect and track hand landmarks.
2. **Depth estimation**: Estimates the 3D position of hand landmarks using the relative size of the hand.
3. **Keyboard plane detection**: Creates a virtual keyboard plane in 3D space.
4. **Key press detection**: Determines when a finger has "pressed" a virtual key by tracking when it crosses the keyboard plane threshold.
5. **Visualization**: Provides real-time feedback to the user about hand position and key presses.
6. **System integration**: Translates virtual key presses into actual system input.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [MediaPipe](https://mediapipe.dev/) for the hand tracking solution
- [OpenCV](https://opencv.org/) for computer vision capabilities
- All contributors who have helped shape this project

## Future Improvements

- Multi-hand support for faster typing
- Improved depth estimation for better accuracy
- Gesture shortcuts for common commands
- Support for more keyboard layouts and languages
- Mobile device compatibility
- VR/AR integration possibilities

## Contact

If you have any questions or suggestions, please open an issue or contact the project maintainers.