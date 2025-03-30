# Virtual-Keyboard
Accurately hand fingers detection, tracking &amp; Virtual Keyboard

## MediaPipe Hand Tracking with OpenCV
Welcome to the MediaPipe Hand Tracking project! This repository contains a Python-based application that leverages MediaPipe and OpenCV to detect and visualize hand landmarks in real-time using a webcam. The program tracks hand movements and draws annotations on the video feed, making it a great starting point for projects in computer vision, gesture recognition, or human-computer interaction (HCI).
### Features
- Real-Time Hand Tracking: Detects and tracks hand landmarks with high accuracy using MediaPipe.
- Landmark Visualization: Draws hand joints and connections on the video feed.
- Lightweight and Simple: Minimal code for easy understanding and modification.
- Webcam Integration: Works with any standard webcam via OpenCV.

### Demo
Demo GIF
(Note: demo GIF will be added soon.)
### Prerequisites
Before running the project, ensure you have the following:
- Python 3.7 or higher
- A webcam (integrated or external)
- Required Python libraries (see Dependencies (#dependencies) below)

### Installation
1. Clone the Repository:
bash
```
git clone https://github.com/[Your-Username]/MediaPipe-Hand-Tracking.git
cd MediaPipe-Hand-Tracking
```
2. Install Dependencies:
Install the required Python packages using pip:
bash
```
pip install opencv-python mediapipe
```
3. Verify Webcam:
Ensure your webcam is connected and functional. The code defaults to device 0 (change cv2.VideoCapture(0) if using a different device).

### Usage
1. Run the Script:
bash
```
python hand_tracking.py
```
(Note: Rename your script to hand_tracking.py or adjust the command to match your file name.)

Int2. eract with the Program:
- A window titled "Hand Tracking" will open, showing the live webcam feed.
- Place your hand in front of the camera to see real-time hand landmark detection.
- The program draws dots on hand joints and lines connecting them.

3. Exit the Program:
- Press q while the window is active to close the application.

### Dependencies
- OpenCV (opencv-python): For video capture and image processing.
- MediaPipe (mediapipe): For hand detection and landmark tracking.

Install all dependencies with:
bash
```
pip install opencv-python mediapipe
```
### How It Works
1. Video Capture:
- OpenCV’s VideoCapture reads frames from the webcam in BGR format.
2. Hand Detection:
- Frames are converted to RGB and processed by MediaPipe’s Hands solution.
- MediaPipe detects up to two hands by default and returns landmark coordinates.

3. Visualization:
- The drawing_utils module from MediaPipe annotates the frame with hand landmarks (21 per hand) and connections.
- Annotated frames are displayed in real-time.

### Customization
- Camera Source: Change cv2.VideoCapture(0) to another index (e.g., 1) for a different camera.
- Detection Settings: Modify Hands() parameters, e.g., max_num_hands=1 to track only one hand, or adjust min_detection_confidence (default 0.5) for sensitivity.
- Visualization Style: Customize colors or styles in draw_landmarks by passing a DrawingSpec (see MediaPipe docs).

Example tweak for single-hand tracking:
python
```
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
```
###Troubleshooting
- Webcam Not Working: Ensure the correct camera index is set in cv2.VideoCapture(0).
- No Landmarks Detected: Check lighting conditions, hand positioning, or lower min_detection_confidence.
- Window Not Opening: Verify OpenCV is installed correctly (pip show opencv-python).

### Contributing
-  Contributions are welcome! To contribute:
- Fork the repository.
- Create a new branch (git checkout -b feature-name).
- Make your changes and commit (git commit -m "Add feature").
- Push to your branch (git push origin feature-name).
- Open a pull request.

Please include comments in your code and adhere to Python style guidelines (PEP 8).
### License
This project is licensed under the MIT License. See the LICENSE file for details.
### Acknowledgments
- Powered by Google’s MediaPipe for hand tracking.
- Built with OpenCV for video processing.

### Contact
For questions or suggestions, reach out via GitHub Issues or contact the maintainer at muhiddin1979@kaist.ac.kr.

