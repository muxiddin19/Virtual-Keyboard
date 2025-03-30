Welcome to the AI Virtual On-Screen Keyboard project! This repository contains a Python-based application that uses computer vision and hand-tracking technology to create a virtual keyboard controllable via hand gestures. Built with OpenCV, cvzone, and Pynput, this tool allows users to type on-screen by hovering over and "clicking" virtual keys using their fingers, with real-time feedback and sound effects.

# Features
- Hand Gesture Typing: Detects hand movements to select and press keys on a virtual QWERTY keyboard.
- Visual Feedback: Highlights keys on hover and changes color on click.
- Audio Feedback: Plays a keypress sound (kc.wav) for each successful input.
- Real-Time Text Display: Shows typed text in a dedicated output box on the screen.
- Customizable Layout: Displays a standard QWERTY keyboard with adjustable button sizes and positions.

## Demo
Demo GIF
(Note: Will be replaced with an actual demo GIF.)

# Prerequisites
Before running the project, ensure you have the following installed:
Python 3.7 or higher
A webcam (integrated or external)
Required Python libraries (see Dependencies (#dependencies) below)

# Installation
1. Clone the Repository:
bash```
git clone https://github.com/Titan-Spy/AI-Virtual-On-Screen-Keyboard-Using-Computer-Vision.git
cd AI-Virtual-On-Screen-Keyboard-Using-Computer-Vision
```

2. Install Dependencies:
Install the required Python packages using pip:
bash```
pip install opencv-python cvzone pynput pygame numpy
```

3. Download Sound File:
- The project uses a kc.wav file for keypress sounds. Ensure this file is in the root directory of the project.  

- If you don’t have it, replace kc.wav in the code with your own sound file or remove the sound feature by commenting out the pygame.mixer.Sound.play(mc) line.

4.  Verify Webcam:
- Ensure your webcam is connected and functional. The code defaults to device 0 (change cv2.VideoCapture(0) if using a different device).

# Usage
1. Run the Script:
bash```
python main.py
```
(Note: Rename your script to main.py or adjust the command to match your file name.)

2. Interact with the Keyboard:
- A window will open displaying a virtual QWERTY keyboard.
- Position your hand in front of the webcam.
- Hover your index finger over a key to highlight it.
- Bring your index and middle fingers close together (distance < 30 pixels) to "click" the key.
- Typed text appears in the purple box at the bottom of the screen.
3. Exit the Program:
- Press Ctrl+C in the terminal or close the OpenCV window.

