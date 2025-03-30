# AI Virtual Keyboard using Hand Tracking

An AI-powered virtual keyboard using computer vision and hand gesture recognition. This project uses OpenCV, cvzone, and MediaPipe for hand detection, and simulates keyboard inputs using pynput. It also includes a backspace and space functionality for realistic typing.

---

## ✨ **Features**
- Real-time hand tracking using a webcam
- Virtual keyboard display with dynamic key highlighting
- Supports all standard alphabet keys, space, and backspace
- Accurate finger distance detection for click simulation
- Simple and user-friendly interface

---

## 🛠 **Tech Stack**
- **Python**
- **OpenCV** - Image processing and video capture
- **MediaPipe** - Hand tracking model
- **cvzone** - Hand detection utilities
- **Numpy** - Efficient array operations
- **pynput** - Keyboard input simulation

---

## 💡 **Prerequisites**
Make sure you have Python installed. Then install the required packages using the following commands:

```bash
pip install opencv-python
pip install cvzone
pip install numpy
pip install pynput
```

---

## 📝 **How to Run**
1. Clone the repository:
    ```bash
    git clone https://github.com/YourUsername/AI-Virtual-Keyboard.git
    cd AI-Virtual-Keyboard
    ```

2. Run the Python script:
    ```bash
    python VirtualMouseAi.py
    ```

3. Ensure your webcam is connected.
4. Use your index and middle fingers to simulate key presses.
5. Press **Q** to quit.

---

## 🖼 **Demo**
Here are some screenshots of the project in action:
![Screenshot (82)](https://github.com/user-attachments/assets/8555fc33-c073-48ef-90f6-3a84e1a37ca4)


![Screenshot (83)](https://github.com/user-attachments/assets/4276715e-a940-48a6-9698-191d83a04f44)


---

## 🔎 **How It Works**
- **Hand Detection**: Using cvzone's HandDetector to track hands in real-time.
- **Finger Detection**: Identifies the position of fingers using landmarks.
- **Distance Calculation**: Measures distance between index and middle fingers to detect clicks.
- **Keyboard Simulation**: pynput is used to simulate actual keyboard inputs.
- **Backspace & Space**: Additional keys implemented for better typing control.

---

## 🛠 **Project Structure**
```bash
AI-Virtual-Keyboard/
├── VirtualMouseAi.py         # Main Python script
├── HandTrackingModule.py     # Hand detection logic using MediaPipe
├── Model/                    # (Optional) Model for gesture recognition
├── images/                   # Demo images
├── README.md                 # Project Documentation
```

---

## ✅ **Future Enhancements**
- Implement gesture recognition for additional commands
- Add voice input support
- Improve accuracy for faster typing

---

## 🧑‍💻 **Contributing**
Contributions are welcome! Feel free to submit issues and pull requests.

---

## 🛡 **License**
This project is licensed under the MIT License. See `LICENSE` for more details.

---

## 📢 **Acknowledgments**
- [OpenCV](https://opencv.org/)
- [MediaPipe](https://developers.google.com/mediapipe/)
- [cvzone](https://github.com/cvzone/cvzone)
- [pynput](https://pypi.org/project/pynput/)

---

## 👤 **Contact**
For any inquiries, feel free to reach out at [harshalsharma288@gmail.com](mailto:harshalsharma288@gmail.com) or create an issue on GitHub.

