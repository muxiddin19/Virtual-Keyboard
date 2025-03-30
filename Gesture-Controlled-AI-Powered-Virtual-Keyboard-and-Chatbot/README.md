# Gesture-Controlled AI-Powered Virtual Keyboard and Chatbot  
*(English & Arabic)*

## Objective  
This project implements a real-time virtual keyboard controlled by hand gestures, supporting both English and Arabic typing, with AI-driven text suggestions and an integrated chatbot.

## Technologies  
- **Languages**: Python  
- **Framework**: Flask  
- **Libraries**:  
  - OpenCV (for hand gesture detection)  
  - MediaPipe (for hand-tracking)  
  - Google Generative AI (for text suggestions and chatbot integration)  
  - Flask-CORS (for handling cross-origin requests)  
  - Arabic Reshaper (for proper Arabic text reshaping)  
  - Bidi Algorithm (for handling right-to-left Arabic text)  
  - NumPy and PIL (for image processing and rendering)

## Key Features  
- **Real-time Hand Gesture Detection**: Utilizes OpenCV and MediaPipe for accurate hand-tracking and gesture recognition.  
- **Dual-Language Virtual Keyboard**: Supports seamless switching between English and Arabic layouts.  
- **AI-Powered Text Suggestions**: Integrated Google Generative AI provides real-time text suggestions and autocompletion in both languages.  
- **Arabic Text Reshaping**: Implements Arabic Reshaper and Bidi Algorithm to ensure proper alignment and display of Arabic text.  
- **Integrated Chatbot**: Provides an AI chatbot using Google Generative AI for user interaction.  
- **Web-based Interface**: Real-time video streaming of hand gestures, with immediate feedback on typed text and chatbot responses.

