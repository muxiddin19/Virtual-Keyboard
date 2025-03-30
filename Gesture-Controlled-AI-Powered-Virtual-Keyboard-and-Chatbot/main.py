#
import cv2
from flask import Flask, render_template, Response, jsonify, request
from flask_cors import CORS
import mediapipe as mp
import time
import arabic_reshaper
from bidi.algorithm import get_display
import numpy as np
from PIL import ImageFont, ImageDraw, Image
import google.generativeai as genai
from dotenv import load_dotenv
import os
import google.generativeai as genai
from collections import defaultdict

app = Flask(__name__)
CORS(app)

# Variable to hold the typed text
typed_text = ""
selected_suggestion_index = 0

# Initialize the camera (webcam)
camera = cv2.VideoCapture(0)

#GEMINI_API_KEY="AIzaSyBT9CgTQlf68ZMwpWTRD1-doQLlK92JpYQ"
load_dotenv()
# Securely retrieve the API key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Configure the API with the retrieved key
genai.configure(api_key=GEMINI_API_KEY)
# Configure Google Generative AI SDK
#genai.configure(api_key=os.environ.get("GEMINI_API_KEY", "AIzaSyBT9CgTQlf68ZMwpWTRD1-doQLlK92JpYQ"))
#genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
#genai.configure(api_key=os.getenv("GEMINI_API_KEY"))


# Create the model
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="gemini-1.5-pro",
    generation_config=generation_config,
)

# Initialize hand detection model and other resources
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
fontpath = "arial.ttf"
font = ImageFont.truetype(fontpath, 32)

# Define English and Arabic keyboards
english_keys = [
    # Row 1
    {'char': 'Q', 'pos': ((50, 100), (90, 140)), 'pressed': False, 'start_time': 0},
    {'char': 'W', 'pos': ((120, 100), (160, 140)), 'pressed': False, 'start_time': 0},
    {'char': 'E', 'pos': ((190, 100), (230, 140)), 'pressed': False, 'start_time': 0},
    {'char': 'R', 'pos': ((260, 100), (300, 140)), 'pressed': False, 'start_time': 0},
    {'char': 'T', 'pos': ((330, 100), (370, 140)), 'pressed': False, 'start_time': 0},
    {'char': 'Y', 'pos': ((400, 100), (440, 140)), 'pressed': False, 'start_time': 0},
    {'char': 'U', 'pos': ((470, 100), (510, 140)), 'pressed': False, 'start_time': 0},
    {'char': 'I', 'pos': ((540, 100), (580, 140)), 'pressed': False, 'start_time': 0},
    {'char': 'O', 'pos': ((610, 100), (650, 140)), 'pressed': False, 'start_time': 0},
    {'char': 'P', 'pos': ((680, 100), (720, 140)), 'pressed': False, 'start_time': 0},

    # Row 2
    {'char': 'A', 'pos': ((70, 170), (110, 210)), 'pressed': False, 'start_time': 0},
    {'char': 'S', 'pos': ((140, 170), (180, 210)), 'pressed': False, 'start_time': 0},
    {'char': 'D', 'pos': ((210, 170), (250, 210)), 'pressed': False, 'start_time': 0},
    {'char': 'F', 'pos': ((280, 170), (320, 210)), 'pressed': False, 'start_time': 0},
    {'char': 'G', 'pos': ((350, 170), (390, 210)), 'pressed': False, 'start_time': 0},
    {'char': 'H', 'pos': ((420, 170), (460, 210)), 'pressed': False, 'start_time': 0},
    {'char': 'J', 'pos': ((490, 170), (530, 210)), 'pressed': False, 'start_time': 0},
    {'char': 'K', 'pos': ((560, 170), (600, 210)), 'pressed': False, 'start_time': 0},
    {'char': 'L', 'pos': ((630, 170), (670, 210)), 'pressed': False, 'start_time': 0},

    # Row 3
    {'char': 'Z', 'pos': ((100, 240), (140, 280)), 'pressed': False, 'start_time': 0},
    {'char': 'X', 'pos': ((170, 240), (210, 280)), 'pressed': False, 'start_time': 0},
    {'char': 'C', 'pos': ((240, 240), (280, 280)), 'pressed': False, 'start_time': 0},
    {'char': 'V', 'pos': ((310, 240), (350, 280)), 'pressed': False, 'start_time': 0},
    {'char': 'B', 'pos': ((380, 240), (420, 280)), 'pressed': False, 'start_time': 0},
    {'char': 'N', 'pos': ((450, 240), (490, 280)), 'pressed': False, 'start_time': 0},
    {'char': 'M', 'pos': ((520, 240), (560, 280)), 'pressed': False, 'start_time': 0},

    # Space button
    {'char': ' ', 'pos': ((250, 310), (370, 350)), 'label': 'SPACE', 'pressed': False, 'start_time': 0},

    # Backspace button
    {'char': 'Backspace', 'pos': ((385, 310), (590, 350)), 'label': 'BACKSPACE', 'pressed': False, 'start_time': 0},

    # Change Language button
    {'char': 'Language', 'pos': ((45, 310), (235, 350)), 'label': 'LANGUAGE', 'pressed': False, 'start_time': 0},

    # Next Suggestion button
    {'char': 'NextSuggestion', 'pos': ((750, 100), (845, 140)), 'label': 'NEXT', 'pressed': False, 'start_time': 0},

    # Select Suggestion button
    {'char': 'SelectSuggestion', 'pos': ((750, 150), (885, 190)), 'label': 'SELECT', 'pressed': False, 'start_time': 0}
]

arabic_keys = [
    # Row 1 (Right-to-left order for Arabic)
    {'char': 'ض', 'pos': ((40, 100), (80, 140)), 'pressed': False, 'start_time': 0},
    {'char': 'ص', 'pos': ((110, 100), (150, 140)), 'pressed': False, 'start_time': 0},
    {'char': 'ث', 'pos': ((180, 100), (220, 140)), 'pressed': False, 'start_time': 0},
    {'char': 'ق', 'pos': ((250, 100), (290, 140)), 'pressed': False, 'start_time': 0},
    {'char': 'ف', 'pos': ((320, 100), (360, 140)), 'pressed': False, 'start_time': 0},
    {'char': 'غ', 'pos': ((400, 100), (440, 140)), 'pressed': False, 'start_time': 0},
    {'char': 'ع', 'pos': ((470, 100), (510, 140)), 'pressed': False, 'start_time': 0},
    {'char': 'ه', 'pos': ((540, 100), (570, 140)), 'pressed': False, 'start_time': 0},
    {'char': 'خ', 'pos': ((600, 100), (640, 140)), 'pressed': False, 'start_time': 0},
    {'char': 'ح', 'pos': ((670, 100), (710, 140)), 'pressed': False, 'start_time': 0},
    {'char': 'ج', 'pos': ((750, 100), (790, 140)), 'pressed': False, 'start_time': 0},
    {'char': 'د', 'pos': ((820, 100), (860, 140)), 'pressed': False, 'start_time': 0},


    # Row 2
    {'char': 'ش', 'pos': ((70, 170), (110, 210)), 'pressed': False, 'start_time': 0},
    {'char': 'س', 'pos': ((140, 170), (180, 210)), 'pressed': False, 'start_time': 0},
    {'char': 'ي', 'pos': ((210, 170), (250, 210)), 'pressed': False, 'start_time': 0},
    {'char': 'ب', 'pos': ((280, 170), (320, 210)), 'pressed': False, 'start_time': 0},
    {'char': 'ل', 'pos': ((350, 170), (390, 210)), 'pressed': False, 'start_time': 0},
    {'char': 'ا', 'pos': ((420, 170), (460, 210)), 'pressed': False, 'start_time': 0},
    {'char': 'ت', 'pos': ((490, 170), (530, 210)), 'pressed': False, 'start_time': 0},
    {'char': 'ن', 'pos': ((560, 170), (600, 210)), 'pressed': False, 'start_time': 0},
    {'char': 'م', 'pos': ((630, 170), (670, 210)), 'pressed': False, 'start_time': 0},
    {'char': 'ك', 'pos': ((700, 170), (740, 210)), 'pressed': False, 'start_time': 0},
    {'char': 'ط', 'pos': ((770, 170), (810, 210)), 'pressed': False, 'start_time': 0},

    # Row 3
    {'char': 'ئ', 'pos': ((100, 240), (140, 280)), 'pressed': False, 'start_time': 0},
    {'char': 'ء', 'pos': ((170, 240), (210, 280)), 'pressed': False, 'start_time': 0},
    {'char': 'ؤ', 'pos': ((240, 240), (280, 280)), 'pressed': False, 'start_time': 0},
    {'char': 'ر', 'pos': ((310, 240), (350, 280)), 'pressed': False, 'start_time': 0},
    {'char': 'لا', 'pos': ((380, 240), (420, 280)), 'pressed': False, 'start_time': 0},
    {'char': 'ى', 'pos': ((450, 240), (490, 280)), 'pressed': False, 'start_time': 0},
    {'char': 'ة', 'pos': ((520, 240), (560, 280)), 'pressed': False, 'start_time': 0},
    {'char': 'و', 'pos': ((590, 240), (630, 280)), 'pressed': False, 'start_time': 0},
    {'char': 'ز', 'pos': ((660, 240), (700, 280)), 'pressed': False, 'start_time': 0},
    {'char': 'ظ', 'pos': ((730, 240), (770, 280)), 'pressed': False, 'start_time': 0},

    # Space button
    {'char': ' ', 'pos': ((250, 300), (350, 340)), 'label': 'مسافه', 'pressed': False, 'start_time': 0},

    {'char': 'Backspace', 'pos': ((370, 300), (530, 340)), 'label': 'رجوع للخلف', 'pressed': False, 'start_time': 0},

    {'char': 'Language', 'pos': ((150, 300), (220, 340)), 'label': 'اللغه', 'pressed': False, 'start_time': 0},

    # Next Suggestion button
    {'char': 'NextSuggestion', 'pos': ((860, 170), (920, 210)), 'label': 'التالي', 'pressed': False, 'start_time': 0},

    # Select Suggestion button
    {'char': 'SelectSuggestion', 'pos': ((860, 220), (920, 260)), 'label': 'اختر', 'pressed': False, 'start_time': 0}
]

# Current language and keyboard
current_language = 'English'
keys = english_keys if current_language == 'English' else arabic_keys

# Track the state of pressed keys
key_pressed = {key['char']: False for key in english_keys}
key_press_delay = 0.3  # 200 milliseconds delay

# Load English and Arabic word dictionaries
def load_word_dictionary(file_path="english.txt"):
    word_dict = defaultdict(list)
    with open(file_path, "r", encoding="utf-8") as file:
        for word in file:
            word = word.strip().lower()
            for i in range(1, len(word) + 1):
                word_dict[word[:i]].append(word)
    return word_dict

def load_arabic_word_dictionary(file_path="arabic.txt"):
    word_dict = defaultdict(list)
    with open(file_path, "r", encoding="utf-8") as file:
        for word in file:
            word = word.strip()
            reshaped_word = arabic_reshaper.reshape(word)  # Reshape Arabic words to handle correct display
            bidi_word = get_display(reshaped_word)
            for i in range(1, len(word) + 1):
                word_dict[word[:i]].append(bidi_word)
    return word_dict

english_word_dict = load_word_dictionary("english.txt")
arabic_word_dict = load_arabic_word_dictionary("arabic.txt")

# Function to get suggestions for typed text based on current language
def get_suggestions(typed_text, current_language, english_word_dict, arabic_word_dict, max_suggestions=5):
    # Split the text to get the last word being typed
    words = typed_text.split()
    if not words:
        return []
    
    # Get the last word typed
    last_word = words[-1]
    
    if current_language == 'English':
        return get_english_suggestions(last_word, english_word_dict, max_suggestions)
    elif current_language == 'Arabic':
        return get_arabic_suggestions(last_word, arabic_word_dict, max_suggestions)
    else:
        return []

# Function to get English suggestions
def get_english_suggestions(typed_text, word_dict, max_suggestions=5):
    typed_text = typed_text.lower()
    suggestions = word_dict.get(typed_text, [])
    return suggestions[:max_suggestions]

# Function to get Arabic suggestions
def get_arabic_suggestions(typed_text, word_dict, max_suggestions=5):
    typed_text = typed_text.lower()
    suggestions = word_dict.get(typed_text, [])
    return suggestions[:max_suggestions]

# Function to draw word suggestions
def draw_suggestions(image, suggestions, selected_index=0):
    img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    start_x, start_y = 1000, 170
    padding = 10
    color = (255, 255, 255)
    draw.rectangle([1000, 100, 1210, 150],fill=color, outline=(0, 0, 0), width=2)
    draw.text((1010,110), "Suggestions:", font=font, fill=(0,0,0))
    for i, suggestion in enumerate(suggestions):
        suggestion_color = (0, 255, 0) if i == selected_index else (255, 255, 255)
        draw.text((start_x, start_y), suggestion, font=font, fill=suggestion_color)
        start_y += 40 + padding

    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# Function to check if fingertip is on a key
def check_key_press(x, y, keys):
    for key in keys:
        (x1, y1), (x2, y2) = key['pos']
        if x1 <= x <= x2 and y1 <= y <= y2:
            return key
    return None
# Function to draw the keyboard on the image using PIL
def draw_keyboard(image, keys):
    img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    for key in keys:
        (x1, y1), (x2, y2) = key['pos']
        color = (0, 255, 0) if key['pressed'] else (255, 255, 255)
        draw.rectangle([x1, y1, x2, y2], fill=color, outline=(0, 0, 0), width=2)

        char_label = key.get('label', key['char'])
        if current_language == 'Arabic' and char_label not in ['LANGUAGE', 'SPACE', 'BACKSPACE']:
            char_label = get_display(arabic_reshaper.reshape(char_label))

        draw.text((x1 + 5, y1 + 5), char_label, font=font, fill=(0, 0, 0))

    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


# Function to draw all hand landmarks
def draw_hand_landmarks(image, hand_landmarks, mp_hands):
    for idx, landmark in enumerate(hand_landmarks.landmark):
        x = int(landmark.x * image.shape[1])
        y = int(landmark.y * image.shape[0])
        if idx == mp_hands.HandLandmark.INDEX_FINGER_TIP:
            cv2.circle(image, (x, y), 10, (0, 255, 0), -1)
        else:
            cv2.circle(image, (x, y), 5, (255, 0, 0), -1)
    return hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

# Main loop frame function
def generate_frames():
    global typed_text, keys, key_pressed, current_language, selected_suggestion_index

    font_path = "arial.ttf"
    font = ImageFont.truetype(font_path, 32)

    while camera.isOpened():
        success, image = camera.read()
        if not success:
            break

        image = cv2.flip(image, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                index_fingertip = draw_hand_landmarks(image, hand_landmarks, mp_hands)
                x, y = int(index_fingertip.x * image.shape[1]), int(index_fingertip.y * image.shape[0])

                key = check_key_press(x, y, keys)
                if key:
                    current_time = time.time()
                    if not key['pressed']:
                        key['start_time'] = current_time
                        key['pressed'] = True
                    else:
                        if current_time - key['start_time'] >= key_press_delay:
                            if not key_pressed[key['char']]:
                                if key['char'] == 'Backspace':
                                    typed_text = typed_text[:-1]
                                    
                                elif key['char'] == 'Language': # change the curent language "English/Arabic"
                                    current_language = 'Arabic' if current_language == 'English' else 'English'
                                    keys = arabic_keys if current_language == 'Arabic' else english_keys
                                    selected_suggestion_index = 0
                                elif key['char'] == 'NextSuggestion': # move to the next word of suggestion words by the index
                                    selected_suggestion_index = (selected_suggestion_index + 1) % len(suggestions)
                                elif key['char'] == 'SelectSuggestion' and suggestions: # get the selected suggestion word
                                    words = typed_text.split(' ')
                                    if len(words) > 0:
                                        selected_word = suggestions[selected_suggestion_index]
                                        if current_language == 'Arabic': # Reshape Arabic suggestion if in Arabic language
                                            reshaped_word = arabic_reshaper.reshape(selected_word)
                                            bidi_word = get_display(reshaped_word)
                                            words[-1] = bidi_word  # Replace the last word with the suggestion
                                        else:
                                            words[-1] = selected_word  # For English, no reshaping is needed 

                                        typed_text = ' '.join(words) + " " # Update the typed text
                                        selected_suggestion_index = 0 # return the value to the defult
                                else:
                                    typed_text += key['char']
                                key_pressed[key['char']] = True
                else:
                    for key in keys:
                        key['pressed'] = False
                        key_pressed[key['char']] = False

        suggestions = get_suggestions(typed_text, current_language, english_word_dict, arabic_word_dict)
        image = draw_keyboard(image, keys)
        image = draw_suggestions(image, suggestions, selected_index=selected_suggestion_index)

        ret, buffer = cv2.imencode('.jpg', image)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


# API endpoint to stream the video
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# Serve the index.html file
@app.route('/')
def index():
    return render_template('test.html')

# API endpoint to get the current typed text (for real-time text update)
@app.route('/get_text', methods=['GET'])
def get_text():
    global typed_text
    return jsonify({"text": typed_text})

# # Endpoint to handle chat requests
@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('input')
    if user_input:
        chat_session = model.start_chat(history=[{"role": "user", "parts": [user_input]}])
        response = chat_session.send_message(user_input)
        return jsonify({"response": response.text})
    else:
        return jsonify({"error": "No input provided"}), 400


if __name__ == '__main__':
    app.run(debug=True)