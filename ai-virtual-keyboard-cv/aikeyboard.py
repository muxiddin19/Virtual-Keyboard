import cv2
from cvzone.HandTrackingModule import HandDetector
from time import sleep
import numpy as np
import cvzone
from pynput.keyboard import Controller
import math


cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)


detector = HandDetector(detectionCon=0.8)


keys = [
    ["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P"],
    ["A", "S", "D", "F", "G", "H", "J", "K", "L", ";"],
    ["Z", "X", "C", "V", "B", "N", "M", ",", ".", "/"],
    ["Space", "Backspace"]
]
finalText = ""

keyboard = Controller()



class Button():
    def __init__(self, pos, text, size=[85, 85]):
        self.pos = pos
        self.size = size
        self.text = text



buttonList = []
for i in range(len(keys)):
    for j, key in enumerate(keys[i]):

        if key == "Space":
            buttonList.append(Button([100 * j + 50, 100 * i + 50], key, size=[500, 85]))
        elif key == "Backspace":
            buttonList.append(Button([700, 350], key, size=[150, 85]))
        else:
            buttonList.append(Button([100 * j + 50, 100 * i + 50], key))



def drawAll(img, buttonList):
    for button in buttonList:
        x, y = button.pos
        w, h = button.size
        cvzone.cornerRect(img, (x, y, w, h), 20, rt=0)
        cv2.rectangle(img, button.pos, (x + w, y + h), (255, 0, 255), cv2.FILLED)
        cv2.putText(img, button.text, (x + 20, y + 65), cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 4)
    return img



def findDistanceCustom(lmList, p1, p2):
    x1, y1 = lmList[p1][0], lmList[p1][1]
    x2, y2 = lmList[p2][0], lmList[p2][1]
    distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return distance



while True:
    success, img = cap.read()


    if not success or img is None:
        print("Error: Failed to capture image. Check your camera.")
        continue


    hands, img = detector.findHands(img, flipType=False)


    img = drawAll(img, buttonList)

    if hands:
        lmList = hands[0]['lmList']

        for button in buttonList:
            x, y = button.pos
            w, h = button.size


            if x < lmList[8][0] < x + w and y < lmList[8][1] < y + h:
                cv2.rectangle(img, (x - 5, y - 5), (x + w + 5, y + h + 5), (175, 0, 175), cv2.FILLED)
                cv2.putText(img, button.text, (x + 20, y + 65), cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 4)


                l = findDistanceCustom(lmList, 8, 12)
                print(f"Distance: {l}")


                if l < 30:
                    if button.text == "Space":
                        finalText += " "
                    elif button.text == "Backspace":
                        finalText = finalText[:-1] if len(finalText) > 0 else finalText
                    else:
                        keyboard.press(button.text)
                        finalText += button.text
                    sleep(0.15)


    cv2.rectangle(img, (50, 500), (1200, 600), (175, 0, 175), cv2.FILLED)
    cv2.putText(img, finalText, (60, 570), cv2.FONT_HERSHEY_PLAIN, 5, (255, 255, 255), 5)


    cv2.imshow("Virtual AI Keyboard", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
