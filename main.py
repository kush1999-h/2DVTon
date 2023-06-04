import cv2
import cvzone
from cvzone.PoseModule import PoseDetector
import os
import time

cap = cv2.VideoCapture(0)
detector = PoseDetector()

shirt_folder_path = "Resources/Shirts"
list_shirts = os.listdir(shirt_folder_path)
fixed_ratio = 262 / 190  # widthOfShirt/widthOfPoint11to12
shirt_ratio_height_width = 581 / 440
image_number = 1

default_shirt_width = 200

left_hand_raised_time = 0
right_hand_raised_time = 0
hand_raise_duration = 5  # 5 seconds

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    img = detector.findPose(img)
    lmList, bbox_info = detector.findPosition(img, bboxWithHands=False, draw=False)

    if lmList:
        # Get the coordinates of the left and right hands
        left_hand_coordinates = lmList[15][1:4]
        right_hand_coordinates = lmList[16][1:4]

        # Calculate the distance between left and right hands
        hand_distance = right_hand_coordinates[1] - left_hand_coordinates[1]

        # Check if the left hand is raised
        if hand_distance > 50:
            left_hand_raised_time += 1
            right_hand_raised_time = 0

            if left_hand_raised_time >= hand_raise_duration:
                left_hand_raised_time = 0
                if image_number < len(list_shirts) - 1:
                    image_number += 1

        # Check if the right hand is raised
        elif hand_distance < -50:
            right_hand_raised_time += 1
            left_hand_raised_time = 0

            if right_hand_raised_time >= hand_raise_duration:
                right_hand_raised_time = 0
                if image_number > 0:
                    image_number -= 1

    if lmList:
        lm11 = lmList[11][1:4]
        lm12 = lmList[12][1:4]
        lm23 = lmList[23][1:4]
        lm24 = lmList[24][1:4]
        # Calculate the desired width of the shirt based on the distance between landmarks
        width_of_shirt = int((lm11[0] - lm12[0]) * fixed_ratio)
        length_of_shirt = int(abs(lm11[1] - lm23[1]) * fixed_ratio)
        print(width_of_shirt)
        print(length_of_shirt)
        # Check if the width is valid and the person is facing the front
        if width_of_shirt > 0 and lmList[11][2] < lmList[23][2]:
            print(lm11)
            img_shirt = cv2.imread(os.path.join(shirt_folder_path, list_shirts[image_number]), cv2.IMREAD_UNCHANGED)
            img_shirt = cv2.resize(img_shirt, (width_of_shirt, length_of_shirt))
            current_scale = (lm11[0] - lm12[0]) / 190
            offset = int(44 * current_scale), int(48 * current_scale)

            try:
                img = cvzone.overlayPNG(img, img_shirt, (lm12[0] - offset[0], lm12[1] - offset[1]))
            except:
                pass

    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == ord('a'):
        if image_number > 0:
            image_number -= 1
    elif key == ord('d'):
        if image_number < len(list_shirts) - 1:
            image_number += 1
    elif key == 27:
        break

cap.release()
cv2.destroyAllWindows()
