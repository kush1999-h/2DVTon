import cvzone
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from cvzone.PoseModule import PoseDetector
import os
import cv2
import numpy as np

from poseapi.poseapi import settings

detector = PoseDetector()
shirt_folder_path = "Resources/Shirts"
list_shirts = os.listdir(shirt_folder_path)
fixed_ratio = 262 / 190
shirt_ratio_height_width = 581 / 440
image_number = 1

@csrf_exempt
def process_frame(request):
    if request.method == "POST":
        # Get the image frame from the request
        frame = np.frombuffer(request.FILES['frame'].read(), dtype=np.uint8)
        img = cv2.imdecode(frame, cv2.IMREAD_COLOR)

        img = cv2.flip(img, 1)
        img = detector.findPose(img)
        lmList, bbox_info = detector.findPosition(img, bboxWithHands=False, draw=False)
        if lmList:
            lm11 = lmList[11][1:4]
            lm12 = lmList[12][1:4]

            width_of_shirt = int((lm11[0] - lm12[0]) * fixed_ratio)
            if width_of_shirt > 0 and lmList[11][2] < lmList[23][2]:
                img_shirt = cv2.imread(os.path.join(shirt_folder_path, list_shirts[image_number]), cv2.IMREAD_UNCHANGED)
                img_shirt = cv2.resize(img_shirt, (width_of_shirt, int(width_of_shirt * shirt_ratio_height_width)))
                current_scale = (lm11[0] - lm12[0]) / 190
                offset = int(44 * current_scale), int(48 * current_scale)

                try:
                    img = cvzone.overlayPNG(img, img_shirt, (lm12[0] - offset[0], lm12[1] - offset[1]))
                except:
                    pass

        # Save the processed image
        processed_image_path = os.path.join(settings.STATIC_ROOT, 'processed_image.png')
        cv2.imwrite(processed_image_path, img)

        return JsonResponse({'processed_image_path': processed_image_path})
    else:
        return JsonResponse({'message': 'Invalid request method'})
