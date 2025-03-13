import numpy as np
from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *
from tensorflow.keras.models import load_model
from collections import deque
import streamlit as st
import tempfile

# YOLO model
model = YOLO("Yolo-Weights/yolov8n.pt")

# Detect shoplifting model
shoplifting_model = load_model("Shoplifting detection/LRCN_model.h5")

SEQUENCE_LENGTH = 20  # frames number for a prediction
IMAGE_HEIGHT, IMAGE_WIDTH = 64, 64 # input image shape

# Tracking
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

frame_buffer = {} # save frames with id
skip_frames_window = 2  # step to skip frame
frame_count = 0
overlap_frame = 10

st.title("Smart Retail Surveillance System")
uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

if uploaded_file:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    cap = cv2.VideoCapture(tfile.name)
    stframe = st.empty()

    while cap.isOpened():

        success, img = cap.read()
        if not success:
            break

        results = model(img, stream=True, verbose=False)

        detections = np.empty((0, 5))

        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                conf = math.ceil(box.conf[0] * 100) / 100
                cls = int(box.cls[0])

                if cls == 0 and conf > 0.5: # person
                    current_arr = [x1, y1, x2, y2, conf]
                    detections = np.vstack((detections, current_arr))

        resultsTracker = tracker.update(detections)
        frame_count += 1

        for result in resultsTracker:
            x1, y1, x2, y2, id = map(int, result)
            w, h = x2-x1, y2-y1
                    
            cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))
            cvzone.putTextRect(img, f'id: {id}', (max(0, x1), max(35, y1)), scale=2, thickness=3, offset=10)

            cx, cy = x1 + w // 2, y1 + h // 2
            cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

            if frame_count % skip_frames_window != 0: continue

            # else: predict
            person_img = img[y1:y2, x1:x2]

            if person_img is not None and person_img.size > 0:
                # processing input image for shoplifting model
                person_img = cv2.resize(person_img, (IMAGE_WIDTH, IMAGE_HEIGHT))
                person_img = person_img / 255.0 
                person_img = np.expand_dims(person_img, axis=0)  # (1, H, W, 3)

                # save to buffer
                if id not in frame_buffer:
                    frame_buffer[id] = deque(maxlen=SEQUENCE_LENGTH)
                frame_buffer[id].append(person_img)

            # predict
            if len(frame_buffer[id]) == SEQUENCE_LENGTH:
                sequence = np.array(list(frame_buffer[id])) 

                temp_buffer_id = frame_buffer[id]
                frame_buffer[id].clear()
                frame_buffer[id].extend(list(temp_buffer_id)[-overlap_frame:]) # overlap
                
                prediction = shoplifting_model.predict(sequence)[0]
                shoplifting_prob = prediction[1]
                # print(prediction)

                cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))

                if shoplifting_prob >= 0.4:
                    cvzone.putTextRect(img, f'id: {id} Shoplifting!', (max(0, x1), max(35, y1)), scale=2, thickness=3, offset=10, colorR=(0, 0, 255))

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        stframe.image(img, channels="RGB")

    cap.release()

# streamlit run app.py