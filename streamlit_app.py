import streamlit as st
import cv2
import numpy as np

st.title("AI-Driven Lane Detection and Navigation Path Prediction")

uploaded_video = st.file_uploader("Upload a road video", type=["mp4","avi","mov"])

frame_window = st.image([])

def preprocess(frame):

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)

    blur = cv2.GaussianBlur(enhanced,(5,5),0)

    edges = cv2.Canny(blur,50,150)

    return edges


def region_of_interest(img):

    height,width = img.shape

    mask = np.zeros_like(img)

    polygon = np.array([[
        (0,height),
        (width,height),
        (width//2,height//2)
    ]])

    cv2.fillPoly(mask,polygon,255)

    return cv2.bitwise_and(img,mask)


def detect_lines(img):

    lines = cv2.HoughLinesP(img,
                            2,
                            np.pi/180,
                            100,
                            minLineLength=40,
                            maxLineGap=50)

    return lines


def draw_lanes(frame,lines):

    centers = []

    if lines is None:
        return frame, centers

    for line in lines:

        x1,y1,x2,y2 = line[0]

        cv2.line(frame,(x1,y1),(x2,y2),(0,255,0),4)

        cx = int((x1+x2)/2)
        cy = int((y1+y2)/2)

        centers.append((cx,cy))

        cv2.circle(frame,(cx,cy),5,(255,0,0),-1)

    return frame, centers


def predict_direction(centers,width):

    if len(centers)==0:
        return "UNKNOWN"

    avg_x = np.mean([c[0] for c in centers])

    center = width/2

    if avg_x < center - 40:
        direction = "LEFT"

    elif avg_x > center + 40:
        direction = "RIGHT"

    else:
        direction = "STRAIGHT"

    return direction


if uploaded_video is not None:

    tfile = open("temp_video.mp4","wb")
    tfile.write(uploaded_video.read())

    cap = cv2.VideoCapture("temp_video.mp4")

    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:
            break

        height,width = frame.shape[:2]

        edges = preprocess(frame)
        cropped = region_of_interest(edges)
        lines = detect_lines(cropped)

        frame, centers = draw_lanes(frame,lines)

        direction = predict_direction(centers,width)

        cv2.putText(frame,
                    "Direction: "+direction,
                    (40,60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0,255,255),
                    3)

        frame_window.image(frame,channels="BGR")

    cap.release()