# AI-Driven Lane Detection and Navigation Path Prediction
# Night Video

import cv2
import numpy as np
import os

VIDEO_PATH = "road_night.mp4"
OUTPUT_PATH = "output_lane_night_fixed.mp4"

# ---------------- PREPROCESSING ---------------- #

def preprocess(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # stronger contrast enhancement for night scenes
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)

    blur = cv2.GaussianBlur(enhanced, (5,5), 0)

    # lower thresholds detect faint lane markings
    edges = cv2.Canny(blur, 30, 120)

    return edges


# ---------------- ROI ---------------- #

def region_of_interest(img):

    h, w = img.shape
    mask = np.zeros_like(img)

    polygon = np.array([[
        (int(w*0.05), h),
        (int(w*0.95), h),
        (int(w*0.60), int(h*0.60)),
        (int(w*0.40), int(h*0.60)),
    ]])

    cv2.fillPoly(mask, polygon, 255)

    return cv2.bitwise_and(img, mask)


# ---------------- HOUGH TRANSFORM ---------------- #

def detect_lines(img):

    lines = cv2.HoughLinesP(
        img,
        rho=1,
        theta=np.pi/180,
        threshold=60,
        minLineLength=50,
        maxLineGap=80
    )

    return lines


# ---------------- LANE AVERAGING ---------------- #

def average_lane_lines(frame, lines):

    left_lines = []
    right_lines = []

    if lines is None:
        return frame, []

    h, w = frame.shape[:2]

    for line in lines:

        x1,y1,x2,y2 = line[0]

        if x2 == x1:
            continue

        slope = (y2-y1)/(x2-x1)
        intercept = y1 - slope*x1

        if slope < -0.3:
            left_lines.append((slope,intercept))

        elif slope > 0.3:
            right_lines.append((slope,intercept))


    def make_line(data):

        if len(data) == 0:
            return None

        slope = np.mean([d[0] for d in data])
        intercept = np.mean([d[1] for d in data])

        y1 = h
        y2 = int(h*0.6)

        x1 = int((y1-intercept)/slope)
        x2 = int((y2-intercept)/slope)

        return (x1,y1,x2,y2)


    left = make_line(left_lines)
    right = make_line(right_lines)

    overlay = frame.copy()
    centers = []

    if left:
        cv2.line(frame,(left[0],left[1]),(left[2],left[3]),(0,255,0),6)
        centers.append(((left[0]+left[2])//2,(left[1]+left[3])//2))

    if right:
        cv2.line(frame,(right[0],right[1]),(right[2],right[3]),(0,255,0),6)
        centers.append(((right[0]+right[2])//2,(right[1]+right[3])//2))


    # fill lane region
    if left and right:

        pts = np.array([
            [left[0],left[1]],
            [left[2],left[3]],
            [right[2],right[3]],
            [right[0],right[1]]
        ])

        cv2.fillPoly(overlay,[pts],(0,200,0))

        frame = cv2.addWeighted(frame,0.8,overlay,0.2,0)

    return frame, centers


# ---------------- PATH PREDICTION ---------------- #

def predict_direction(centers, width):

    if len(centers) == 0:
        return "UNKNOWN",0

    avg_x = np.mean([c[0] for c in centers])

    center = width/2

    deviation = avg_x-center


    if deviation < -50:
        direction = "LEFT"

    elif deviation > 50:
        direction = "RIGHT"

    else:
        direction = "STRAIGHT"


    return direction, deviation


# ---------------- DRAW ARROW ---------------- #

def draw_arrow(frame,direction):

    h,w = frame.shape[:2]

    start = (w//2,h-40)

    if direction == "LEFT":
        end = (w//2-120,h-150)

    elif direction == "RIGHT":
        end = (w//2+120,h-150)

    else:
        end = (w//2,h-150)


    cv2.arrowedLine(
        frame,
        start,
        end,
        (0,0,255),
        6,
        tipLength=0.3
    )

    return frame


# ---------------- TEXT ---------------- #

def annotate_frame(frame,direction,deviation):

    h,w = frame.shape[:2]

    overlay = frame.copy()

    cv2.rectangle(overlay,(20,20),(420,110),(0,0,0),-1)

    frame = cv2.addWeighted(overlay,0.4,frame,0.6,0)


    cv2.putText(
        frame,
        f"Direction: {direction}",
        (40,60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,255,255),
        2
    )


    cv2.putText(
        frame,
        f"Deviation: {deviation:.1f}px",
        (40,95),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (220,220,220),
        2
    )


    if abs(deviation) > 90:

        cv2.putText(
            frame,
            "LANE DEPARTURE!",
            (w//2-170,60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0,0,255),
            3
        )

    return frame


# ---------------- MAIN ---------------- #

if not os.path.exists(VIDEO_PATH):

    print("Video not found")
    exit()


cap = cv2.VideoCapture(VIDEO_PATH)

fps = cap.get(cv2.CAP_PROP_FPS) or 25

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


fourcc = cv2.VideoWriter_fourcc(*"mp4v")

out = cv2.VideoWriter(
    OUTPUT_PATH,
    fourcc,
    fps,
    (w,h)
)

frame_count = 0


while True:

    ret,frame = cap.read()

    if not ret:
        break


    edges = preprocess(frame)

    roi = region_of_interest(edges)

    lines = detect_lines(roi)

    frame,centers = average_lane_lines(frame,lines)

    direction,deviation = predict_direction(centers,w)

    frame = draw_arrow(frame,direction)

    frame = annotate_frame(frame,direction,deviation)


    out.write(frame)


    cv2.imshow("Lane Detection",frame)

    if cv2.waitKey(1) == 27:
        break


    frame_count += 1


cap.release()

out.release()

cv2.destroyAllWindows()

print("Frames processed:",frame_count)

print("Saved to:",OUTPUT_PATH)