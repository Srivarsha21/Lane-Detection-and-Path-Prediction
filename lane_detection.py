# AI-Driven Lane Detection and Navigation Path Prediction
# Main Program

import cv2
import numpy as np
import os

# Preprocessing Module

def preprocess(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blur = cv2.GaussianBlur(enhanced, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    return edges

# Region of Interest

def region_of_interest(img):
    height, width = img.shape
    mask = np.zeros_like(img)
    polygon = np.array([[
        (0, height),
        (width, height),
        (int(width * 0.55), int(height * 0.55)),
        (int(width * 0.45), int(height * 0.55)),
    ]])
    cv2.fillPoly(mask, polygon, 255)
    return cv2.bitwise_and(img, mask)

# Lane Line Detection

def detect_lines(img):
    lines = cv2.HoughLinesP(img,
                            2,
                            np.pi / 180,
                            100,
                            minLineLength=40,
                            maxLineGap=50)
    return lines

# Separate left/right lanes and average them

def average_lane_lines(frame, lines):
    left_lines = []
    right_lines = []

    if lines is None:
        return frame, []

    height, width = frame.shape[:2]

    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 == x1:
            continue
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1

        if slope < -0.3:
            left_lines.append((slope, intercept))
        elif slope > 0.3:
            right_lines.append((slope, intercept))

    def make_line(lines_list, y1, y2):
        if not lines_list:
            return None
        avg_slope = np.mean([l[0] for l in lines_list])
        avg_intercept = np.mean([l[1] for l in lines_list])
        x1 = int((y1 - avg_intercept) / avg_slope)
        x2 = int((y2 - avg_intercept) / avg_slope)
        return (x1, y1, x2, y2)

    y_bottom = height
    y_top = int(height * 0.6)

    centers = []
    overlay = frame.copy()

    left = make_line(left_lines, y_bottom, y_top)
    right = make_line(right_lines, y_bottom, y_top)

    if left:
        cv2.line(frame, (left[0], left[1]), (left[2], left[3]), (0, 255, 0), 5)
        centers.append((int((left[0] + left[2]) / 2), int((left[1] + left[3]) / 2)))

    if right:
        cv2.line(frame, (right[0], right[1]), (right[2], right[3]), (0, 255, 0), 5)
        centers.append((int((right[0] + right[2]) / 2), int((right[1] + right[3]) / 2)))

    # Draw filled lane polygon if both lines found
    if left and right:
        pts = np.array([
            [left[0], left[1]], [left[2], left[3]],
            [right[2], right[3]], [right[0], right[1]]
        ], dtype=np.int32)
        cv2.fillPoly(overlay, [pts], (0, 200, 0))
        cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)

    return frame, centers

# Path Prediction

def predict_direction(centers, width):
    if len(centers) == 0:
        return "UNKNOWN", 0

    avg_x = np.mean([c[0] for c in centers])
    center = width / 2
    deviation = avg_x - center

    if deviation < -40:
        direction = "LEFT"
    elif deviation > 40:
        direction = "RIGHT"
    else:
        direction = "STRAIGHT"

    return direction, deviation

# Steering Arrow

def draw_arrow(frame, direction):
    h, w = frame.shape[:2]
    start = (w // 2, h - 40)

    if direction == "LEFT":
        end = (w // 2 - 120, h - 150)
    elif direction == "RIGHT":
        end = (w // 2 + 120, h - 150)
    else:
        end = (w // 2, h - 150)

    cv2.arrowedLine(frame, start, end, (0, 0, 255), 5, tipLength=0.3)
    return frame

# Annotate frame

def annotate_frame(frame, direction, deviation):
    h, w = frame.shape[:2]

    overlay = frame.copy()
    cv2.rectangle(overlay, (20, 20), (400, 100), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)

    cv2.putText(frame,
                f"Direction: {direction}",
                (35, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.putText(frame,
                f"Deviation: {deviation:.1f}px",
                (35, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    if abs(deviation) > 80:
        cv2.putText(frame,
                    "!! LANE DEPARTURE WARNING",
                    (w // 2 - 230, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    return frame

# MAIN LOOP

VIDEO_PATH = "road_video.mp4"
OUTPUT_PATH = "output_lane_detection.mp4"

if not os.path.exists(VIDEO_PATH):
    print(f"[ERROR] Video file '{VIDEO_PATH}' not found.")
    exit(1)

cap = cv2.VideoCapture(VIDEO_PATH)

fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

print(f"[INFO] Processing: {VIDEO_PATH} ({width}x{height} @ {fps:.1f}fps)")
frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    edges = preprocess(frame)
    cropped = region_of_interest(edges)
    lines = detect_lines(cropped)
    frame, centers = average_lane_lines(frame, lines)
    direction, deviation = predict_direction(centers, width)
    frame = draw_arrow(frame, direction)
    frame = annotate_frame(frame, direction, deviation)

    out.write(frame)

    if frame_idx == 10:
        cv2.imwrite("output_sample.png", frame)
        print("[INFO] Sample frame saved: output_sample.png")

    frame_idx += 1

    # Uncomment below to display live (requires a display/GUI):
    cv2.imshow("Lane Detection", frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
out.release()
try:
    cv2.destroyAllWindows()
except Exception:
    pass  # No display available (headless environment)

print(f"[INFO] Done. Processed {frame_idx} frames.")
print(f"[INFO] Output saved to: {OUTPUT_PATH}")