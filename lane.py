import cv2
import numpy as np


previous_left_lane = None
previous_right_lane = None


def detect_lanes(frame):
    
    hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)

    # Define color ranges for yellow and white
    lower_yellow = np.array([15, 38, 115], dtype=np.uint8)
    upper_yellow = np.array([35, 204, 255], dtype=np.uint8)
    lower_white = np.array([0, 200, 0], dtype=np.uint8)
    upper_white = np.array([255, 255, 255], dtype=np.uint8)

    # Threshold the frame to get yellow and white regions
    mask_yellow = cv2.inRange(hls, lower_yellow, upper_yellow)
    mask_white = cv2.inRange(hls, lower_white, upper_white)
    mask_lane = cv2.bitwise_or(mask_yellow, mask_white)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(mask_lane, (5, 5), 0)

    # Perform Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)

    # Define a region of interest (ROI)
    height, width = frame.shape[:2]
    roi_vertices = [(0, height), (width / 2, height / 2), (width, height)]
    mask = np.zeros_like(edges)
    cv2.fillPoly(mask, np.array([roi_vertices], dtype=np.int32), 255)
    masked_edges = cv2.bitwise_and(edges, mask)

    # Apply Hough line transformation to detect lanes
    lines = cv2.HoughLinesP(masked_edges, 1, np.pi / 180, threshold=50, minLineLength=100, maxLineGap=50)

    return lines


def estimate_lane(lines):
    # Separate lines into left and right lane candidates
    left_lines = []
    right_lines = []

    for line in lines:
        x1, y1, x2, y2 = line[0]

        # Calculate slope and intercept of the line
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1

        # Filter out lines with extreme slopes or near-horizontal slopes
        if abs(slope) > 0.5 and abs(slope) < 10:
            if slope < 0:
                left_lines.append((slope, intercept))
            else:
                right_lines.append((slope, intercept))

    # Calculate average slope and intercept for left and right lanes
    left_lane = np.average(left_lines, axis=0) if len(left_lines) > 0 else None
    right_lane = np.average(right_lines, axis=0) if len(right_lines) > 0 else None

    return left_lane, right_lane


def draw_lane(frame, left_lane, right_lane):
    global previous_left_lane, previous_right_lane

    # Define y-coordinate range for drawing the lane lines
    y1 = frame.shape[0]
    y2 = int(y1 / 2)

    # Extrapolate the lane lines from the bottom to the top of the ROI
    if left_lane is not None:
        slope, intercept = left_lane
        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)
        left_lane = (x1, y1, x2, y2)
    else:
        left_lane = previous_left_lane

    if right_lane is not None:
        slope, intercept = right_lane
        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)
        right_lane = (x1, y1, x2, y2)
    else:
        right_lane = previous_right_lane

    # Update the previous lane lines for the next frame
    previous_left_lane = left_lane
    previous_right_lane = right_lane

    # Draw the lane lines on the frame
    if left_lane is not None:
        cv2.line(frame, (left_lane[0], left_lane[1]), (left_lane[2], left_lane[3]), (0, 255, 0), 5)
    if right_lane is not None:
        cv2.line(frame, (right_lane[0], right_lane[1]), (right_lane[2], right_lane[3]), (0, 255, 0), 5)

    return frame




def detect_objects(frame):
    
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    boxes, _ = hog.detectMultiScale(frame, winStride=(8, 8), padding=(2, 2), scale=1.05)

    return boxes


def process_frame(frame):
    # Detect lanes
    lanes = detect_lanes(frame)

    # Detect objects
    objects = detect_objects(frame)

    # Draw lanes
    if lanes is not None:
        for line in lanes:
            x1, y1, x2, y2 = line[0]
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)

    # Draw objects
    if objects is not None:
        for (x, y, w, h) in objects:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    return frame


# Input and output video files
input_video = 'input.mp4'
output_video = 'output.mp4'

# Open the input video file
video_capture = cv2.VideoCapture(input_video)
fps = video_capture.get(cv2.CAP_PROP_FPS)
frame_size = (int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))

# Define the codec for output video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video, fourcc, fps, frame_size)

# Process each frame in the input video
while video_capture.isOpened():
    ret, frame = video_capture.read()
    if not ret:
        break

    # Process the frame
    processed_frame = process_frame(frame)

    # Write the processed frame to the output video file
    out.write(processed_frame)

    # Display the processed frame
    cv2.imshow('Output', processed_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and writer
video_capture.release()
out.release()

# Close all windows
cv2.destroyAllWindows()
