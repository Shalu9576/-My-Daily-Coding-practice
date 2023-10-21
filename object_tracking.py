import math
import cv2
from object_detection1 import ObjectDetection, get_direction

# Initialize Object Detection
od = ObjectDetection()

cap = cv2.VideoCapture("test1.mp4")

# Initialize count
count = 0
count_up = 0
count_down = 0
center_points_prev_frame = []

tracking_objects = {}
track_id = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Draw blue and red lines
    cv2.line(frame, (300, 0), (300, frame.shape[0]), (255, 0, 0), 2)  # Blue line (existing)
    cv2.line(frame, (900, 0), (900, frame.shape[0]), (0, 0, 255), 2)  # Red line (entering)

    # Point current frame
    center_points_cur_frame = []

    # Detect objects on frame
    (class_ids, scores, boxes) = od.detect(frame)
    for box in boxes:
        (x, y, w, h) = box
        cx = int((x + x + w) / 2)
        cy = int((y + y + h) / 2)
        center_points_cur_frame.append((cx, cy))
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Only at the beginning we compare previous and current frame
    if count <= 2:
        for pt in center_points_cur_frame:
            for pt2 in center_points_prev_frame:
                distance = math.hypot(pt2[0] - pt[0], pt2[1] - pt[1])

                if distance < 20:
                    tracking_objects[track_id] = pt
                    track_id += 1
    else:
        tracking_objects_copy = tracking_objects.copy()
        center_points_cur_frame_copy = center_points_cur_frame.copy()

        for object_id, pt2 in tracking_objects_copy.items():
            object_exists = False
            for pt in center_points_cur_frame_copy:
                distance = math.hypot(pt2[0] - pt[0], pt2[1] - pt[1])

                # Update IDs position
                if distance < 20:
                    tracking_objects[object_id] = pt
                    object_exists = True
                    if pt in center_points_cur_frame:
                        center_points_cur_frame.remove(pt)
                    continue

            # Remove IDs lost
            if not object_exists:
                tracking_objects.pop(object_id)

        # Add new IDs found
        for pt in center_points_cur_frame:
            tracking_objects[track_id] = pt
            track_id += 1

    for object_id, pt in tracking_objects.items():
        cv2.circle(frame, pt, 5, (0, 0, 255), -1)
        cv2.putText(frame, str(object_id), (pt[0], pt[1] - 7), 0, 1, (0, 0, 255), 2)

    # Detect object intersections with lines and update counts
    for object_id, pt in tracking_objects.items():
        if pt[0] < 300 and od.crosses_line(pt, (300, 0), (300, frame.shape[0])):
            direction = get_direction(pt, (300, frame.shape[1]))  # Corrected the second argument
            if "North" in direction:
                count_up += 1
            elif "South" in direction:
                count_down += 1
        elif pt[0] > 900 and od.crosses_line(pt, (900, 0), (900, frame.shape[0])):
            direction = od.get_direction(pt, (900, frame.shape[1]))  # Corrected the second argument
            if "North" in direction:
                count_up += 1
            elif "South" in direction:
                count_down += 1

    # Display count up and count down on the frame
    cv2.putText(frame, "Count Up: " + str(count_up), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(frame, "Count Down: " + str(count_down), (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    print("Tracking objects")
    print(tracking_objects)

    print("CUR FRAME LEFT PTS")
    print(center_points_cur_frame)

    cv2.imshow("Frame", frame)

    # Make a copy of the points
    center_points_prev_frame = center_points_cur_frame.copy()

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
