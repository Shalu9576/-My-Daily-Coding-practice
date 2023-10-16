import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from tracker import Tracker

model = YOLO('yolov8s.pt')

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        colorsBGR = [x, y]
        print(colorsBGR)

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

cap = cv2.VideoCapture('test1.mp4')

my_file = open("classes.txt", "r")
data = my_file.read()
class_list = data.split("\n")

count = 0
tracker = Tracker()

# Define the lines using the specified points
line1 = [(116, 362), (427, 398)]
line2 = [(591, 376), (870, 348)]

line1_c = set()
line2_c = set()

def distance_to_line(point, line_start, line_end):
    x0, y0 = point
    x1, y1 = line_start
    x2, y2 = line_end

    # Calculate the distance between the point and the line using the formula
    distance = np.abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1) / np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)

    return distance

while True:
    ret, frame = cap.read()
    if not ret:
        break

    count += 1
    if count % 3 != 0:
        continue
    frame = cv2.resize(frame, (1020, 500))

    results = model.predict(frame)

    a = results[0].boxes.boxes
    px = pd.DataFrame(a).astype("float")
    list = []

    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        d = int(row[5])

        c = class_list[d]
        list.append([x1, y1, x2, y2])

    bbox_idx = tracker.update(list)
    for bbox in bbox_idx:
        x3, y3, x4, y4, id = bbox
        cx = int(x3 + x4) // 2
        cy = int(y3 + y4) // 2

        # Check if the vehicle is close to the lines within a threshold distance
        distance_threshold = 10  # Adjust this threshold as needed
        if distance_to_line((cx, cy), line1[0], line1[1]) <= distance_threshold:
            cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 2)
            cv2.circle(frame, (x4, y4), 5, (255, 0, 255), -1)
            cv2.putText(frame, str(int(id)), (x3, y3), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 2)
            line1_c.add(id)

        if distance_to_line((cx, cy), line2[0], line2[1]) <= distance_threshold:
            cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 2)
            cv2.circle(frame, (x4, y4), 5, (255, 0, 255), -1)
            cv2.putText(frame, str(int(id)), (x3, y3), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 2)
            line2_c.add(id)
    first = (len(line1_c))
    second = (len(line2_c))
    cv2.putText(frame, str(first), (50, 80), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 0), 2)
    cv2.putText(frame, str(second), (596, 80), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 0), 2)
    cv2.line(frame, line1[0], line1[1], (0, 0, 255), 3)
    cv2.line(frame, line2[0], line2[1], (0, 0, 255), 3)

    # Display vehicle counts
    cv2.putText(frame, f'Number of Vehicles Entering', (7, 39), cv2.FONT_HERSHEY_SIMPLEX, 1, [128, 0, 128], thickness=2, lineType=cv2.LINE_AA)
    cv2.putText(frame, f'Number of Vehicles Leaving', (540, 39), cv2.FONT_HERSHEY_SIMPLEX, 1, [128, 0, 128], thickness=2, lineType=cv2.LINE_AA)

    cv2.imshow("RGB", frame)
    if cv2.waitKey(1person
bicycle
car
motorbike
aeroplane
bus
train
truck
boat
traffic light
fire hydrant
stop sign
parking meter
bench
bird
cat
dog
horse
sheep
cow
elephant
bear
zebra
giraffe
backpack
umbrella
handbag
tie
suitcase
frisbee
skis
snowboard
sports ball
kite
baseball bat
baseball glove
skateboard
surfboard
tennis racket
bottle
wine glass
cup
fork
knife
spoon
bowl
banana
apple
sandwich
orange
broccoli
carrot
hot dog
pizza
donut
cake
chair
sofa
pottedplant
bed
diningtable
toilet
tvmonitor
laptop
mouse
remote
keyboard
cell phone
microwave
oven
toaster
sink
refrigerator
book
clock
vase
scissors
teddy bear
hair drier
toothbrush) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
