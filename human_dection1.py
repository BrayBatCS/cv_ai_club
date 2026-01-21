from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")  

video = cv2.VideoCapture(0)

while True:
    ret, frame = video.read()
    if not ret:
        break

    results = model(frame)

    for result in results:
        annotated_frame = result.plot()

    cv2.imshow("YOLOv8 Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break