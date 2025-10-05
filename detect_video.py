import cv2
from ultralytics import YOLO

model = YOLO("yolo12n.pt")

video_path = "cow_video2.mp4"

capture = cv2.VideoCapture(video_path)

print("Press 'q' to exit.")

while True:
    ret, frame = capture.read()
    if not ret:
        break

    results = model(frame, conf=0.5)

    detections = results[0]

    annotated_frame = detections.plot()

    cv2.imshow("Video Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
