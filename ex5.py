import cv2
from ultralytics import YOLO

model = YOLO("best260408.pt")

video_path = "ex5-26.mp4"
cap = cv2.VideoCapture(video_path)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

out = cv2.VideoWriter(
    "result5.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)

frame_count = 0

while cap.isOpened():

    ret, frame = cap.read()

    if not ret:
        break

    frame_count += 1

    results = model(frame)

    helmet_count = 0

    for result in results:
        for box in result.boxes:

            helmet_count += 1

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 0, 255),
                2
            )

    if frame_count % 30 == 0:
        print(
            f"Frame {frame_count}: "
            f"Detected helmets = {helmet_count}"
        )

    cv2.putText(
        frame,
        f"Helmets: {helmet_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2
    )

    out.write(frame)

    cv2.imshow("Result", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()