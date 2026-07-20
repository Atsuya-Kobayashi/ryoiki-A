import cv2
from ultralytics import YOLO
import math

model = YOLO("best260408.pt")

video_path = "ex9-26.mp4"
cap = cv2.VideoCapture(video_path)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

out = cv2.VideoWriter(
    "result9.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)

prev_centers = {}

frame_count = 0

stationary_count = 0

while cap.isOpened():

    ret, frame = cap.read()

    if not ret:
        break

    frame_count += 1

    results = model.track(
        frame,
        persist=True
    )

    moving_count = 0

    if results[0].boxes.id is not None:

        boxes = results[0].boxes.xyxy.cpu().numpy()
        ids = results[0].boxes.id.cpu().numpy()

        for box, track_id in zip(boxes, ids):

            track_id = int(track_id)

            x1, y1, x2, y2 = map(int, box)

            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            color = (0, 255, 0)

            if track_id in prev_centers:

                px, py = prev_centers[track_id]

                distance = math.sqrt(
                    (cx - px) ** 2 + (cy - py) ** 2
                )

                if distance >= 4:
                    color = (0, 0, 255)
                    moving_count += 1
                else:
                    color = (0, 255, 0)

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                color,
                2
            )

            prev_centers[track_id] = (cx, cy)

    if moving_count >= 3:

        message = "TEAM MOVING"
        text_color = (0, 0, 255)

        if stationary_count >= 30:

            print("Snap frame:", frame_count)

            cv2.putText(
                frame,
                f"SNAP FRAME: {frame_count}",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )

        stationary_count = 0

    else:

        message = "TEAM STATIONARY"
        text_color = (0, 255, 0)

        stationary_count += 1

    cv2.putText(
        frame,
        message,
        (width - 360, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        text_color,
        2
    )

    out.write(frame)

    cv2.imshow("Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()