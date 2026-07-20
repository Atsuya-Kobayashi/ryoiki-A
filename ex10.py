import cv2
from ultralytics import YOLO
import math


def find_snap_frame(video_path):

    model = YOLO("best260408.pt")

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("動画を開けません:", video_path)
        return None

    prev_centers = {}

    frame_count = 0
    stationary_count = 0

    distance_threshold = 6
    moving_threshold = 3
    stationary_threshold = 30
    confidence_threshold = 0.5

    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:
            break

        frame_count += 1

        results = model.track(
            frame,
            persist=True,
            conf=confidence_threshold,
            verbose=False
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

                if track_id in prev_centers:

                    px, py = prev_centers[track_id]

                    distance = math.sqrt(
                        (cx - px) ** 2 + (cy - py) ** 2
                    )

                    if distance >= distance_threshold:
                        moving_count += 1

                prev_centers[track_id] = (cx, cy)

        if moving_count >= moving_threshold:

            if stationary_count >= stationary_threshold:

                snap_frame = frame_count

                cap.release()

                return snap_frame

            stationary_count = 0

        else:
            stationary_count += 1

    cap.release()

    return None


snap_frame_ex5 = find_snap_frame("ex5-26.mp4")
snap_frame_ex9 = find_snap_frame("ex9-26.mp4")

print("ex5-26.mp4 Snap frame:", snap_frame_ex5)
print("ex9-26.mp4 Snap frame:", snap_frame_ex9)

if snap_frame_ex5 is not None and snap_frame_ex9 is not None:

    frame_difference = abs(
        snap_frame_ex5 - snap_frame_ex9
    )

    print("Difference:", frame_difference, "frames")

else:
    print("どちらかの動画でスナップを検出できませんでした。")