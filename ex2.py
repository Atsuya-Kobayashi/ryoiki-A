import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

image_path = "ex2-26.png"
image = cv2.imread(image_path)

results = model(image_path, conf=0.16)

person_count = 0

for result in results:
    boxes = result.boxes

    for box in boxes:
        cls = int(box.cls[0])

        if cls == 0:
            person_count += 1

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                (0, 0, 255),
                2
            )

print("Detected persons:", person_count)

cv2.putText(
    image,
    f"Persons: {person_count}",
    (20, 40),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (0, 0, 255),
    2
)

cv2.imwrite("result2.png", image)

cv2.imshow("Result", image)
cv2.waitKey(0)
cv2.destroyAllWindows()