from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
model = YOLO("best.pt")
img = cv2.imread('../data/not_used(test)/20240116_002642.jpg')
results = model.predict(img, stream=True)
for result in results:
    boxes = result.boxes.cpu().numpy()  # Get boxes on CPU in numpy format
    print(len(boxes))
    for box in boxes:  # Iterate over boxes
        r = box.xyxy[0].astype(int)  # Get corner points as int
        class_id = int(box.cls[0])  # Get class ID
        class_name = model.names[class_id]  # Get class name using the class ID
        print(f"Class: {class_name}, Box: {r}")  # Print class name and box coordinates
        cv2.rectangle(img, r[:2], r[2:], (0, 255, 0), 2)  # Draw boxes on the image
# img = cv2.resize(img,(680,320))
scale_percent = 20  # percent of original size
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)
print(dim)
dim=(640, 480)
# resize image
img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
print(img.shape)
cv2.imshow('im', img)
# plt.imshow(img)
cv2.waitKey()
cv2.destroyAllWindows()
