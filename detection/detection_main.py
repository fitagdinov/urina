from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
def model_init(path= "best.pt"):
    model = YOLO(path)
    return model
def result(model,img):
    results = model.predict(img, stream=True)
    boxes=[]
    for result in results:
        boxes += result.boxes.cpu().numpy()  # Get boxes on CPU in numpy format
    return [box.xyxy[0].astype(int) for box in boxes]
def show(model,img):
    boxes = result(model,img)
    for r in boxes:
        cv2.rectangle(img, r[:2], r[2:], (0, 255, 0), 2)  # Draw boxes on the image

    dim = (640, 480)
    img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    print(img.shape)
    cv2.imshow('The best image in the world/ URINA', img)
    cv2.waitKey()
    cv2.destroyAllWindows()
if __name__ == '__main__':
    img = cv2.imread('../data/not_used(test)/20240116_002642.jpg')
    model = model_init(path= "best.pt")
    show(model, img)