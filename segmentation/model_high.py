import json
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from segment_anything import SamPredictor, sam_model_registry, SamAutomaticMaskGenerator
import time
import matplotlib.pyplot as plt
import os
import rawpy
def read_dng(path):
    with rawpy.imread(path) as raw:
        rgb = raw.postprocess(use_camera_wb=False)
    return rgb
class TwoStageSegmentation(object):
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'using {self.device}')

        sam = sam_model_registry['vit_b'](checkpoint='sam_vit_b.pth')
        # sam = sam_model_registry['vit_h'](checkpoint='sam_vit_h.pth')
        sam = sam.to(self.device)
        self.mask_generator = SamAutomaticMaskGenerator(sam)

    def _crop_image(self, image, ksize=5, scale=1, delta=0, ddepth=cv2.CV_16S, additional=0, mv=.6, mh=.7):
        blurred = cv2.GaussianBlur(image, (7, 7), 0)
        gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

        # gradients for sobel
        grad_x = cv2.Sobel(gray, ddepth, 1, 0, ksize=ksize, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
        grad_y = cv2.Sobel(gray, ddepth, 0, 1, ksize=ksize, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)

        abs_grad_x = cv2.convertScaleAbs(grad_x)
        abs_grad_y = cv2.convertScaleAbs(grad_y)

        grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
        grad_x_normalized = cv2.normalize(grad_x, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        grad_y_normalized = cv2.normalize(grad_y, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        mask_v = np.abs(grad_y_normalized) > mv  # .7 для тёмного фона, .6 для светлого
        mask_h = np.abs(grad_x_normalized) > mh  # при ksize=3: .8 для горизонтально полосатого фона, .7 для гладкого; при ksize=5: оба .7

        coords_y = np.argwhere(mask_v)
        coords_x = np.argwhere(mask_h)

        # bounding box of non-black pixels
        x0 = max(coords_x.min(axis=0)[0] - additional,0)
        x1 = coords_x.max(axis=0)[0] + additional
        y0 = max(coords_y.min(axis=0)[1] - additional,0)
        y1 = coords_y.max(axis=0)[1] + additional


        # x0 = max(coords_x.min(axis=0)[0] - additional,0)
        # x1 = coords_x.max(axis=0)[0] + additional
        # y0 = max(coords_y.min(axis=0)[1] - 0,0)
        # y1 = coords_y.max(axis=0)[1] + 0

        # Пререпутанно какжется. Изображение 0  ось вертикальная
        return (image[x0:x1, y0:y1].copy(),(y0,x0))

    def _segment(self, cropped, box=None):
        if box is None:
            masks = self.mask_generator.generate(cropped)
        else:
            pass
        print(f'len of masks = {len(masks)}')

        # plt.figure()
        # plt.imshow(image)
        # show_anns(masks)
        # plt.axis('off')
        # plt.show()

        return masks

    def _get_squares(self, masks):
        squares = []
        for i in range(len(masks)):
            mask = masks[i]['segmentation']
            # здесь контур будет только один
            contours, hierarchy = cv2.findContours(np.array(mask, np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.04 * peri, True)
                if len(approx) == 4:
                    (x, y, w, h) = cv2.boundingRect(approx)
                    ar = w / float(h)
                    # если это квадрат, то aspect ratio будет около единицы + погрешности
                    shape = 'square' if ar >= 0.8 and ar <= 1.15 else 'rectangle'
                    if shape == 'square':
                        squares.append(approx)

        while len(squares) > 12:
            # здесь лучше не искать просто так самый маленький, а, например,
            # удалять аутлаер, который находить, например, по трём сигмам
            squares = sorted(squares, key=lambda x: np.sum(x))
            squares = squares[:-1]
        print(len(squares))

        # if len(squares) != 12: TODO
        return squares
    def get_result(self, title, with_crop=True):
        ext = title.split('.')[-1].lower()
        if ext == 'dng':
            image = read_dng(title)
            # image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        elif ext in ['jpg', 'jpeg', 'png']:
            image = cv2.imread(title)
            image = image[:,:,::-1]

        # тут нужно аккуратно учесть трейдоф между тем, чтобы не слишком сильно
        # уменьшать размеры исходных изображений, и тем, чтобы это самое
        # изображение поместилось в память и обработалось за разумное время

        scaled = 1

        while image.shape[0] * image.shape[1] >= 3000000:
            scale_percent = 80 # процент оригинальной фотографии
            scaled *= scale_percent / 100
            width = int(image.shape[1] * scale_percent / 100)
            height = int(image.shape[0] * scale_percent / 100)
            dim = (width, height)
            image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
            print(f'resized to {dim}, scaled = {scaled}')


        if with_crop:
            cropped,(x0,y0) = self._crop_image(image, additional=20)
        else:
            cropped = image

        masks = self._segment(cropped)

        # cropped_with_mask = image.copy()
        cropped_with_mask = cropped.copy()
        squares = self._get_squares(masks)
        squares = sorted(squares, key=lambda x: x[0][0][0])
        # cv2.drawContours(cropped_with_mask, [squares[0]], -1, (0, 255, 0), 3) # нарисовать только первый квадрат
        # cv2.drawContours(cropped_with_mask, squares[0:12], -1, (0, 255, 0), 3)  # нарисовать все квадраты
        return image, squares, scaled, (x0,y0),cropped_with_mask
model = TwoStageSegmentation()
def depth_transform(depth,coordinates):
    if type(depth) is float:
        depth_list=np.ones(len(coordinates),dtype=np.float32)*depth
    elif (type(depth) is np.ndarray or type(depth) is list):
        depth_list=depth
    else:
        #ERROR
        return
    box_coordinats=[]
    for i,squar in enumerate(coordinates):
        x=squar[:,0,0]
        y=squar[:,0,1]
        x1=np.min(x)
        x2=np.max(x)
        y1=np.min(y)
        y2=np.max(y)
        s_x=(x2-x1)//2
        s_y=(y2-y1)//2
        d=depth_list[i]
        x1=int(x1+d*s_x)
        x2=int(x2-d*s_x)
        y1=int(y1+d*s_y)
        y2=int(y2-d*s_y)
        box_coordinats.append([x1,y1,x2,y2])
    return box_coordinats

def main(path,
         depth=0.5):
    print(f'IMAGE #: {path}')
    start = time.time()
    result, coordinates, scale ,(x0,y0),cropped_with_mask= model.get_result(path)
    stop = time.time()
    print(f'two stage segmentation took {stop - start} seconds\n')

    # postprocessing. ROBERT
    # coordinates are points for rectengle. Not polygons

    #deep

    #resize for general image

    coordinates=sorted(coordinates, key=lambda x: np.mean(x,axis=(0))[0,0])
    #add white crop
    right_crop=coordinates[-1]
    x1=right_crop[0,0,0]
    x2=right_crop[-1,0,0]
    s_x=x2-x1
    white_crop=right_crop +  np.array([[int(s_x*2.3),0]])
    coordinates.append(white_crop)

    box_coordinats=depth_transform(depth,coordinates)
    global_coordinats = [i+np.array([x0,y0,x0,y0]) for i in box_coordinats]

    crop_list=[]
    for x1,y1,x2,y2 in global_coordinats:
        crop_list.append(result[y1:y2,x1:x2])
    return crop_list

def
data_folder = 'images'
r = []
g = []
b = []
for title in os.listdir(data_folder):
    r.append([])
    g.append([])
    b.append([])
    path = os.path.join(data_folder, title)
    ext = title.split('.')[-1].lower()
    if ext not in ['dng', 'png', 'jpg', 'jpeg']:
      continue
    depth = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    s = time.time()
    crops = main(path,depth)
    print('global time:', time.time()-s)
    response = {}
    for c in crops:
        plt.imshow(c[:,:,::-1])
    averages = []
    for i, crop in enumerate(crops):
        r_average = np.mean(crop[:, :, 0])
        g_average = np.mean(crop[:, :, 1])
        b_average = np.mean(crop[:, :, 2])
        response[i] = [r_average, g_average, b_average]
        averages.append(response[i])
        # print(response[i])
    white = averages[-1]
    for i, x in enumerate(averages):
      averages[i][0] = averages[i][0] * 255 / white[0]
      averages[i][1] = averages[i][1] * 255 / white[1]
      averages[i][2] = averages[i][2] * 255 / white[2]
      r[-1].append(averages[i][0])
      g[-1].append(averages[i][1])
      b[-1].append(averages[i][2])
      print(response[i])