import argparse
import time
import numpy as np
import cv2
import torch
from matplotlib import pyplot as plt
from PIL import Image 

from s3fd.nets import S3FDNet
from s3fd.box_utils import nms_

PATH_WEIGHT = '/mnt/weights/model_best_nomask.pth'


class S3FD():

    def __init__(self, device='cuda'):

        tstamp = time.time()
        self.device = device

        print('[S3FD] loading with', self.device)
        self.net = S3FDNet('test')
        state_dict = torch.load(PATH_WEIGHT, map_location=self.device)
        self.net.load_state_dict(state_dict)
        self.net.eval()
        print('[S3FD] finished loading (%.4f sec)' % (time.time() - tstamp))
    
    def detect_faces(self, image, conf_th=0.8, scales=[1]):

        w, h = image.shape[1], image.shape[0]

        bboxes = np.empty(shape=(0, 5))

        img_mean = np.array([104., 117., 123.])[:, np.newaxis, np.newaxis].astype('float32')

        with torch.no_grad():
            for s in scales:
                scaled_img = cv2.resize(image, dsize=(0, 0), fx=s, fy=s, interpolation=cv2.INTER_LINEAR)
                scaled_img = np.swapaxes(scaled_img, 1, 2)
                scaled_img = np.swapaxes(scaled_img, 1, 0)
                scaled_img = scaled_img[[2, 1, 0], :, :]
                scaled_img = scaled_img.astype('float32')
                scaled_img -= img_mean
                scaled_img = scaled_img[[2, 1, 0], :, :]
                x = torch.from_numpy(scaled_img).unsqueeze(0)
                y = self.net(x)

                detections = y.data
                scale = torch.Tensor([w, h, w, h])

                for i in range(detections.size(1)):
                    j = 0
                    while detections[0, i, j, 0] > conf_th:
                        score = detections[0, i, j, 0]
                        pt = (detections[0, i, j, 1:] * scale).cpu().numpy()
                        bbox = (pt[0], pt[1], pt[2], pt[3], score)
                        bboxes = np.vstack((bboxes, bbox))
                        j += 1

            keep = nms_(bboxes, 0.1)
            bboxes = bboxes[keep]

        return bboxes


def plot_figures(figures, nrows=1, ncols=1):
    _, axeslist = plt.subplots(ncols=ncols, nrows=nrows)
    for ind, title in enumerate(figures):
        axeslist.ravel()[ind].imshow(figures[title], cmap=plt.gray())
        axeslist.ravel()[ind].set_title(title)
    plt.tight_layout()
    plt.show()


def draw_bboxes(image, bounding_boxes, fill=0.0, thickness=3):
    
    # it will be returned
    output = image.copy()

    # fill with transparency
    if fill > 0.0:

        # fill inside bboxes
        img_fill = image.copy()
        for bbox in bounding_boxes:
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[2]), int(bbox[3]))
            img_fill = cv2.rectangle(img_fill, p1, p2, (0, 255, 0), -1)
        
        # overlay
        cv2.addWeighted(img_fill, fill, output, 1.0 - fill, 0, output)

    # edge with thickness
    for bbox in bounding_boxes:
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[2]), int(bbox[3]))
        green = int(bbox[4] * 255)
        output = cv2.rectangle(output, p1, p2, (255, green, 0), thickness)
    
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--image_path', type=str, default='test.jpg')
    args = parser.parse_args()

    # load image with cv in RGB.
    IMAGE_PATH = args.image_path
    img = cv2.imread(IMAGE_PATH)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # load detector.
    model = S3FD(device='cuda')

    t = time.time()
    bboxes = model.detect_faces(img, conf_th=0.95)
    print('S3FD : %d faces in %.4f seconds.' % (len(bboxes), time.time() - t))
    img1 = draw_bboxes(img, bboxes)
    sizes = []

    print(bboxes)

    '''
    for box in bboxes:
        sizes.append((box[2] - box[0]) * (box[3] - box[1]))
    print(min(sizes))
    print(max(sizes))
    '''

    pil_img = Image.fromarray(img1)
    pil_img.save('./result/test.jpg')


if __name__ == '__main__':
    main()
