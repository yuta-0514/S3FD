import numpy as np
import os
import time
import cv2
import argparse
import torch
from matplotlib import pyplot as plt
from PIL import Image

from s3fd.nets import S3FDNet
from s3fd.box_utils import nms_

 
PATH_WEIGHT = '/mnt/weights/model_best_nomask.pth'
img_mean = np.array([104., 117., 123.])[:, np.newaxis, np.newaxis].astype('float32')
IMAGE_DIR = '/mnt/MAFA/images/'


class S3FD():

    def __init__(self, device='cuda'):

        tstamp = time.time()
        self.device = device

        print('[S3FD] loading with', self.device)
        self.net = S3FDNet('test').to(self.device)
        state_dict = torch.load(PATH_WEIGHT, map_location=self.device)
        self.net.load_state_dict(state_dict)
        self.net.eval()
        print('[S3FD] finished loading (%.4f sec)' % (time.time() - tstamp))
    
    def detect_faces(self, image, conf_th=0.8, scales=[1]):

        w, h = image.shape[1], image.shape[0]

        bboxes = np.empty(shape=(0, 5))

        with torch.no_grad():
            for s in scales:
                scaled_img = cv2.resize(image, dsize=(0, 0), fx=s, fy=s, interpolation=cv2.INTER_LINEAR)
                scaled_img = np.swapaxes(scaled_img, 1, 2)
                scaled_img = np.swapaxes(scaled_img, 1, 0)
                scaled_img = scaled_img[[2, 1, 0], :, :]
                scaled_img = scaled_img.astype('float32')
                scaled_img -= img_mean
                scaled_img = scaled_img[[2, 1, 0], :, :]
                x = torch.from_numpy(scaled_img).unsqueeze(0).to(self.device)
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


class MakeDataset():
    def __init__(self):
        self.split_file = '/mnt/MAFA/test_anno.txt'
        self.image_dir = IMAGE_DIR
        self.data_dict = dict()

        with open(self.split_file, 'r') as fp:
            lines = [line.rstrip('\n') for line in fp]

            i = 0
            while i < len(lines):
                print('%6d / %6d' % (i, len(lines)))
                img_name = lines[i]
                num_face = int(lines[i + 1])

                if num_face != 0:
                    rect_list = list()
                    for j in range(num_face):
                        r = [float(x) for x in lines[i + 2 + j].split()[0:4]]
                        rect = [r[0], r[1], r[0] + r[2], r[1] + r[3]]
                        rect_list.append(rect)
                    self.data_dict[img_name] = rect_list
                    i = i + num_face + 2
                else:
                    i = i + 1 + 2


def IoU(boxA, boxB):
    area_A = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    area_B = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    xx1 = np.maximum(boxA[0], boxB[0])
    yy1 = np.maximum(boxA[1], boxB[1])
    xx2 = np.minimum(boxA[2], boxB[2])
    yy2 = np.minimum(boxA[3], boxB[3])
    w_inter = np.maximum(0, xx2 - xx1 + 1)
    h_inter = np.maximum(0, yy2 - yy1 + 1)
    area_inter = w_inter * h_inter

    return area_inter / (area_A + area_B - area_inter)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--conf', type=float, default=0.95)
    parser.add_argument('--iou', type=float, default=0.5)
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    device = args.device
    conf_threshold = args.conf
    iou_threshold = args.iou

    WD = MakeDataset()

    det = S3FD(device=device)
    scale_list = [0.5, 1]

    N = len(WD.data_dict.keys())
    total_iou = 0.0
    total_recall = 0.0
    total_precision = 0.0
    total_f1score = 0.0
    total_time = 0.0

    for image_index, image_name in enumerate(WD.data_dict.keys(), 1):
        print('%5d / %5d : %s' % (image_index, N, image_name))
        image = cv2.imread(os.path.join(IMAGE_DIR, image_name))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        boxes = WD.data_dict[image_name]

        img_time = time.time()
        pred_boxes = det.detect_faces(image, conf_threshold, scale_list)
        img_time = time.time() - img_time
        
        true_num = len(boxes)
        positive_num = len(pred_boxes)
        img_iou = 0.0
        img_recall = 0.0
        img_precision = 0.0
        img_f1score = 0.0

        pred_dict = dict()

        for box in boxes:
            max_iou = 0
            for i, pred_box in enumerate(pred_boxes):
                if i not in pred_dict.keys():
                    pred_dict[i] = 0
                iou = IoU(box, pred_box)
                if iou > max_iou:
                    max_iou = iou
                if iou > pred_dict[i]:
                    pred_dict[i] = iou
            img_iou += max_iou
        
        if true_num * positive_num > 0:
            true_positive = 0.0
            for i in pred_dict.keys():
                if pred_dict[i] > iou_threshold:
                    true_positive += 1.0
            img_recall = true_positive / true_num
            img_precision = true_positive / positive_num
            if img_recall * img_precision == 0:
                img_f1score = 0.0
            else:
                img_f1score = (2*img_recall*img_precision) / (img_recall+img_precision)
            img_iou = img_iou / true_num
        
            print('- | TP = %02d | TN =    |' % (true_positive))
            print('  | FP = %02d | FN = %02d |' % (positive_num - true_positive, true_num - true_positive))

        total_iou += img_iou
        total_recall += img_recall
        total_precision += img_precision
        total_f1score += img_f1score
        total_time += img_time

        print('- Avg.            IoU =', total_iou / image_index)
        print('- Avg.         Recall =', total_recall / image_index)
        print('- Avg.      Precision =', total_precision / image_index)
        print('- Avg.       F1-score =', total_f1score / image_index)
        print('- Avg. Inference Time =', total_time / image_index)

        torch.cuda.empty_cache() #GPUのメモリ不足を防ぐ


if __name__ == '__main__':
    main()
