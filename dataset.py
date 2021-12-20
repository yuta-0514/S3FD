import torch
from PIL import Image, ImageDraw
import torch.utils.data as data
import numpy as np
import random
from s3fd.augmentations import preprocess

# マスクなしのとき
def make_dataset():
    train_dataset = WIDERDetection('/S3FD_pytorch/data/train.txt', mode='train')
    val_dataset = WIDERDetection('/S3FD_pytorch/data/val.txt', mode='val')
    return train_dataset, val_dataset


# マスクありのとき(WIDER FACE)
def make_masked_dataset():
    train_dataset = WIDERDetection('/S3FD_pytorch/masked_data/train.txt', mode='train')
    val_dataset = WIDERDetection('/S3FD_pytorch/masked_data/val.txt', mode='val')
    return train_dataset, val_dataset


class WIDERDetection(data.Dataset):
    def __init__(self, list_file, mode='train'):
        super(WIDERDetection, self).__init__()
        self.mode = mode
        self.fnames = []
        self.boxes = []
        self.labels = []

        with open(list_file) as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip().split()
            num_faces = int(line[1])
            box = []
            label = []
            for i in range(num_faces):
                x = float(line[2 + 5 * i])
                y = float(line[3 + 5 * i])
                w = float(line[4 + 5 * i])
                h = float(line[5 + 5 * i])
                c = int(line[6 + 5 * i])
                if w <= 0 or h <= 0:
                    continue
                box.append([x, y, x + w, y + h])
                label.append(c)
            if len(box) > 0:
                self.fnames.append(line[0])
                self.boxes.append(box)
                self.labels.append(label)

        self.num_samples = len(self.boxes)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        img, target, h, w = self.pull_item(index)
        return img, target

    def pull_item(self, index):
        while True:
            image_path = self.fnames[index]
            img = Image.open(image_path)
            if img.mode == 'L':
                img = img.convert('RGB')

            im_width, im_height = img.size
            boxes = self.annotransform(
                np.array(self.boxes[index]), im_width, im_height)
            label = np.array(self.labels[index])
            bbox_labels = np.hstack((label[:, np.newaxis], boxes)).tolist()
            img, sample_labels = preprocess(
                img, bbox_labels, self.mode, image_path)
            sample_labels = np.array(sample_labels)
            if len(sample_labels) > 0:
                target = np.hstack(
                    (sample_labels[:, 1:], sample_labels[:, 0][:, np.newaxis]))

                assert (target[:, 2] > target[:, 0]).any()
                assert (target[:, 3] > target[:, 1]).any()
                break 
            else:
                index = random.randrange(0, self.num_samples)

        return torch.from_numpy(img), target, im_height, im_width
        
    def annotransform(self, boxes, im_width, im_height):
        boxes[:, 0] /= im_width
        boxes[:, 1] /= im_height
        boxes[:, 2] /= im_width
        boxes[:, 3] /= im_height
        return boxes


def detection_collate(batch):
    targets = []
    imgs = []
    for sample in batch:
        imgs.append(sample[0])
        targets.append(torch.FloatTensor(sample[1]))
    return torch.stack(imgs, 0), targets
