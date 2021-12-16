import time
import torch
import argparse
import cv2
import numpy as np
import torch.nn as nn
from torch.autograd import Function
from torchvision import models, transforms
from PIL import Image, ImageDraw

from s3fd.nets import S3FDNet


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-gpu', '--use-cuda', action='store_true', default=True,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument('-i', '--image-path', type=str, default='test.jpg',
                        help='Input image path')
    parser.add_argument('-w', '--weight', type=str, default='/S3FD_pytorch/weights/model_best.pth',
                        help='Input image path')
    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print("Using GPU for acceleration")
    else:
        print("Using CPU for computation")
    return args


def letterbox_image(image, size):
    iw, ih = image.size
    w, h = size
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)

    image = image.resize((nw,nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128,128,128))
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    return new_image


def preprocess_image(img):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
    preprocessing = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    return preprocessing(img.copy()).unsqueeze(0)


def decode(loc, priors, variances):
    boxes = torch.cat((
        priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
        priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes


def nms(boxes, scores, overlap=0.5, top_k=200):
    keep = scores.new(scores.size(0)).zero_().long()
    if boxes.numel() == 0:
        return keep
    x1 = boxes[:, 0]  # xmin
    y1 = boxes[:, 1]  # ymin
    x2 = boxes[:, 2]  # xmax
    y2 = boxes[:, 3]  # ymax
    area = torch.mul(x2 - x1, y2 - y1)
    v, idx = scores.sort(0)
    idx = idx[-top_k:]
    xx1 = boxes.new()
    yy1 = boxes.new()
    xx2 = boxes.new()
    yy2 = boxes.new()
    w = boxes.new()
    h = boxes.new()

    count = 0
    while idx.numel() > 0:
        i = idx[-1]
        keep[count] = i
        count += 1
        if idx.size(0) == 1:
            break
        idx = idx[:-1]
        xx1 = torch.index_select(x1, 0, idx)
        yy1 = torch.index_select(y1, 0, idx)
        xx2 = torch.index_select(x2, 0, idx)
        yy2 = torch.index_select(y2, 0, idx)
        xx1 = torch.clamp(xx1, min=x1[i].item())
        yy1 = torch.clamp(yy1, min=y1[i].item())
        xx2 = torch.clamp(xx2, max=x2[i].item())
        yy2 = torch.clamp(yy2, max=y2[i].item())
        w = xx2 - xx1
        h = yy2 - yy1 
        w = torch.clamp(w, min=0.0) 
        h = torch.clamp(h, min=0.0) 
        inter = w*h 
        rem_areas = torch.index_select(area, 0, idx)
        union = (rem_areas - inter) + area[i]
        IoU = inter/union 
        idx = idx[IoU.le(overlap)] 
    return keep, count

class GradCam:
    def __init__(self, model, use_cuda):
        self.model = model
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

    def forward(self, input_img):
        return self.model(input_img)  

    def __call__(self, input_img, num_map):
        if self.cuda:
            input_img = input_img.to('cuda')  # torch.Size([1, 3, 640, 640])
        
        loc_data, conf, prior = model(input_img)
        conf_data = nn.Softmax(dim=-1)(conf)
        prior_data = prior.type('torch.Tensor').to('cuda')
        
        num = loc_data.size(0)
        num_priors = prior_data.size(0)

        conf_preds = conf_data.view(num, num_priors, 2).transpose(2, 1)
        batch_priors = prior_data.view(-1, num_priors, 4).expand(num, num_priors, 4)
        batch_priors = batch_priors.contiguous().view(-1, 4)

        decoded_boxes = decode(loc_data.view(-1, 4), batch_priors, [0.1, 0.2])
        decoded_boxes = decoded_boxes.view(num, num_priors, 4)

        output = torch.zeros(num, 2, 750, 5)

        for i in range(num):
            boxes = decoded_boxes[i].clone()
            conf_scores = conf_preds[i].clone()

            for cl in range(1, 2):
                c_mask = conf_scores[cl].gt(0.05)
                scores = conf_scores[cl][c_mask]
                
                if scores.dim() == 0:
                    continue
                l_mask = c_mask.unsqueeze(1).expand_as(boxes)
                boxes_ = boxes[l_mask].view(-1, 4)
                ids, count = nms(boxes_, scores, 0.3, 5000)
                count = count if count < 750 else 750

                output[i, cl, :count] = torch.cat((scores[ids[:count]].unsqueeze(1), boxes_[ids[:count]]), 1)
        

        scores = output[0,1,:,0]
        # torch.size([750])

        features = self.model.feature_maps4cxq[num_map]
        # features = torch.Size([1, 512, 80, 80])
        features.retain_grad()

        mask = scores > 0.5
        loss = torch.sum(scores[mask])
        # tensor(6.8891, grad_fn=<SumBackward0>)

        loss.backward()

        grads_val = features.grad.cpu().data.numpy()
        # shape:(1, 512, 80, 80)

        target = features
        # torch.Size([1, 512, 80, 80])
        target = target.cpu().data.numpy()[0, :]
        # shape: (512, 80, 80)

        weights = np.mean(grads_val, axis=(2, 3))[0, :]
        # shape: (512,)
        cam = np.zeros(target.shape[1:], dtype=np.float32)
        # shape: (80, 80)

        for i, w in enumerate(weights):
            cam += w * target[i, :, :]
        
        if np.max(cam)==0:
            cam = None
        else:
            # Shape(80, 80)
            # cam = np.maximum(cam, 0)
            cam = cv2.resize(cam, input_img.shape[3:1:-1])
            cam = cam - np.min(cam)
            cam = cam / np.max(cam)
        return cam  # shape: (640, 640) 0~1に正規化

MEANS = np.array([104., 117., 123.])


def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)


if __name__ == '__main__':
    args = get_args()  

    image = Image.open(args.image_path)
    iw, ih = image.size  # 640, 480
    w, h = 640, 640
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)
    box = [(w-nw)//2, (h-nh)//2, nw+(w-nw)//2, nh+(h-nh)//2]


    model = S3FDNet('train').to('cuda')
    model.load_state_dict(torch.load(args.weight))
    model.eval()

    # model.phase = 'Grad-CAM'
    grad_cam = GradCam(model=model, use_cuda=args.use_cuda)


    image = Image.open(args.image_path)
    image = image.convert('RGB')
    image_shape = np.array(np.shape(image)[0:2])
    crop_img = np.array(letterbox_image(image, (640,640)))  # (640, 640, 3)

    input_img = torch.from_numpy(np.expand_dims(np.transpose(crop_img-MEANS,(2,0,1)),0))\
        .type(torch.FloatTensor).requires_grad_(True)    
    #torch.Size([1, 3, 640, 640])

    img = cv2.imread(args.image_path, 1)  # (H, W, BGR)
    img = np.float32(img) / 255
    img = img[:, :, ::-1]
    # (480, 640, 3)
    

    for i in range(6):
        num_map = i
        grayscale_cam = grad_cam(input_img, num_map)
        # shape: (640, 640)
        if grayscale_cam is None:
            continue
        else:
            grayscale_cam_1 = Image.fromarray(np.uint8(grayscale_cam*255)) 
            grayscale_cam_2 = grayscale_cam_1.resize((iw,ih),box=box)
            grayscale_cam_3 = np.array(grayscale_cam_2, np.float32) / 255  
            # (480, 640)
            cam = show_cam_on_image(img, grayscale_cam_3)
            save_path = "cam" + str(i) + ".jpg"
            cv2.imwrite(save_path, cam)  # 0-255  # (480, 640, 3)
