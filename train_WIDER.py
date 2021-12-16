import os
import time
import shutil
import argparse
from typing import Sized
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torch.utils.data as data
import numpy as np
from torch.autograd import Variable
import torch.distributed as dist
import torch.backends.cudnn as cudnn

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(log_dir="./logs")

from dataset import make_dataset, detection_collate
from dataset import make_masked_dataset #Masked face
from dataset import MAFA_dataset
from s3fd.nets import S3FDNet,weights_init
from s3fd.multiboxloss import MultiBoxLoss
from s3fd.box_utils import PriorBox


def to_python_float(t):
    if hasattr(t, 'item'):
        return t.item()
    else:
        return t[0]


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    rt /= args.world_size
    return rt


def adjust_learning_rate(optimizer, epoch, step, len_epoch):
    """LR schedule that should yield 76% converged accuracy with batch size 256"""
    factor = epoch // 10

    if epoch >= 30:
        factor = factor + 1

    lr = 1e-3 * (0.1 ** factor)

    """Warmup"""
    if epoch < 1:
        lr = lr * float(1 + step + epoch * len_epoch) / (5. * len_epoch)

    if(step % 10 == 0 and step > 1):
        print("Epoch = {}, step = {}, lr = {}".format(epoch, step, lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def save_checkpoint(state, is_best, epoch):
    filename = os.path.join('/mnt/weights/', "S3FD_" + str(epoch)+ ".pth")
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join('/mnt/weights/', 'model_best.pth'))


def main():
    def train(train_loader, model, priors, criterion, optimizer, epoch):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        loc_loss = AverageMeter()
        cls_loss = AverageMeter()

        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        # switch to train mode
        model.train()
        end = time.time()

        for i, data in enumerate(train_loader, 1):
            input, targets = data
            train_loader_len = len(train_loader)

            adjust_learning_rate(optimizer, epoch, i, train_loader_len)

            # measure data loading time
            data_time.update(time.time() - end)

            input_var = Variable(input.to(device))
            target_var = [Variable(ann.to(device), requires_grad=False) for ann in targets]

            # compute output
            output = model(input_var)
            loss_l, loss_c = criterion(output, priors,target_var)
            loss = loss_l + loss_c

            reduced_loss = loss.data
            reduced_loss_l = loss_l.data
            reduced_loss_c = loss_c.data
            losses.update(to_python_float(reduced_loss), input.size(0))
            loc_loss.update(to_python_float(reduced_loss_l), input.size(0))
            cls_loss.update(to_python_float(reduced_loss_c), input.size(0))

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            torch.cuda.synchronize()
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 10 == 0 and i >= 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Speed {3:.3f} ({4:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'loc_loss {loc_loss.val:.3f} ({loc_loss.avg:.3f})\t'
                    'cls_loss {cls_loss.val:.3f} ({cls_loss.avg:.3f})'.format(
                    epoch, i, train_loader_len,
                    total_batch_size / batch_time.val,
                    total_batch_size / batch_time.avg,
                    batch_time=batch_time,
                    data_time=data_time, loss=losses, loc_loss=loc_loss, cls_loss=cls_loss))
        return losses.avg


    def val(val_loader, model, priors, criterion):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        loc_loss = AverageMeter()
        cls_loss = AverageMeter()

        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        # switch to train mode
        model.eval()
        end = time.time()

        for i, data in enumerate(val_loader, 1):
            input, targets = data
            val_loader_len = len(val_loader)

            # measure data loading time
            data_time.update(time.time() - end)

            input_var = Variable(input.to(device))
            target_var = [Variable(ann.to(device), requires_grad=False) for ann in targets]

            # compute output
            output = model(input_var)
            loss_l, loss_c = criterion(output, priors, target_var)
            loss = loss_l + loss_c

            reduced_loss = loss.data
            reduced_loss_l = loss_l.data
            reduced_loss_c = loss_c.data
            losses.update(to_python_float(reduced_loss), input.size(0))
            loc_loss.update(to_python_float(reduced_loss_l), input.size(0))
            cls_loss.update(to_python_float(reduced_loss_c), input.size(0))

            torch.cuda.synchronize()
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 10 == 0 and i >= 0:
                print('[{0}/{1}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Speed {2:.3f} ({3:.3f})\t'
                    'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'loc_loss {loc_loss.val:.3f} ({loc_loss.avg:.3f})\t'
                    'cls_loss {cls_loss.val:.3f} ({cls_loss.avg:.3f})'.format(
                    i, val_loader_len,
                    total_batch_size / batch_time.val,
                    total_batch_size / batch_time.avg,
                    batch_time=batch_time,
                    data_time=data_time, loss=losses, loc_loss=loc_loss, cls_loss=cls_loss))
        return losses.avg
    
    
    parser = argparse.ArgumentParser(
        description='S3FD face Detector Training With Pytorch')
    # parser.add_argument('--dataset',default='face',help='Train target')
    parser.add_argument('--epochs', default=90, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--batch_size',default=4, type=int,
                        help='Batch size for training')
    parser.add_argument('--distributed', default=True, type=str,
                        help='use distribute training')
    parser.add_argument('--masked', default=False, 
                        help="use-masked data", action="store_true")

    args = parser.parse_args()


    cudnn.benchmark = True

    if not os.path.exists('/mnt/weights/'):
        os.mkdir('/mnt/weights/')


    args = parser.parse_args()
    minmum_loss = np.inf
    args.gpu = 0
    world_size = 1
    total_batch_size = world_size * args.batch_size
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    print('Loading wider dataset...')

    if args.masked==True:
        # train_dataset, val_dataset = MAFA_dataset()
        train_dataset, val_dataset = make_masked_dataset()
    else:
        train_dataset, val_dataset = make_masked_dataset()

    train_loader = data.DataLoader(train_dataset, args.batch_size,
                                num_workers=4,
                                shuffle=True,
                                collate_fn=detection_collate,
                                pin_memory=True)

    val_batchsize = args.batch_size // 2
    val_loader = data.DataLoader(val_dataset, val_batchsize,
                                num_workers=4,
                                shuffle=False,
                                collate_fn=detection_collate,
                                pin_memory=True)

    # build S3FD 
    print("Building net...")
    s3fd_net = S3FDNet('train')
    model = s3fd_net

    # 学習済みのvggを取得
    vgg_weights = torch.load('/mnt/weights/' + 'vgg16_reducedfc.pth')
    print('Load base network....')
    model.vgg.load_state_dict(vgg_weights)

    model = model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)
    criterion = MultiBoxLoss(device=device)

    print('Initializing weights...')
    model.extras.apply(weights_init)
    model.loc.apply(weights_init)
    model.conf.apply(weights_init)

    # load PriorBox
    priorbox = PriorBox(input_size=[640,640],feature_maps=[[160, 160], [80, 80], [40, 40], [20, 20], [10, 10], [5, 5]]
    )
    with torch.no_grad():
        priors = priorbox.forward()
        priors = priors.to(device)

    # train/valの実施
    for epoch in range(args.epochs):
        end = time.time()
        train_loss = train(train_loader, model, priors, criterion, optimizer, epoch)
        val_loss = val(val_loader, model, priors, criterion)
        writer.add_scalar("train_loss",train_loss,epoch)
        writer.add_scalar("val_loss",val_loss,epoch)

        is_best = val_loss < minmum_loss
        minmum_loss = min(val_loss, minmum_loss)
        save_checkpoint(model.state_dict(), is_best, epoch)
        epoch_time = time.time() -end
        print('Epoch %s time cost %f' %(epoch, epoch_time))
    writer.close()


if __name__ == '__main__':
    main()