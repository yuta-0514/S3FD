import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiBoxLoss(nn.Module):
    def __init__(self, num_classes=2, neg_pos=3, device='cpu'):
        super(MultiBoxLoss, self).__init__()
        self.num_classes = num_classes
        self.threshold = [0.1, 0.35, 0.5]
        self.negpos_ratio = neg_pos
        self.variance = [0.1, 0.2]
        self.device = device

    def forward(self, predictions, priors,targets):
        loc_data, conf_data, priors = predictions
        num = loc_data.size(0)  # batch size
        priors = priors[:loc_data.size(1), :]
        num_priors = (priors.size(0))
        num_classes = self.num_classes
 
        loc_t = torch.Tensor(num, num_priors, 4).to(self.device) 
        conf_t = torch.LongTensor(num, num_priors).to(self.device)

        for idx in range(num):   
            truths = targets[idx][:, :-1].to(self.device) 
            labels = targets[idx][:, -1].to(self.device)
            defaults = priors.to(self.device)  
            match(self.threshold, truths, defaults, self.variance, labels, loc_t, conf_t, idx)

        pos = conf_t > 0
        # Localization Loss (Smooth L1)
        # Shape: [batch,num_priors,4]

        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
        loc_p = loc_data[pos_idx].view(-1, 4)
        loc_t = loc_t[pos_idx].view(-1, 4)

#        loss_l = F.smooth_l1_loss(loc_p, loc_t, size_average=False)
        loss_l = F.smooth_l1_loss(loc_p, loc_t, reduction='sum')

        # Compute max conf across batch for hard negative mining
        batch_conf = conf_data.view(-1, self.num_classes)
#        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1))
        loss_c = F.cross_entropy(batch_conf, conf_t.view(-1), reduction='none')

        # Hard Negative Mining
        num_pos = pos.long().sum(1, keepdim=True)
        loss_c = loss_c.view(num, -1)
        loss_c[pos] = 0  # filter out pos boxes for now
        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
        num_neg = torch.clamp(self.negpos_ratio*num_pos, max=pos.size(1)-1)
        neg = idx_rank < num_neg.expand_as(idx_rank)

        # Confidence Loss Including Positive and Negative Examples
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)
        conf_p = conf_data[(pos_idx+neg_idx).gt(0)].view(-1, self.num_classes)
        targets_weighted = conf_t[(pos+neg).gt(0)]
#        loss_c = F.cross_entropy(conf_p, targets_weighted, size_average=False)
        loss_c = F.cross_entropy(conf_p, targets_weighted, reduction='sum')

        # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + ¿Lloc(x,l,g)) / N
        N = num_pos.data.sum()
        loss_l /= N
        loss_c /= N

        return loss_l, loss_c



def match(threshold, truths, priors, variances, labels, loc_t, conf_t, idx):
    # jaccard index
    overlaps = jaccard(truths,point_form(priors))

    # (Bipartite Matching)
    # [1,num_objects] best prior for each ground truth
    best_prior_overlap, best_prior_idx = overlaps.max(1, keepdim=True)

    # [1,num_priors] best ground truth for each prior
    best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim=True)
    
    best_truth_idx.squeeze_(0)
    best_truth_overlap.squeeze_(0)
    best_prior_idx.squeeze_(1)  ## [3,1] --> [3]
    best_prior_overlap.squeeze_(1)  ## [3,1] --> [3]
    best_truth_overlap.index_fill_(0, best_prior_idx, 2)  # ensure best prior
    # TODO refactor: index  best_prior_idx with long tensor
    # ensure every gt matches with its prior of max overlap
    for j in range(best_prior_idx.size(0)):
        best_truth_idx[best_prior_idx[j]] = j
    
    # 追加
    _th1, _th2, _th3 = threshold 
    N = (torch.sum(best_prior_overlap >= _th2) +
         torch.sum(best_prior_overlap >= _th3)) // 2
    
    matches = truths[best_truth_idx]          # Shape: [num_priors,4]
    conf = labels[best_truth_idx]      # Shape: [num_priors]  e DB ÌNX
    conf[best_truth_overlap < _th2] = 0 

    best_truth_overlap_clone = best_truth_overlap.clone()
    add_idx = best_truth_overlap_clone.gt(
        _th1).eq(best_truth_overlap_clone.lt(_th2))
    best_truth_overlap_clone[~add_idx] = 0
    stage2_overlap, stage2_idx = best_truth_overlap_clone.sort(descending=True)

    stage2_overlap = stage2_overlap.gt(_th1)

    if N > 0:
        N = torch.sum(stage2_overlap[:N]) if torch.sum(
            stage2_overlap[:N]) < N else N
        conf[stage2_idx[:N]] += 1

    loc = encode(matches, priors, variances)  ## e DB ÉÎµÄItZbgÌlÖ
    loc_t[idx] = loc    # [num_priors,4] encoded offsets to learn
    conf_t[idx] = conf  # [num_priors] top class label for each prior


def jaccard(box_a, box_b):
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, 2]-box_b[:, 0]) *
              (box_b[:, 3]-box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]


def intersect(box_a, box_b):
    A = box_a.size(0)
    B = box_b.size(0)
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, 0] * inter[:, :, 1]


def point_form(boxes):
    return torch.cat((boxes[:, :2] - boxes[:, 2:]/2,     # xmin, ymin
                     boxes[:, :2] + boxes[:, 2:]/2), 1)  # xmax, ymax


def encode(matched, priors, variances):
    # dist b/t match center and prior's center
    g_cxcy = (matched[:, :2] + matched[:, 2:])/2 - priors[:, :2]
    # encode variance
    g_cxcy /= (variances[0] * priors[:, 2:])
    # match wh / prior wh
    g_wh = (matched[:, 2:] - matched[:, :2]) / priors[:, 2:]
    g_wh = torch.log(g_wh) / variances[1]
    # return target for smooth_l1_loss
    return torch.cat([g_cxcy, g_wh], 1)  # [num_priors,4]
