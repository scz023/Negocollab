import torch
import torch.nn as nn
import torch.nn.functional as F
from opencood.loss.point_pillar_loss import sigmoid_focal_loss

class OccLoss(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.pos_cls_weight = args['pos_cls_weight']
        self.cls = args['cls']
    
    def forward(self, output_dict, target_dict_single):
        occ_preds_single = output_dict['occ_preds_single']
        positives = target_dict_single['pos_equal_one']
        negatives = target_dict_single['neg_equal_one']
        batch_size = positives.shape[0]

        occ_positives = torch.logical_or(positives[...,0], positives[...,1]).unsqueeze(-1).float() # N, H, W
        occ_negatives = torch.logical_and(negatives[...,0], negatives[...,1]).unsqueeze(-1).float() # N, H, W

    
        """
        occ_preds_single: N, 1, H, W

        occ_positives: N, H, W, 1
        occ_negatives: N, H, W, 1

        """

        positives_level = F.max_pool2d(occ_positives.permute(0,3,1,2), kernel_size=1).permute(0,2,3,1)
        negatives_level = 1 - F.max_pool2d((1 - occ_negatives).permute(0,3,1,2), kernel_size=1).permute(0,2,3,1)

        occ_labls = positives_level.view(batch_size, -1, 1)
        positives_level = occ_labls
        negatives_level = negatives_level.view(batch_size, -1, 1)

        pos_normalizer = positives_level.sum(1, keepdim=True).float()

        occ_preds = occ_preds_single.permute(0, 2, 3, 1).contiguous() \
                    .view(batch_size, -1,  1)
        occ_weights = positives_level * self.pos_cls_weight + negatives_level * 1.0
        occ_weights /= torch.clamp(pos_normalizer, min=1.0)
        occ_loss = sigmoid_focal_loss(occ_preds, occ_labls, weights=occ_weights, **self.cls)
        occ_loss = occ_loss.sum() / batch_size

        return occ_loss