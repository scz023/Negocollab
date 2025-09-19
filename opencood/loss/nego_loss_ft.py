import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
import os

from opencood.loss.contrastive_loss import ContrastiveLoss
from opencood.loss.distill_loss import SemDistillLoss, StruDistillLoss
from opencood.loss.occ_loss import OccLoss
from opencood.loss.point_pillar_depth_loss import PointPillarDepthLoss
from opencood.loss.point_pillar_loss import PointPillarLoss
from opencood.loss.point_pillar_pyramid_loss import PointPillarPyramidLoss
from opencood.tools.feature_show import feature_show
from opencood.loss.point_pillar_loss import sigmoid_focal_loss
    

class NegoLossFt(nn.Module):
    def __init__(self, args):
        super().__init__()     
        
        self.collab_task_ratio = args['collab_task']['ratio']
        self.det_loss_func = PointPillarDepthLoss(args['det'])
        self.pyramid_loss_func = PointPillarPyramidLoss(args['det'])        
        
        self.loss_dict = {}
    
    def forward(self, output_dict, target_dict,  target_dict_single, suffix=""):
        """
        计算ego 的ft损失
        output_dict:{
            'modality_name_list': modality names
            'newtype_modality_list': names of new modalities
            'fusion_name': Ego Agent的融合方法
        }
        """

        newtype_modality_list = output_dict['newtype_modality_list']    
        agent_modality_list = output_dict['agent_modality_list']
        fusion_name = output_dict['fusion_name']   

        if fusion_name == 'pyramid':
            det_loss = \
                self.pyramid_loss_func(output_dict, target_dict) \
                + self.pyramid_loss_func(output_dict, target_dict_single, '_single')
        else:
            det_loss = self.det_loss_func(output_dict, target_dict)
        
        self.loss_dict.update({"det_loss":det_loss})

        ego_modality = agent_modality_list[0]
        for modality_name in newtype_modality_list: 
            if modality_name == ego_modality:
                self.loss_dict.update({f'{modality_name}_loss': det_loss})
            else:
                self.loss_dict.update({f'{modality_name}_loss': torch.tensor(0.0)}) 
   
        total_loss = det_loss  
        self.total_loss = total_loss
        return total_loss
        
    def logging(self, epoch, batch_id, batch_len, writer = None, pbar=None):
        
        writer.add_scalar('Total_loss', self.total_loss.item(),
                            epoch*batch_len + batch_id)
        
        for k, v in self.loss_dict.items():
            writer.add_scalar(k, v.item(), epoch*batch_len + batch_id)
            # print(k, v.item())
           
        print_msg ="[epoch %d][%d/%d], || Loss: %.4f" % (epoch, batch_id + 1, batch_len, self.total_loss.item())
        for k, v in self.loss_dict.items():
            k = k.replace("_loss", "").capitalize()
            print_msg = print_msg + f" || {k}: {v.item():.4f}"
        
        if pbar is None:
            print(print_msg)   
        else:
            pbar.set_description(print_msg)
    
