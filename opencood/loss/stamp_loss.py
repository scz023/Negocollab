import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
import os

from opencood.loss.contrastive_loss import ContrastiveLoss
from opencood.loss.distill_loss import SemDistillLoss, StruDistillLoss
from opencood.loss.point_pillar_depth_loss import PointPillarDepthLoss
from opencood.loss.point_pillar_loss import PointPillarLoss
from opencood.loss.point_pillar_pyramid_loss import PointPillarPyramidLoss
from opencood.tools.feature_show import feature_show
from opencood.loss.point_pillar_loss import sigmoid_focal_loss


class MSELossCustomize(nn.Module):
    """
    Compute avg mse loss in each dim, and sum across cavs and batch.
    """
    def __init__(self) -> None:
        super().__init__()
        self.mse_per_data = nn.MSELoss(reduction='none')
    
    def forward(self, x, y):
        loss = self.mse_per_data(x, y)
        if len(x.shape) == 4: # B, C, H, W
            loss = loss.mean(dim=(2, 3))
            loss = loss.sum()
        elif len(x.shape) == 3: # C, H, W
            loss = loss.mean(dim=(1, 2))
            loss = loss.sum()
        elif len(x.shape) == 2 or len(x.shape) == 1: # cb_unify: [num_codes, num_codes] or unit: [num_codes]
            loss = loss.sum()        
        return loss

class InvarianceLoss(nn.Module):
    """
    Compute avg mse loss in each dim, and sum across cavs and batch.
    """
    def __init__(self, args) -> None:
        super().__init__()
        self.mse_per_data = nn.MSELoss(reduction='none')

        self.smoothLoss = nn.SmoothL1Loss(reduction='none')
        
        # Gaussian Smooth
        # self.smooth = True
        kernel_size = args['mask_gaussian_smooth']['k_size']
        c_sigma = args['mask_gaussian_smooth']['c_sigma']
        self.std_ratio = args['std_ratio']
        self.min_atten_weight = args['min_atten_weight'] if 'min_atten_weight' in args.keys() else 1
        self.gaussian_filter = nn.Conv2d(1, 1, kernel_size=kernel_size, stride=1, padding=(kernel_size-1)//2)
        self.init_gaussian_filter(kernel_size, c_sigma)
        self.gaussian_filter.requires_grad = False
        
    def init_gaussian_filter(self, k_size=5, sigma=1):
        def _gen_gaussian_kernel(k_size=5, sigma=1):
            center = k_size // 2
            x, y = np.mgrid[0 - center : k_size - center, 0 - center : k_size - center]
            g = 1 / (2 * np.pi * sigma) * np.exp(-(np.square(x) + np.square(y)) / (2 * np.square(sigma)))
            return g
        gaussian_kernel = _gen_gaussian_kernel(k_size, sigma)
        self.gaussian_filter.weight.data = torch.Tensor(gaussian_kernel).to(self.gaussian_filter.weight.device).unsqueeze(0).unsqueeze(0)
        self.gaussian_filter.bias.data.zero_()
    
    def forward(self, x, y, target_dict_single = None, std_standard = None):
        """
        x: b c h w
        y: b c h w
        mask_mse: b 1 h_ w_
        std_standard: 0, 1 若为0, 对齐到x的标准差, 若为1, 对齐到y的标准差
        
        return:
          mean_loss: x  y 均值一致
          std_loss: x y 标准差均为 1
        """      
        
        if target_dict_single is not None:
            """Pay more attenton to objects"""
            _, _, H, W = x.shape
            mask = target_dict_single['pos_equal_one']
            mask = torch.logical_or(mask[...,0], mask[...,1]).unsqueeze(1).float() # N, H, W
            
            # feature_show(mask[0], 'mask_show/mask_init')
             
            mask = F.interpolate(mask, (H, W), mode='bilinear')
            
            # feature_show(mask[0], 'mask_show/mask_interpolate')
            
            mask = self.gaussian_filter(mask)
            
            # feature_show(mask[0], 'mask_show/mask_gaussian_filt')            
            
            mask = torch.where(mask > 0, 1, self.min_atten_weight)
            
            mask = mask.detach()
            # feature_show(mask[0], 'mask_show/mask_final')

            x = x * mask
            y = y * mask

        mean_loss = self.mse_per_data(x, y)
        mean_loss = mean_loss.mean(dim=(2, 3))
        mean_loss = mean_loss.sum(1).mean(0) # B C; 对每个通道的损失求和, 对每个batch的损失计算平均值
        
        # std_loss = torch.Tensor([0])
        # std_loss = std_loss.to(x.device)
        # return mean_loss, std_loss
        

        # std of x and y, compute std in focal areas without score
        std_x = torch.sqrt(x.var(dim=(2, 3)) + 1e-4)
        std_y = torch.sqrt(y.var(dim=(2, 3)) + 1e-4)
        
        # aim_std = torch.ones_like(std_x, device=std_x.device)

        # std_loss = self.smoothLoss(aim_std, std_x) + self.smoothLoss(aim_std, std_y)

        
        if std_standard is None:
            std_loss = torch.sqrt((1 - std_x) ** 2 + 1e-4) / 2 + \
                        torch.sqrt((1 - std_y) ** 2 + 1e-4) / 2
        elif std_standard == 0:
            # std向x对齐
            std_aim = std_x.clone()
            std_aim.detach()
            std_loss = torch.sqrt((std_aim - std_y)**2 + 1e-4)
        elif std_standard == 1:
             # std向y对齐
            std_aim = std_y.clone()
            std_aim.detach()
            std_loss = torch.sqrt((std_aim - std_x)**2 + 1e-4)


        std_loss = std_loss.sum(1).mean(0)
            
        # scaled_std_loss = torch.tensor(0)
        scaled_std_loss = self.std_ratio * std_loss
        
        return mean_loss + scaled_std_loss, std_loss


class StampLoss(nn.Module):
    def __init__(self, args):
        super().__init__()

        # ratio for fs loss
        self.l2p_ratio = args['l2p']['ratio']
        self.l2l_ratio = args['l2l']['ratio']
        self.p2l_ratio = args['p2l']['ratio']
        
        self.fs_loss_func = nn.MSELoss()
        
        self.det_loss_func = PointPillarDepthLoss(args['det'])
        self.pyramid_loss_func = PointPillarPyramidLoss(args['det'])
        
        self.loss_dict = {}
    
    def forward(self, output_dict, target_dict,  target_dict_single, suffix=""):
        """
        output_dict:{
            'local_modality_list': self.modality_name_list
            
            'protocol_feature': protocol_feature,
            'reverted_procotol_feat': reverted_procotol_feat_dict,
    
            "local_feat": deep_clone_dict(local_feature_dict),
            "adapted_local_feat": deep_clone_dict(adapted_local_feat_dict),
            "reverted_local_feat": deep_clone_dict(reverted_local_feat_dict),
    
            'protocol_fusion_name': protocol_fusion_method,        
            'local_fusion_method_dict': local_fusion_method_dict,
            'preds_reverted_protocol_dict': preds_reverted_protocol_dict,
            'preds_reverted_local_dict': preds_reverted_local_dict,
            'preds_adapted_local_dict': preds_adapted_local_dict
            
            
        }
        """
        local_modality_list = output_dict['local_modality_list']
        
        protocol_feature = output_dict['protocol_feature']
        reverted_procotol_feat = output_dict['reverted_procotol_feat']
        
        local_feat = output_dict['local_feat']
        adapted_local_feat = output_dict['adapted_local_feat']
        reverted_local_feat = output_dict['reverted_local_feat']
        
        protocol_fusion_name = output_dict['protocol_fusion_name']
        local_fusion_method = output_dict['local_fusion_method']
        preds_reverted_protocol = output_dict['preds_reverted_protocol']
        preds_reverted_local = output_dict['preds_reverted_local']
        preds_adapted_local = output_dict['preds_adapted_local']
        
        
        
        l2p_fs_loss = 0
        l2l_fs_loss = 0
        p2l_fs_loss = 0
        modality_loss_dict = {}
        for modality_name in local_modality_list:
            m_l2p_fs_loss = self.fs_loss_func(protocol_feature, adapted_local_feat[modality_name])
            m_l2l_fs_loss = self.fs_loss_func(local_feat[modality_name], reverted_local_feat[modality_name])
            m_p2l_fs_loss = self.fs_loss_func(local_feat[modality_name], reverted_procotol_feat[modality_name])

            l2p_fs_loss = l2p_fs_loss + m_l2p_fs_loss
            l2l_fs_loss = l2l_fs_loss + m_l2l_fs_loss
            p2l_fs_loss = p2l_fs_loss + m_p2l_fs_loss            
               
            m_loss = m_l2p_fs_loss * self.l2p_ratio + \
                     m_l2l_fs_loss * self.l2l_ratio + \
                     m_p2l_fs_loss * self.p2l_ratio       
            modality_loss_dict[modality_name] = m_loss
        
        self.loss_dict.update({
            'l2p_fs_loss': l2p_fs_loss,
            'l2l_fs_loss': l2l_fs_loss,
            'p2l_fs_loss': p2l_fs_loss
        })
            
        l2p_det_loss = 0
        l2l_det_loss = 0
        p2l_det_loss = 0
        for modality_name in local_modality_list:
            
            # det loss of adapted local feat perform protocol task
            if protocol_fusion_name == 'pyramid':
                m_l2p_det_loss = self.pyramid_loss_func(preds_adapted_local[modality_name], target_dict['m0']) \
                            + self.pyramid_loss_func(preds_adapted_local[modality_name], target_dict_single['m0'], '_single')
            else:
                m_l2p_det_loss = self.det_loss_func(preds_adapted_local[modality_name], target_dict['m0'])
            
            fusion_name = local_fusion_method[modality_name]
            # det loss of reverted local feat perform local task
            if fusion_name == 'pyramid':
                m_l2l_det_loss = self.pyramid_loss_func(preds_reverted_local[modality_name], target_dict[modality_name]) \
                            + self.pyramid_loss_func(preds_reverted_local[modality_name], target_dict_single[modality_name], '_single')
            else:
                m_l2l_det_loss = self.det_loss_func(preds_reverted_local[modality_name], target_dict[modality_name])    
                
             # det loss of reverted protocol feat perform local task
            if fusion_name == 'pyramid':
                m_p2l_det_loss = self.pyramid_loss_func(preds_reverted_protocol[modality_name], target_dict[modality_name]) \
                            + self.pyramid_loss_func(preds_reverted_protocol[modality_name], target_dict_single[modality_name], '_single')
            else:
                m_p2l_det_loss = self.det_loss_func(preds_reverted_protocol[modality_name], target_dict[modality_name])   
            
            l2p_det_loss = l2p_det_loss + m_l2p_det_loss
            l2l_det_loss = l2l_det_loss + m_l2l_det_loss
            p2l_det_loss = p2l_det_loss + m_p2l_det_loss
            
            m_loss = m_l2p_det_loss  + m_l2l_det_loss + m_p2l_det_loss
            modality_loss_dict[modality_name] = modality_loss_dict[modality_name] + m_loss
            
        self.loss_dict.update({
            'l2p_det_loss': l2p_det_loss,
            'l2l_det_loss': l2l_det_loss,
            'p2l_det_loss': p2l_det_loss
        })
        
        
        total_loss = 0
        for mode, mode_loss in modality_loss_dict.items():
            total_loss = total_loss + mode_loss    
            self.loss_dict.update({f'{mode}_loss': mode_loss})  
        
        self.modality_loss_dict =  modality_loss_dict          

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
    
