import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
import os

from opencood.loss.contrastive_loss import ContrastiveLoss
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
    
    def forward(self, x, y, mask_mse = None, std_standard = None):
        """
        x: b c h w
        y: b c h w
        mask_mse: b 1 h_ w_
        std_standard: 0, 1 若为0, 对齐到x的标准差, 若为1, 对齐到y的标准差
        
        return:
          mean_loss: x  y 均值一致
          std_loss: x y 标准差均为 1
        """
        
        # mask_xy = torch.where(smooth_mask_mse > 0, 1, 0)        
        # x = x * mask_xy
        # y = y * mask_xy
        
        mean_loss = self.mse_per_data(x, y)
        
        if mask_mse is not None:
            """Only Focus on labels and nearby feats"""
            _, _, H, W = x.shape
            mask_mse = F.interpolate(mask_mse, (H, W), mode='nearest')
            smooth_mask_mse = self.gaussian_filter(mask_mse).detach()
            mask_mse = torch.where(mask_mse==1, mask_mse, smooth_mask_mse)
            mean_loss = mean_loss * mask_mse  # mean with score per location 
        
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


class NegoLoss(nn.Module):
    def __init__(self, args):
        super().__init__()
        
        """ 张量均值一致 """
        # self.mean_loss = nn.MSELoss(reduce='mean')

        """ 张量完全一致(每个数据点) """
        # self.identity_loss = nn.MSELoss(reduce='sum')
        
        """ 二维张量分布一致: 每个维度的均值一致, 标准差均为1 """
        self.cycle_loss_func = InvarianceLoss(args['invariance'])
        
        # self.contrastive_loss = ContrastiveLoss(args['contrastive'])
        
        self.unify_method = args['unify']['method']
        if self.unify_method == 'contrastive':
            self.unify_loss_func =  ContrastiveLoss(args['unify']['args'])
        elif self.unify_method == 'invariance':
            self.unify_loss_func = InvarianceLoss(args['invariance'])
            self.unify_mse_cons = args['unify']['args']['mse_cons']
            

        self.unify_ratio = args['unify']['args']['ratio']
        self.cycle_ratio = args['cycle']['ratio']
        self.det_ratio = args['det']['ratio']
        self.nego_sem_ratio = args['nego_sem']['ratio']
        self.cycle_sem_ratio = args['cycle_sem']['ratio']
        # self.re_cycle_ratio = args['re_cycle']['ratio']
        
        self.det_loss_func = PointPillarDepthLoss(args['det'])
        self.pyramid_loss_func = PointPillarPyramidLoss(args['det'])
        
        self.loss_dict = {}
    
    def forward(self, output_dict, target_dict,  target_dict_single, suffix=""):
        """
        output_dict:{
            'modality_name_list': modality names
            'newtype_modality_list': names of allied modalities

            'common_rep': 公共表征

            'pub_modality_feature_dict': feature mapped to pub of all modalities
            
            'fc_before_send': feature before send
            'fc_after_receive': feature after send
            
            
            "fc_af_recombine_send": feature between recombiner and enhancer in sender
            "fc_af_recombine_receive": feature between recombiner and enhancer in receive
        }
        """
        stage = output_dict['stage']
        self.modality_name_list = output_dict['modality_name_list']
        self.newtype_modality_list = output_dict['newtype_modality_list']
        
        common_rep = output_dict['common_rep'].detach() # 确保pub_feature只被cycle_loss影响
        
        pub_modality_feature_dict = output_dict['pub_modality_feature_dict']

        
        
        
        # fc_af_recombine_send = output_dict['fc_af_recombine_send']
        # fc_bf_recombine_receive = output_dict['fc_bf_recombine_receive']
        
        if stage == 'nego' or stage == 'add':
            fc_before_send = output_dict['fc_before_send']
            fc_after_receive = output_dict['fc_after_receive']
    
        modality_loss_dict = {}
    

        if stage == 'nego' or stage == 'add':
            cycle_loss = 0
            std_cycle_loss = 0
            for modality_name in self.newtype_modality_list:
                m_cycle_loss, m_std_cycle = self.cycle_loss_func(fc_before_send[modality_name], \
                                                    fc_after_receive[modality_name], std_standard=0)                            
                std_cycle_loss = std_cycle_loss + m_std_cycle
                cycle_loss = cycle_loss + m_cycle_loss
                
                m_cycle_loss = m_cycle_loss * self.cycle_ratio
                modality_loss_dict[modality_name] =  m_cycle_loss
            self.loss_dict.update({"cycle_loss": cycle_loss,
                                   "std_cycle_loss": std_cycle_loss,
                                })  
                

            unify_loss = 0
            std_unify_loss = 0
            for modality_name in self.newtype_modality_list:            
                m_unify_loss, m_std_unify = self.unify_loss_func(common_rep, \
                                                    pub_modality_feature_dict[modality_name], std_standard=0)
                
                std_unify_loss = std_unify_loss + m_std_unify
                unify_loss = unify_loss + m_unify_loss
                
                m_unify_loss = m_unify_loss * self.unify_ratio
                modality_loss_dict[modality_name] = m_unify_loss + modality_loss_dict[modality_name]
            
            self.loss_dict.update({"unify_loss": unify_loss,
                                   "std_unify_loss": std_unify_loss
                                })
            
            
            modality_preds = output_dict['modality_preds']
            modality_fusion_name = output_dict['modality_fusion_name']
            det_loss = 0
            for modality_name in self.newtype_modality_list:
                if modality_name in modality_preds.keys():
                    fusion_name = modality_fusion_name[modality_name]
                    preds = modality_preds[modality_name]
                    
                    if fusion_name == 'pyramid':
                        mode_det_loss = \
                            self.pyramid_loss_func(preds, target_dict) \
                            + self.pyramid_loss_func(preds, target_dict_single, '_single')
                    else:
                        mode_det_loss = self.det_loss_func(preds, target_dict)
                        
                    det_loss = det_loss + mode_det_loss
                    
                    mode_det_loss = mode_det_loss * self.det_ratio
                    modality_loss_dict[modality_name] = mode_det_loss + \
                        modality_loss_dict[modality_name]
                
            self.loss_dict.update({"det_loss":det_loss
                                })
            
            
            
            # cycle_sem_loss = 0
            # modality_preds_single = output_dict['modality_preds_single']
            # for modality_name in self.newtype_modality_list:
            #     if modality_name in modality_preds.keys():
            #         mode_cycle_sem_loss = self.det_loss_func(modality_preds_single[modality_name], \
            #             target_dict_single, '_single')
                    
            #         cycle_sem_loss = cycle_sem_loss + mode_cycle_sem_loss 
                    
            #         mode_cycle_sem_loss = mode_cycle_sem_loss * self.cycle_sem_ratio
            #         modality_loss_dict[modality_name] = mode_cycle_sem_loss + \
            #             modality_loss_dict[modality_name]
            
            # self.loss_dict.update({"cycle_sem_loss": cycle_sem_loss
            #         })
                    
            
        elif stage == 'align':
            unify_loss = 0
            std_unify_loss = 0
            for modality_name in self.newtype_modality_list:            
                m_unify_loss, m_std_unify = self.unify_loss_func(common_rep, \
                                                    pub_modality_feature_dict[modality_name], std_standard=0)
                modality_loss_dict[modality_name] = m_unify_loss
                
                std_unify_loss = std_unify_loss + m_std_unify
                unify_loss = unify_loss + m_unify_loss
            
            self.loss_dict.update({"unify_loss": unify_loss,
                                    "std_unify_loss": std_unify_loss
                                })

        """cycle loss of feature between recombiner and aligner"""
        # re_cycle_loss = 0
        # std_re_cycle_loss = 0
        # for modality_name in self.newtype_modality_list:
        #     m_re_cycle_loss, m_std_re_cycle_loss = self.invariance_loss\
        #         (fc_af_recombine_send[modality_name], fc_bf_recombine_receive[modality_name])
            
        #     re_cycle_loss = re_cycle_loss + m_re_cycle_loss
        #     std_re_cycle_loss = std_re_cycle_loss + m_std_re_cycle_loss
        #     modality_loss_dict[modality_name] = modality_loss_dict[modality_name] + m_re_cycle_loss
            
        # self.loss_dict.update({"re_cycle_loss": re_cycle_loss,
        #                        "std_re_cycle_loss": std_re_cycle_loss
        #                       })
        
        "按模态汇总损失"
        total_loss = 0
        for mode, mode_loss in modality_loss_dict.items():
            total_loss = total_loss + mode_loss    
            self.loss_dict.update({f'{mode}_loss': mode_loss})  
            
            
        # if stage == 'nego':
        #     preds_nego = output_dict['preds_nego']
        #     nego_sem_loss = self.det_loss_func(preds_nego, target_dict_single, '_single')
            
        #     self.loss_dict.update({"nego_sem_loss": nego_sem_loss
        #                         })
            
        #     total_loss = total_loss + nego_sem_loss * self.nego_sem_ratio
        
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
    
