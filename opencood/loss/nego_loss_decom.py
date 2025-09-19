import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
import os

from opencood.loss.contrastive_loss import ContrastiveLoss
from opencood.loss.point_pillar_loss import PointPillarLoss
from opencood.loss.point_pillar_pyramid_loss import PointPillarPyramidLoss
from opencood.tools.feature_show import feature_show
from opencood.loss.point_pillar_loss import sigmoid_focal_loss


class MSELossPerDim(nn.Module):
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

class InvarianceLossPerDim(nn.Module):
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
        self.std_rate = args['std_rate']
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
    
    def forward(self, x, y, mask_mse = None):
        """
        x: b c h w
        y: b c h w
        mask_mse: b 1 h_ w_
        
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
        
        # return mean_loss
        # std of x and y, compute std in focal areas without score
        std_x = torch.sqrt(x.var(dim=(2, 3)) + 1e-4)
        std_y = torch.sqrt(y.var(dim=(2, 3)) + 1e-4)
        
        # aim_std = torch.ones_like(std_x, device=std_x.device)

        # std_loss = self.smoothLoss(aim_std, std_x) + self.smoothLoss(aim_std, std_y)

        std_loss = torch.sqrt((1 - std_x) ** 2 + 1e-4) / 2 + \
                    torch.sqrt((1 - std_y) ** 2 + 1e-4) / 2

        std_loss = std_loss.sum(1).mean(0)
            
        # scaled_std_loss = torch.tensor(0)
        scaled_std_loss = self.std_rate * std_loss
        
        return mean_loss + scaled_std_loss, std_loss


class NegoLoss(PointPillarPyramidLoss):
    def __init__(self, args):
        super().__init__(args)
        
        # # 默认不使用pub_codebook监督
        # self.pub_cb_supervise = False
        # if 'pub_cb_supervise' in args:
        #     self.pub_cb_supervise = args['pub_cb_supervise']
  
        #     pub_cb_path = 'opencood/logs/pub_codebook/pub_codebook.pth'
        #     if 'pub_cb_path' in args:
        #         pub_cb_path = args['pub_cb_path']
        #     self.codebook_pub = torch.load(pub_cb_path)
        #     # print(torch.norm(self.codebook_pub, dim=1))
            
        # self.unit_loss = True
        # if 'unit_loss' in args:
        #     self.unit_loss = args['unit_loss']
    
            
        # self.cosim_threhold = args['cosim_threhold']
        
        # self.num_unmatched_code = 0
        
        # self.mse_loss = nn.MSELoss()
        # self.mse_loss = MSELossPerDim()
        
        self.stage = args['stage']  

        
        """ 张量均值一致 """
        self.mean_loss = nn.MSELoss(reduce='mean')

        """ 张量完全一致(每个数据点) """
        self.identity_loss = nn.MSELoss(reduce='sum')
        
        """ 二维张量分布一致: 每个维度的均值一致, 标准差均为1 """
        self.invariance_loss = InvarianceLossPerDim(args['invariance'])
        
        # self.contrastive_loss = ContrastiveLoss(args['contrastive'])
        
        self.unify_method = args['unify']['method']
        if self.unify_method == 'contrastive':
            self.unify_loss =  ContrastiveLoss(args['unify']['args'])
        elif self.unify_method == 'invariance':
            self.unify_loss = InvarianceLossPerDim(args['invariance'])
            self.unify_mse_cons = args['unify']['args']['mse_cons']
            
        self.rec_mse_cons = args['rec']['mse_cons']
        self.cycle_mse_cons = args['cycle']['mse_cons']
        self.ra_cycle_mse_cons = args['ra_cycle']['mse_cons']

        self.unify_ratio = args['unify']['args']['ratio']
        self.pre_unify_ratio = args['unify']['args']['pre_ratio']
        self.cycle_ratio = args['cycle']['ratio']
        self.ra_cycle_ratio = args['ra_cycle']['ratio']
        self.keyfeat_cycle_ratio = args['keyfeat_cycle']['ratio']
        
        
        self.loss_dict = {}
    
    def forward(self, output_dict, target_dict, suffix=""):
        """
        output_dict:{
            'modality_name_list': modality names
            'newtype_modality_list': names of allied modalities
            'fixed_mode': Whether fix parameters
            "pub_codebook": public codebook
            'codebook_dict': meta sematics for each modality
            'pub_weight_dict': pub weights per modality
            'avg_pub_weight': average weights of different modality agents 
            'local2pub_dict' local to pub mapping dict for each modality
            
            'fc_before_rbdc': feature before reconstruct by decomposed codebooks,
            'fc_after_rbdc': feature before reconstruct by decomposed codebooks,
            
            'fc_af_recombine_send': feature before send to converter in sender
            'fc_af_convert_recive': feature afterconverter in receiver,
            
            'fc_before_send': feature before send
            'fc_after_receive': feature after send
            
            'keyfeat_bf_align':  key feat before align to pub
            'keyfeat_af_align':  key feat after  align to pub


            'shared_weights_bf_trans': local weights which can ref to pub weights, before translated
            "shared_weights_af_trans": local weights which can ref to pub weights, after translated
            'weights_bf_trans':  All local weights
            'ref_num_modes_per_pubcode': related mode numbers per pubcode
        }
        """
        
        self.modality_name_list = output_dict['modality_name_list']
        self.newtype_modality_list = output_dict['newtype_modality_list']
        fixed_mode = output_dict['fixed_mode']
        pub_codebook = output_dict['pub_codebook']
        codebook_dict = output_dict['codebook_dict']
        local2pub_dict = output_dict['local2pub_dict']
        
        fc_before_rbdc = output_dict['fc_before_rbdc']
        fc_after_rbdc = output_dict['fc_after_rbdc']    
        
        fc_af_recombine_send = output_dict['fc_af_recombine_send']
        fc_bf_recombine_receive = output_dict['fc_bf_recombine_receive']
        
        fc_before_send = output_dict['fc_before_send']
        fc_after_receive = output_dict['fc_after_receive']
        
        # for mode in self.modality_name_list:
        #     weights_bf_trans = output_dict['weights_bf_trans']
        #     feature_show(fc_before_send[mode][0], f'analysis/direct_unify/fc_before_send_{mode}')
        #     feature_show(weights_bf_trans[mode][0], f'analysis/direct_unify/weights_bf_trans_{mode}')
        #     feature_show(fc_after_receive[mode][0], f'analysis/direct_unify/fc_after_receive_{mode}')
            
        # self.mean_loss(fc_before_send['m3'], fc_after_receive['m3'])
        # self.mse_loss(fc_before_send['m3'][0], fc_after_receive['m3'][0])
        # self.mse_loss(weights_bf_trans['m1'][1][0], weights_bf_trans['m3'][1][0])
        # weights_bf_trans['m1'][1][0].shape
        rec_loss = 0
        ra_cycle_loss = 0
        cycle_loss = 0
        
        unit_loss = 0
        orth_loss = 0
        cb_unify_loss = 0
        
        std_rec_loss = 0
        std_ra_cycle_loss = 0
        std_cycle_loss = 0
    
        
        """只计算未在联盟内代理的损失"""
        self.modality_name_list = self.newtype_modality_list
        
        
        """找到每个模态下, 本地码本中每个码元的idx在公共码本中的idx(若公共码本中无对应的码元, 则不加入)"""
        ref_local_idx_dict = {}
        ref_pub_idx_dict = {}
        for modality_name in self.modality_name_list:
            m_cb_len = codebook_dict[modality_name].shape[0]
            ref_local_idx = []
            ref_pub_idx = []
            for idx_local in range(m_cb_len):
                if idx_local in local2pub_dict[modality_name].keys():
                    idx_pub = local2pub_dict[modality_name][idx_local]
                    
                    ref_pub_idx.append(idx_pub)
                    ref_local_idx.append(idx_local)
                    
            ref_local_idx_dict.update({modality_name: ref_local_idx})
            ref_pub_idx_dict.update({modality_name: ref_pub_idx})
          
          
        positives = target_dict['pos_equal_one']
        mask_mse = torch.logical_or(positives[...,0], positives[...,1]).float().unsqueeze(1) # N, 1, H, W
        
        
        # epoch = output_dict['epoch']
        # if epoch <=8:
        #     self.rec_mse_cons = 'all'
        #     self.unify_mse_cons = 'all'
        #     self.cycle_mse_cons = 'all'
        # else:
        #     self.rec_mse_cons = 'focus'
        #     self.unify_mse_cons = 'focus'
        #     self.cycle_mse_cons = 'focus'
            
        mask_mse_dict = {}
        if self.rec_mse_cons == 'all':
            mask_mse_dict.update({'rec_loss': None})
        elif self.rec_mse_cons == 'focus':
            mask_mse_dict.update({'rec_loss': mask_mse})
        
        if self.ra_cycle_mse_cons == 'all':
            mask_mse_dict.update({'ra_cycle_loss': None})
        elif self.ra_cycle_mse_cons == 'focus':
            mask_mse_dict.update({'ra_cycle_loss': mask_mse})
            
        if self.cycle_mse_cons == 'all':
            mask_mse_dict.update({'cycle_loss': None})
        elif self.cycle_mse_cons == 'focus':
            mask_mse_dict.update({'cycle_loss': mask_mse})
        
        if self.unify_method == 'invariance':
            if self.unify_mse_cons == 'all':
                mask_mse_dict.update({'unify_loss': None})
            elif self.unify_mse_cons == 'focus':
                mask_mse_dict.update({'unify_loss': mask_mse})
        
        modality_loss_dict = {}
        for modality_name in self.modality_name_list:
            """Losses about codebook: rec, orth, unit, unify"""
            
            """rec"""
            m_rec_loss, std_rec  = self.invariance_loss(fc_before_rbdc[modality_name], fc_after_rbdc[modality_name], \
                mask_mse = mask_mse_dict['rec_loss'])
            std_rec_loss = std_rec_loss + std_rec


            m_codebook = codebook_dict[modality_name]
            
            """cosim unify"""
            m_cb_unify_loss = 1 - F.cosine_similarity(m_codebook[ref_local_idx], pub_codebook[ref_pub_idx], dim=1)
            m_cb_unify_loss = torch.sum(m_cb_unify_loss)            
            
            """unit"""
            m_cb_len = m_codebook.shape[0]
            m_unit_loss = self.identity_loss(torch.norm(m_codebook, p=2, dim=1),
                                        torch.ones((m_cb_len)).to(m_codebook.device))
            
            """orth"""
            ref_local_idx = ref_local_idx_dict[modality_name]
            ref_pub_idx = ref_pub_idx_dict[modality_name]
            
            m_codebook = F.normalize(m_codebook, p=2, dim=1)
            m_cosim_mat = torch.mm(m_codebook, m_codebook.t())
            identity_matrix = torch.eye(m_cb_len).to(m_codebook.device).detach()
            m_orth_loss = self.identity_loss(m_cosim_mat, identity_matrix)
            
                
            rec_loss = rec_loss + m_rec_loss             
            unit_loss = unit_loss + m_unit_loss 
            orth_loss = orth_loss + m_orth_loss  
            cb_unify_loss = cb_unify_loss + m_cb_unify_loss 
                
            modality_loss = m_rec_loss + m_unit_loss + m_orth_loss + m_cb_unify_loss
                            
                            
            modality_loss_dict[modality_name] = modality_loss



        self.loss_dict.update({
            "rec_loss": rec_loss,
            "unit_loss": unit_loss,
            "orth_loss": orth_loss,
            "cb_unify_loss": cb_unify_loss,      
            "std_rec_loss": std_rec_loss,        
        })
        
        
        
        for modality_name in self.modality_name_list:
            """Loss about cycle consistant: """
            
            m_ra_cycle_loss, std_ra_cycle = self.invariance_loss(fc_af_recombine_send[modality_name], fc_bf_recombine_receive[modality_name], \
                mask_mse = mask_mse_dict['ra_cycle_loss'])
            
            m_cycle_loss, std_cycle = self.invariance_loss(fc_before_send[modality_name], \
                fc_after_receive[modality_name], mask_mse = mask_mse_dict['cycle_loss'])
            
            
            ra_cycle_loss = ra_cycle_loss + m_ra_cycle_loss
            std_ra_cycle_loss = std_ra_cycle_loss + std_ra_cycle
            cycle_loss = cycle_loss + m_cycle_loss
            std_cycle_loss = std_cycle_loss + std_cycle
            
            modality_loss_dict[modality_name] = modality_loss_dict[modality_name] \
                + m_ra_cycle_loss + m_cycle_loss
        
        self.loss_dict.update({
            "ra_cycle_loss": ra_cycle_loss,
            "cycle_loss": cycle_loss,
            "std_cycle_loss": std_cycle_loss,
            "std_ra_cycle_loss": std_ra_cycle_loss,      
        })
        
        
        """ key feat跨模态一致 """
        avg_pub_weight = output_dict['avg_pub_weight']
        w_direct_unify_loss = 0
        std_direct_unify_loss = torch.Tensor([0])[0]            
        
        if self.unify_method == 'contrastive':
            mask_for_unify = target_dict["pos_region_ranges"]
        elif self.unify_method == 'invariance':
            mask_for_unify = mask_mse_dict['unify_loss']
        
        keyfeat_af_align = output_dict['keyfeat_af_align']
        # Compute unify loss to pub weights per modality
        for modality_name in self.modality_name_list:
            ref_local_idx = ref_local_idx_dict[modality_name]
            ref_pub_idx = ref_pub_idx_dict[modality_name]
                
            m_w_direct_unify_loss, std_direct_unify = self.unify_loss(\
                avg_pub_weight[:, ref_pub_idx], \
                keyfeat_af_align[modality_name][:, ref_local_idx], \
                mask_for_unify)
            std_direct_unify_loss = std_direct_unify_loss + std_direct_unify
            
            modality_loss_dict[modality_name] = modality_loss_dict[modality_name] + m_w_direct_unify_loss
            w_direct_unify_loss = w_direct_unify_loss + m_w_direct_unify_loss
        
        self.loss_dict.update({
        "direct_unify_loss": w_direct_unify_loss,
        "std_direct_unify_loss": std_direct_unify_loss
        })   
        
        
        """Align前的Key feat 循环一致, 不启用是因为ra_cycle_loss已经可以约束该项"""
            # indep_loss = torch.Tensor([0])[0]
        # 
        # keyfeat_cycle_loss = torch.Tensor([0])[0]
        # std_keyfeat_cycle_loss = torch.Tensor([0])[0]
        # # weights_bf_trans = output_dict['weights_bf_trans'] # All local weights
        # # keyfeat_bf_align = output_dict['keyfeat_bf_align']
        # keyfeat_bf_align = output_dict['keyfeat_bf_align']
        # keyfeat_bf_align_reverse = output_dict['keyfeat_bf_align_reverse']
        
        # """ align前的key feat循环一致 """
        # for modality_name in self.modality_name_list:
        #     m_keyfeat_cycle_loss, m_std_keyfeat_cycle = self.invariance_loss(\
        #         keyfeat_bf_align[modality_name], keyfeat_bf_align_reverse[modality_name], \
        #         mask_mse = None) 
        #     keyfeat_cycle_loss = keyfeat_cycle_loss + m_keyfeat_cycle_loss
        #     modality_loss_dict[modality_name] = modality_loss_dict[modality_name] + m_keyfeat_cycle_loss
        #     std_keyfeat_cycle_loss = std_keyfeat_cycle_loss + m_std_keyfeat_cycle
        
        # self.loss_dict.update({
        # "keyfeat_cycle_loss": keyfeat_cycle_loss,
        # "std_keyfeat_cycle_loss": std_keyfeat_cycle_loss
        # })   


            # Dim in dim key feat before trans independent
            # for modality_name in self.modality_name_list:
            #     m_cb_len = codebook_dict[modality_name].shape[0]
            #     # b c h w -> b c h*w
            #     m_B = keyfeat_bf_align[modality_name].shape[0]
            #     m_w_bf_trans = keyfeat_bf_align[modality_name].view(m_B, m_cb_len, -1) 
                
            #     # Z-Score 归一化
            #     mean = torch.mean(m_w_bf_trans, dim=2, keepdim=True)
            #     std = torch.std(m_w_bf_trans, dim=2, keepdim=True)
            #     Z_score_norm = transforms.Normalize(mean=mean, std=std)
            #     m_w_bf_trans = Z_score_norm(m_w_bf_trans)
                
            #     cor_dim = torch.bmm(m_w_bf_trans, m_w_bf_trans.transpose(1, 2)) / \
            #                         ((m_w_bf_trans ** 2).sum(dim=2).unsqueeze(-1) + 1e-3)
            #     # (m_w_bf_trans * m_w_bf_trans).sum(dim=2).size()
            #     # torch.bmm(m_w_bf_trans, m_w_bf_trans.transpose(1, 2)).size()
            #     identity_dim = torch.eye(m_cb_len).to(cor_dim.device).unsqueeze(0).expand(m_B, -1, -1)
            #     m_indep_loss = self.identity_loss(cor_dim, identity_dim)
            #     indep_loss = indep_loss + m_indep_loss
            #     modality_loss_dict[modality_name] = modality_loss_dict[modality_name] + m_indep_loss
                            
            # self.loss_dict.update({
            # "indep_loss_bf_align": indep_loss
            # })



        """Pre unify"""
        # Key feat between align1 and align2 unify
        # avg_keyfeat_bf_align2 = output_dict['avg_keyfeat_bf_align2']
        # keyfeat_bf_align2 = output_dict['keyfeat_bf_align2']
        # w_pre_direct_unify_loss = 0
        # std_pre_direct_unify_loss = torch.Tensor([0])[0]            
        
        
        
        # Compute unify loss to pub weights per modality
        # for modality_name in self.modality_name_list:
        #     ref_local_idx = ref_local_idx_dict[modality_name]
        #     ref_pub_idx = ref_pub_idx_dict[modality_name]
                
        #     m_w_pre_direct_unify_loss, std_pre_direct_unify = self.unify_loss(\
        #         avg_keyfeat_bf_align2[:, ref_pub_idx], \
        #         keyfeat_bf_align2[modality_name][:, ref_local_idx], \
        #         mask_for_unify)
        #     std_pre_direct_unify_loss = std_pre_direct_unify_loss + std_pre_direct_unify
            
        #     modality_loss_dict[modality_name] = modality_loss_dict[modality_name] + m_w_pre_direct_unify_loss
        #     w_pre_direct_unify_loss = w_pre_direct_unify_loss + m_w_pre_direct_unify_loss
        
        # self.loss_dict.update({
        # "w_pre_direct_unify_loss": w_pre_direct_unify_loss
        # })   
        # self.loss_dict.update({
        # "std_pre_direct_unify_loss": std_pre_direct_unify_loss
        # })   
                    

        total_loss = 0
        for mode, mode_loss in modality_loss_dict.items():
            total_loss = total_loss + mode_loss
            self.loss_dict.update({f'{mode}_loss': mode_loss})
        
        """Assume losses"""       
        # total_loss = rec_loss + unit_loss + orth_loss + cb_unify_loss + \
        #     keyfeat_cycle_loss * self.keyfeat_cycle_ratio + \
        #     ra_cycle_loss * self.ra_cycle_ratio + \
        #     cycle_loss * self.cycle_ratio + \
        #     w_direct_unify_loss * self.unify_ratio + \
        #     w_pre_direct_unify_loss * self.pre_unify_ratio


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
            print_msg = print_msg + f" || {k}: {v:.4f}"
        
        if pbar is None:
            print(print_msg)   
        else:
            pbar.set_description(print_msg)
    
