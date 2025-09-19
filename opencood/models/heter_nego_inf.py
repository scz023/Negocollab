""" Author: Yifan Lu <yifan_lu@sjtu.edu.cn>

HEAL: An Extensible Framework for Open Heterogeneous Collaborative Perception 
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from icecream import ic
from collections import OrderedDict, Counter
from opencood.models.heter_pyramid_consis import HeterPyramidConsis
from opencood.models.heter_nego_train import HeterNegoTrain
from opencood.models.sub_modules.base_bev_backbone_resnet import ResNetBEVBackbone 
from opencood.models.comm_modules.comm_in_pub_decom import ComminPub, coords_bev_pos_emdbed
from opencood.models.sub_modules.feature_alignnet import AlignNet
from opencood.models.sub_modules.downsample_conv import DownsampleConv
from opencood.models.sub_modules.naive_compress import NaiveCompressor
from opencood.models.fuse_modules.pyramid_fuse import PyramidFusion
from opencood.utils.transformation_utils import normalize_pairwise_tfm
from opencood.utils.model_utils import check_trainable_module, fix_bn, unfix_bn
from opencood.tools.feature_show import feature_show
import importlib
import torchvision


class HeterNegoInf(HeterNegoTrain):
    def __init__(self, args):  
        
        self.modality_name_list = list(set(args['mapping_dict'].values()))
            
        for modality_name in self.modality_name_list:
            model_setting = args[modality_name]
            
            # pub or direct_pub
            # model_setting['comm_args']['comm_space'] = args['comm_space']
            model_setting['comm_args']['modality'] = modality_name           
             
            model_setting['comm_args'].update({'inference': True})  
            
        super().__init__(args)   


        """ ft训练阶段复用 """
        """ 预设置全部模型参数冻结 """    
        for name, p in self.named_parameters():
            p.requires_grad = False
        for name, module in self.named_modules():
            module.eval()

        """下游任务微调阶段, 仅训练接收器""" 
        for modality_name in self.modality_name_list:
            if not self.allied[modality_name]:
                self.set_mode_receiver_trainable(modality_name)
        

    def assemble_heter_features(self, agent_modality_list, modality_feature_dict):
        modality_count_dict = Counter(agent_modality_list)
        counting_dict = {modality_name:0 for modality_name in self.modality_name_list}
        heter_feature_2d_list = []
        for modality_name in agent_modality_list:
            if modality_name not in modality_count_dict:
                continue
            feat_idx = counting_dict[modality_name]
            heter_feature_2d_list.append(modality_feature_dict[modality_name][feat_idx])
            counting_dict[modality_name] += 1

        # heter_feature_2d = torch.stack(heter_feature_2d_list)
        
        return heter_feature_2d_list

    def forward(self, data_dict):
        output_dict = {'pyramid': 'collab',
                       'modality_name_list': self.modality_name_list, 
                       'newtype_modality_list': self.newtype_modality_list}
        
        
        agent_modality_list = data_dict['agent_modality_list'] 
        affine_matrix = normalize_pairwise_tfm(data_dict['pairwise_t_matrix'], self.H, self.W, self.fake_voxel_size)
        record_len = data_dict['record_len'] 
        modality_count_dict = Counter(agent_modality_list)
        modality_feature_dict = {}

        output_dict.update({'agent_modality_list': agent_modality_list})
        
        # 按modality分类推理
        for modality_name in self.modality_name_list:
            if modality_name not in modality_count_dict: # 场景中可能并不存在该模态的cav, 设置判别条件, 避免出错
                continue
            feature = eval(f"self.encoder_{modality_name}")(data_dict, modality_name)
            feature = eval(f"self.backbone_{modality_name}")({"spatial_features": feature})['spatial_features_2d']
            feature = eval(f"self.aligner_{modality_name}")(feature)
            modality_feature_dict[modality_name] = feature

            # if modality_name == agent_modality_list[0]:
            #     feature_show(feature[0], f'analysis/vis_feats_inf/ego_ori')
            # else:
            #     feature_show(feature[0], f'analysis/vis_feats_inf/{modality_name}_ori')

        """
        Crop/Padd camera feature map.
        """
        for modality_name in self.modality_name_list:
            if modality_name in modality_count_dict:
                if self.sensor_type_dict[modality_name] == "camera":
                    # should be padding. Instead of masking
                    feature = modality_feature_dict[modality_name]
                    _, _, H, W = feature.shape
                    target_H = int(H*eval(f"self.crop_ratio_H_{modality_name}"))
                    target_W = int(W*eval(f"self.crop_ratio_W_{modality_name}"))

                    # 对齐特征尺寸, 使特征表示的范围相同
                    crop_func = torchvision.transforms.CenterCrop((target_H, target_W))
                    modality_feature_dict[modality_name] = crop_func(feature)
                    if eval(f"self.depth_supervision_{modality_name}"):
                        output_dict.update({
                            f"depth_items_{modality_name}": eval(f"self.encoder_{modality_name}").depth_items
                        })


        feature_list = self.assemble_heter_features(agent_modality_list, modality_feature_dict)  
        H, W = feature_list[0].shape[1:]
        
        
        # features = torch.stack(feature_list)        
        ego_modality = agent_modality_list[0]
        fusion_method = eval(f"self.fusion_method_{ego_modality}")      
        
        """cav 将本地特征映射到公共空间"""
        pub_modality_feature_dict = {}
        for modality_name in self.modality_name_list:
            if modality_name in modality_count_dict:
                feature = modality_feature_dict[modality_name]

                # if modality_name == 'm3':
                #     pub_feature = torch.nn.functional.interpolate(feature,
                #                                     (H, W),
                #                                     mode='bilinear',
                #                                     align_corners=True)
                    
                    
                # else:
                #     pub_feature = eval(f"self.comm_{modality_name}.sender")(feature)
                # pub_feature = feature

                pub_feature = eval(f"self.comm_{modality_name}.send")(feature)

                pub_modality_feature_dict[modality_name] = pub_feature       
        
        """Assemble heter pub modality features""" 
        pub_feature_list = self.assemble_heter_features(agent_modality_list, pub_modality_feature_dict)  
        pub_features = torch.stack(pub_feature_list)      
        ego_modality = agent_modality_list[0]
        
        # if ego_modality == 'm3':
        #     received_feature = pub_features
        # else:
        #     """output detection resultes in ego perspective"""
        #     record_len_modality = data_dict['record_len_modality'][ego_modality]
        #     received_feature = eval(f'self.comm_{ego_modality}.receiver')\
        #         (pub_features, record_len, affine_matrix,  record_len_modality) 
                
        # received_feature = pub_features

        """output detection resultes in ego perspective"""
        record_len_modality = data_dict['record_len_modality'][ego_modality]
        received_feature = eval(f'self.comm_{ego_modality}.receive')\
            (pub_features, record_len, affine_matrix,  record_len_modality)        
          
        
        # received_feature.retain_grad()
        
        """replace feature in ego perspective with features before send""" 
        # Find ego idxs
        idx = 0
        ego_idxs = [idx]
        # ego_mode = agent_modality_list[idx]
        for batch_n_cavs in record_len[:-1]:
            idx = idx + batch_n_cavs
            ego_idxs.append(idx)
            # ego_mode = ego_mode + '' + agent_modality_list[idx]
        # print(record_len)
        # print(ego_mode)
        
        # assemble features classified by modalities, 由于不同模态特征的尺寸可能不同, 使用单独的
        heter_feature_2d_list = self.assemble_heter_features(agent_modality_list, modality_feature_dict)
        
        # replace feature of ego_idxs in feature with feature in ref heter_feature_2d_list
        # 由于不同模态特征的尺寸可能不同, 不一定可以直接对heter_feature_2d_list进行torch.stack操作
        # 因此采用遍历的方法取出大小相同的ego特征后, 再stack出ego特征张量, 并将ego替换为初始特征
        received_feature[ego_idxs] = torch.stack([heter_feature_2d_list[idx] for idx in ego_idxs])
                
        
        """Fuse and output""" 
        fusion_method = eval(f"self.fusion_method_{ego_modality}")
        
        if fusion_method == 'pyramid':
            fused_feature, occ_outputs = eval(f"self.fusion_net_{ego_modality}")(
                                                    received_feature,
                                                    record_len, 
                                                    affine_matrix, 
                                                    agent_modality_list, 
                                                    self.cam_crop_info)  
            # fused_feature, occ_outputs = eval(f"self.fusion_net_{ego_modality}.forward_collab")(
            #                                         received_feature,
            #                                         record_len, 
            #                                         affine_matrix, 
            #                                         agent_modality_list, 
            #                                         self.cam_crop_info,
            #                                     )
            # fused_feature = eval(f"self.shrink_conv_{ego_modality}")(fused_feature)
            
        else:
            fused_feature = eval(f'self.fusion_net_{ego_modality}')(received_feature, record_len, affine_matrix)
            occ_outputs = None            

        # feature_show(fused_feature[0], 'analysis/vis_feats_inf/ego_fused')
        # feature_show(fused_feature[0], f'analysis/ngcb_oth/{fusion_method}_fused')

        cls_preds = eval(f"self.cls_head_{ego_modality}")(fused_feature)
        reg_preds = eval(f"self.reg_head_{ego_modality}")(fused_feature)
        dir_preds = eval(f"self.dir_head_{ego_modality}")(fused_feature)

        output_dict.update({'cls_preds': cls_preds,
                            'reg_preds': reg_preds,
                            'dir_preds': dir_preds})
        
        output_dict.update({'occ_single_list': 
                            occ_outputs})
        
        output_dict.update({'fusion_name': fusion_method})


        return output_dict