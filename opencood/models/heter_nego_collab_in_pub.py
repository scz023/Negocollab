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
from opencood.models.heter_nego_o import HeterNego
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

def add_clone_modality_dict(output_dict, key, modality_dict):
    # 向output_dict中添加modality_dict的拷贝信息, 避免后续操作改变value值对计算loss的影响
    new_dict = {}
    for k, v in modality_dict.items():
        new_dict.update({k: v.clone()})
        
    output_dict.update({key: new_dict})

class HeterNegoCollabInPub(HeterNego):
    def __init__(self, args):  

        self.modality_name_list = list(set(args['mapping_dict'].values()))
            
        for modality_name in self.modality_name_list:
            model_setting = args[modality_name]
            
            # pub or direct_pub
            model_setting['comm_args']['comm_space'] = args['comm_space']
            model_setting['comm_args']['modality'] = modality_name
                
            # name for visable features
            # model_setting['comm_args'].update({'modality_name': modality_name})   
            if 'modality_name' in model_setting['comm_args'].keys():
                self.feature_show = True
            
             
            model_setting['comm_args'].update({'inference': True})  
            
        super().__init__(args)   
        
        self.local2pub_builed = False


    @property
    def query_pub_wo_emb(self):        
        query = self.bev_posemb.transpose(1, 0).view(self.max_codes, self.H_size, self.W_size)
        return query

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
        output_dict = {'pyramid': 'collab'}
        
        
        """Setup public code and find mapping dict from local to pub"""
        if not self.local2pub_builed:
            super().setup_pubcb_local2pub()
            self.local2pub_builed = True
        # output_dict.update({"pub_codebook": self.pub_codebook})
        
        # codebook_dict = {}
        for modality_name in self.modality_name_list:
            eval(f'self.comm_{modality_name}.pub_change')(self.local2pub_dict[modality_name], \
                                                    self.pub2local_dict[modality_name], self.pub_codebook)
            
            # if not self.fixed[modality_name]:
                # codebook_dict[modality_name] = eval(f'self.comm_{modality_name}.codebook')
        
        # output_dict.update({"codebook_dict": codebook_dict, "local2pub_dict":self.local2pub_dict})           
        
        # modality_type_list = data_dict['modality_type_list'] 
        # agent_modality_list = data_dict['agent_modality_list'] 
        agent_modality_list = data_dict['agent_modality_list'] 
        affine_matrix = normalize_pairwise_tfm(data_dict['pairwise_t_matrix'], self.H, self.W, self.fake_voxel_size)
        record_len = data_dict['record_len'] 
        modality_count_dict = Counter(agent_modality_list)
        modality_feature_dict = {}
        
        # 按modality分类推理
        for modality_name in self.modality_name_list:
            if modality_name not in modality_count_dict: # 场景中可能并不存在该模态的cav, 设置判别条件, 避免出错
                continue
            feature = eval(f"self.encoder_{modality_name}")(data_dict, modality_name)
            feature = eval(f"self.backbone_{modality_name}")({"spatial_features": feature})['spatial_features_2d']
            feature = eval(f"self.aligner_{modality_name}")(feature)
            modality_feature_dict[modality_name] = feature
               
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
        
        """cav将本地映射转换为公共权重, 并共享"""
        pub_weight_dict = {}
        # self.bev_posemb = self.bev_posemb.to(feature.device)
        # for modality_name in self.modality_name_list:
        #     if modality_name not in modality_count_dict:
        #         continue
        #     pub_weight_dict[modality_name] = \
        #       eval(f"self.comm_{modality_name}.sender")(modality_feature_dict[modality_name], self.query_pub)
        for modality_name in self.modality_name_list:
            if modality_name not in modality_count_dict:
                continue
            ketfeat_bf_align2 = \
                eval(f"self.comm_{modality_name}.sender")(modality_feature_dict[modality_name])
            pub_weight_dict[modality_name] = \
                eval(f"self.comm_{modality_name}.sender_align")(ketfeat_bf_align2)
                            
        
        """Assemble heter features""" 
        pub_weight_list = self.assemble_heter_features(agent_modality_list, pub_weight_dict)  
        pub_weight = torch.stack(pub_weight_list)      
        
        """output detection resultes in ego perspective"""
        ego_modality = agent_modality_list[0]
        feature = eval(f'self.comm_{ego_modality}.receiver')(pub_weight, record_len, affine_matrix, \
            record_len_modality = data_dict['record_len_modality'][ego_modality])
        
        
        
        # self.bev_posemb.to(feature.device)
        # feature_show(super().query_pub, f'analysis/feature_maps/pub_comp')
        # feature_show(self.query_pub_wo_emb, f'analysis/feature_maps/pub_comp_wo_emb')
        
        """replace feature in ego perspective with features before send""" 
        # Find ego idxs
        idx = 0
        ego_idxs = [idx]
        for batch_n_cavs in record_len[:-1]:
            idx = idx + batch_n_cavs
            ego_idxs.append(idx)
            
        # assemble features classified by modalities, 由于不同模态特征的尺寸可能不同, 使用单独的
        heter_feature_2d_list = self.assemble_heter_features(agent_modality_list, modality_feature_dict)
        
        # replace feature of ego_idxs in feature with feature in ref heter_feature_2d_list
        # 由于不同模态特征的尺寸可能不同, 不一定可以直接对heter_feature_2d_list进行torch.stack操作
        # 因此采用遍历的方法取出大小相同的ego特征后, 再stack出ego特征张量
        feature[ego_idxs] = torch.stack([heter_feature_2d_list[idx] for idx in ego_idxs])
                
        # feature_show(feature[0], f'analysis/direct_unify/init_feature')
        
        """Fuse and output""" 
        fusion_method = eval(f"self.fusion_method_{ego_modality}")
        
        if fusion_method == 'pyramid':
            fused_feature, occ_outputs = eval(f"self.fusion_net_{ego_modality}.forward_collab")(
                                                    feature,
                                                    record_len, 
                                                    affine_matrix, 
                                                    agent_modality_list, 
                                                    self.cam_crop_info,
                                                )
        else:
            fused_feature = eval(f'self.fusion_net_{ego_modality}')(feature, record_len, affine_matrix)
            occ_outputs = None
            
        

        if eval(f"self.shrink_flag_{ego_modality}"):
            fused_feature = eval(f"self.shrink_conv_{ego_modality}")(fused_feature)


        if hasattr(self, 'feature_show'):
            feature_show(fused_feature[0], f'analysis/direct_unify/fused_feature')
            feature_show(fused_feature[1], f'analysis/direct_unify/fused_feature_neb')

        cls_preds = eval(f"self.cls_head_{ego_modality}")(fused_feature)
        reg_preds = eval(f"self.reg_head_{ego_modality}")(fused_feature)
        dir_preds = eval(f"self.dir_head_{ego_modality}")(fused_feature)

        output_dict.update({'cls_preds': cls_preds,
                            'reg_preds': reg_preds,
                            'dir_preds': dir_preds})
        
        output_dict.update({'occ_single_list': 
                            occ_outputs})

        return output_dict