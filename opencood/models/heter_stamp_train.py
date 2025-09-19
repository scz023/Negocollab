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
from opencood.models.comm_modules.comm_for_stamp import CommForStamp
from opencood.models.fuse_modules import build_fusion_net
from opencood.models.sub_modules.base_bev_backbone_resnet import ResNetBEVBackbone 
from opencood.models.comm_modules.comm_in_pub_vanilla import ComminPubVanilla
from opencood.models.sub_modules.feature_alignnet import AlignNet
from opencood.models.sub_modules.downsample_conv import DownsampleConv
from opencood.models.sub_modules.resize_net import ResizeNet
from opencood.models.sub_modules.keyfeat_modules import Negotiator
from opencood.models.sub_modules.naive_compress import NaiveCompressor
from opencood.models.fuse_modules.pyramid_fuse import PyramidFusion
from opencood.utils.transformation_utils import normalize_pairwise_tfm
from opencood.utils.model_utils import check_trainable_module, fix_bn, unfix_bn
from opencood.tools.feature_show import feature_show
import importlib
import torchvision

def deep_clone_dict(old_dict):
    # 向output_dict中添加modality_dict的拷贝信息, 避免后续操作改变value值对计算loss的影响
    new_dict = {}
    for k, v in old_dict.items():
        new_dict.update({k: v.clone()})
        
    return new_dict

class HeterStampTrain(nn.Module):
    def __init__(self, args):
        super(HeterStampTrain, self).__init__()

        self.inference = True if ('inference' in args.keys() and args['inference']) else False
       
        self.modality_name_list = list(set(args['modality_setting'].keys()))
        
        protocol_exists = False
        if 'm0' in self.modality_name_list: 
            protocol_exists = True
            self.modality_name_list.remove('m0')

        self.cav_range = args['lidar_range']
        self.sensor_type_dict = OrderedDict()
        self.cam_crop_info = {} 
        
        self.stage = args['stage'] if 'stage' in args.keys() else 'inf'

        if protocol_exists:
            # setup protocol model
            protocol_model_setting = args['m0']
            sensor_name = protocol_model_setting['sensor_type']
            self.sensor_type_dict['m0'] = sensor_name
            fusion_name = protocol_model_setting['fusion_net']["method"]

            # import model
            encoder_filename = "opencood.models.heter_encoders"
            encoder_lib = importlib.import_module(encoder_filename)
            encoder_class = None
            target_model_name = protocol_model_setting['core_method'].replace('_', '')

            for name, cls in encoder_lib.__dict__.items():
                if name.lower() == target_model_name.lower():
                    encoder_class = cls

            """
            Encoder building
            """
            setattr(self, f"encoder_m0", encoder_class(protocol_model_setting['encoder_args']))
            if protocol_model_setting['encoder_args'].get("depth_supervision", False):
                setattr(self, f"depth_supervision_m0", True)
            else:
                setattr(self, f"depth_supervision_m0", False)

            """
            Backbone building 
            """
            setattr(self, f"backbone_m0", ResNetBEVBackbone(protocol_model_setting['backbone_args']))

            """
            Aligner building
            """
            setattr(self, f"aligner_m0", AlignNet(protocol_model_setting['aligner_args']))
            if sensor_name == "camera":
                camera_mask_args = protocol_model_setting['camera_mask_args']
                setattr(self, f"crop_ratio_W_m0", (self.cav_range[3]) / (camera_mask_args['grid_conf']['xbound'][1]))
                setattr(self, f"crop_ratio_H_m0", (self.cav_range[4]) / (camera_mask_args['grid_conf']['ybound'][1]))
                setattr(self, f"xdist_m0", (camera_mask_args['grid_conf']['xbound'][1] - camera_mask_args['grid_conf']['xbound'][0]))
                setattr(self, f"ydist_m0", (camera_mask_args['grid_conf']['ybound'][1] - camera_mask_args['grid_conf']['ybound'][0]))
                self.cam_crop_info[modality_name] = {
                    f"crop_ratio_W_m0": eval(f"self.crop_ratio_W_m0"),
                    f"crop_ratio_H_m0": eval(f"self.crop_ratio_H_m0"),
                }
            
            
            """
            Fusion, by default multiscale fusion: 
            Note the input of PyramidFusion has downsampled 2x. (SECOND required)
            """
            setattr(self, f"fusion_net_m0", build_fusion_net(protocol_model_setting['fusion_net']))
            setattr(self, f"fusion_method_m0", fusion_name)


            """
            Heads
            """
            setattr(self, f"cls_head_m0", nn.Conv2d(protocol_model_setting['in_head'], protocol_model_setting['anchor_number'],
                                    kernel_size=1))
            setattr(self, f"reg_head_m0", nn.Conv2d(protocol_model_setting['in_head'], 7 * protocol_model_setting['anchor_number'],
                                    kernel_size=1))            
            setattr(self, f"dir_head_m0", nn.Conv2d(protocol_model_setting['in_head'], \
                            protocol_model_setting['dir_args']['num_bins'] * protocol_model_setting['anchor_number'],
                                    kernel_size=1))
        
        # setup local model
        for modality_name in self.modality_name_list:
            model_setting = args[modality_name]
            sensor_name = model_setting['sensor_type']
            self.sensor_type_dict[modality_name] = sensor_name
            fusion_name = model_setting['fusion_net']["method"]

            # import model
            encoder_filename = "opencood.models.heter_encoders"
            encoder_lib = importlib.import_module(encoder_filename)
            encoder_class = None
            target_model_name = model_setting['core_method'].replace('_', '')

            for name, cls in encoder_lib.__dict__.items():
                if name.lower() == target_model_name.lower():
                    encoder_class = cls

            """
            Encoder building
            """
            setattr(self, f"encoder_{modality_name}", encoder_class(model_setting['encoder_args']))
            if model_setting['encoder_args'].get("depth_supervision", False):
                setattr(self, f"depth_supervision_{modality_name}", True)
            else:
                setattr(self, f"depth_supervision_{modality_name}", False)

            """
            Backbone building 
            """
            setattr(self, f"backbone_{modality_name}", ResNetBEVBackbone(model_setting['backbone_args']))

            """
            Aligner building
            """
            setattr(self, f"aligner_{modality_name}", AlignNet(model_setting['aligner_args']))
            if sensor_name == "camera":
                camera_mask_args = model_setting['camera_mask_args']
                setattr(self, f"crop_ratio_W_{modality_name}", (self.cav_range[3]) / (camera_mask_args['grid_conf']['xbound'][1]))
                setattr(self, f"crop_ratio_H_{modality_name}", (self.cav_range[4]) / (camera_mask_args['grid_conf']['ybound'][1]))
                setattr(self, f"xdist_{modality_name}", (camera_mask_args['grid_conf']['xbound'][1] - camera_mask_args['grid_conf']['xbound'][0]))
                setattr(self, f"ydist_{modality_name}", (camera_mask_args['grid_conf']['ybound'][1] - camera_mask_args['grid_conf']['ybound'][0]))
                self.cam_crop_info[modality_name] = {
                    f"crop_ratio_W_{modality_name}": eval(f"self.crop_ratio_W_{modality_name}"),
                    f"crop_ratio_H_{modality_name}": eval(f"self.crop_ratio_H_{modality_name}"),
                }

            """
            Communication module
            """
            
            setattr(self, f"comm_{modality_name}", CommForStamp(model_setting['comm_args']))
            # setattr(self, f"comm_{modality_name}", ComminPubVanilla(model_setting['comm_args'], self.pub_codebook, init_codebook))
            
            
            """
            Fusion, by default multiscale fusion: 
            Note the input of PyramidFusion has downsampled 2x. (SECOND required)
            """
            setattr(self, f"fusion_net_{modality_name}", build_fusion_net(model_setting['fusion_net']))
            setattr(self, f"fusion_method_{modality_name}", fusion_name)

            """
            Heads
            """
            setattr(self, f"cls_head_{modality_name}", nn.Conv2d(model_setting['in_head'], model_setting['anchor_number'],
                                    kernel_size=1))
            setattr(self, f"reg_head_{modality_name}", nn.Conv2d(model_setting['in_head'], 7 * model_setting['anchor_number'],
                                    kernel_size=1))            
            setattr(self, f"dir_head_{modality_name}", nn.Conv2d(model_setting['in_head'], \
                            model_setting['dir_args']['num_bins'] * model_setting['anchor_number'],
                                    kernel_size=1))
            
        self.compress = False
        # if 'compressor' in args:
        #     self.compress = True
        #     self.compressor = NaiveCompressor(args['compressor']['input_dim'],
        #                                       args['compressor']['compress_ratio'])
        
        """For feature transformation"""
        self.H = (self.cav_range[4] - self.cav_range[1])
        self.W = (self.cav_range[3] - self.cav_range[0])
        self.fake_voxel_size = 1
    
    
        """ 预设置全部模型参数冻结 """    
        for name, p in self.named_parameters():
            p.requires_grad = False
        for name, module in self.named_modules():
            module.eval()
    
        """ 更新本地comm模块的参数 """
        # if hasattr(self, 'inference') and (not self.inference):
        if not self.inference:
            # 已加入联盟的智能体模态类别
            for modality_name in self.modality_name_list:
                self.set_mode_comm_trainable(modality_name)
                print(f"New agent type: {modality_name}")
                
                        
        # check again which module is not fixed.
        check_trainable_module(self)

    def set_mode_comm_trainable(self, modality_name):
        for name, p in self.named_parameters():
            if f'comm_{modality_name}' in name:
                p.requires_grad = True
        for name, module in self.named_modules():
            if f'comm_{modality_name}' in name:
                 module.train()


    def forward(self, data_dict):
        output_dict = {"local_modality_list": self.modality_name_list}
        affine_matrix = normalize_pairwise_tfm(data_dict['pairwise_t_matrix'], self.H, self.W, self.fake_voxel_size)
        record_len = data_dict['record_len'] 


        """Get protocol feature"""
        protocol_feature = eval(f"self.encoder_m0")(data_dict, 'm0')
        protocol_feature = eval(f"self.backbone_m0")({"spatial_features": protocol_feature})['spatial_features_2d']
        protocol_feature = eval(f"self.aligner_m0")(protocol_feature)
            
        if self.sensor_type_dict['m0'] == "camera":
            # should be padding. Instead of masking
            _, _, H, W = protocol_feature.shape
            target_H = int(H*eval(f"self.crop_ratio_H_'m0'"))
            target_W = int(W*eval(f"self.crop_ratio_W_'m0'"))

            # 对齐特征尺寸
            crop_func = torchvision.transforms.CenterCrop((target_H, target_W))
            protocol_feature = crop_func(protocol_feature)
            if eval(f"self.depth_supervision_'m0'"):
                output_dict.update({
                    f"depth_items_'m0'": eval(f"self.encoder_'m0'").depth_items
                })

        reverted_procotol_feat_dict = {}
        for modality_name in self.modality_name_list:
            reverted_procotol_feat = eval(f'self.comm_{modality_name}.revert')(protocol_feature)
            reverted_procotol_feat_dict[modality_name] = reverted_procotol_feat
        
        output_dict.update({
                    'protocol_feature': protocol_feature,
                    'reverted_procotol_feat': reverted_procotol_feat_dict,
                    })

        

        """Get local feature"""
        local_feature_dict = {}
        adapted_local_feat_dict = {}
        reverted_local_feat_dict = {}
        for modality_name in self.modality_name_list:
            feature = eval(f"self.encoder_{modality_name}")(data_dict, modality_name)
            feature = eval(f"self.backbone_{modality_name}")({"spatial_features": feature})['spatial_features_2d']
            feature = eval(f"self.aligner_{modality_name}")(feature)
            
            """
            Crop/Padd camera feature map.
            """
            for modality_name in self.modality_name_list:
                if self.sensor_type_dict[modality_name] == "camera":
                    # should be padding. Instead of masking
                    _, _, H, W = feature.shape
                    target_H = int(H*eval(f"self.crop_ratio_H_{modality_name}"))
                    target_W = int(W*eval(f"self.crop_ratio_W_{modality_name}"))

                    # 对齐特征尺寸
                    crop_func = torchvision.transforms.CenterCrop((target_H, target_W))
                    feature = crop_func(feature)
                    if eval(f"self.depth_supervision_{modality_name}"):
                        output_dict.update({
                            f"depth_items_{modality_name}": eval(f"self.encoder_{modality_name}").depth_items
                        })
            
            local_feature_dict[modality_name] = feature

            adapted_feature = eval(f"self.comm_{modality_name}.adapt")(feature)      
            adapted_local_feat_dict[modality_name] = adapted_feature

            reverted_feature = eval(f"self.comm_{modality_name}.revert")(adapted_feature)   
            reverted_local_feat_dict[modality_name] = reverted_feature   

        output_dict.update({
            "local_feat": deep_clone_dict(local_feature_dict),
            "adapted_local_feat": deep_clone_dict(adapted_local_feat_dict),
            "reverted_local_feat": deep_clone_dict(reverted_local_feat_dict),
            })

        
        """Perform Tasks"""
        local_fusion_method_dict = {}


        preds_reverted_protocol_dict = {}
        for modality_name in self.modality_name_list:
            """Fuse and output""" 
            fusion_method = eval(f"self.fusion_method_{modality_name}")
            local_fusion_method_dict[modality_name] = fusion_method
            
            feature = reverted_procotol_feat_dict[modality_name]
            if fusion_method == 'pyramid':
                fused_feature, occ_outputs = eval(f"self.fusion_net_{modality_name}")(
                                                        feature,
                                                        record_len, 
                                                        affine_matrix)       
            else:
                fused_feature = eval(f'self.fusion_net_{modality_name}')(feature, record_len, affine_matrix)
                occ_outputs = None            

            cls_preds = eval(f"self.cls_head_{modality_name}")(fused_feature)
            reg_preds = eval(f"self.reg_head_{modality_name}")(fused_feature)
            dir_preds = eval(f"self.dir_head_{modality_name}")(fused_feature)
        
            preds_reverted_protocol_dict.update({
                                modality_name:
                                {'cls_preds': cls_preds,
                                'reg_preds': reg_preds,
                                'dir_preds': dir_preds,
                                'pyramid': 'collab',
                                'occ_single_list': occ_outputs}
                                })
            
        
        preds_reverted_local_dict = {}
        for modality_name in self.modality_name_list:
            """Fuse and output""" 
            fusion_method = eval(f"self.fusion_method_{modality_name}")
            
            feature = reverted_local_feat_dict[modality_name]
            if fusion_method == 'pyramid':
                fused_feature, occ_outputs = eval(f"self.fusion_net_{modality_name}")(
                                                        feature,
                                                        record_len, 
                                                        affine_matrix)       
            else:
                fused_feature = eval(f'self.fusion_net_{modality_name}')(feature, record_len, affine_matrix)
                occ_outputs = None            

            cls_preds = eval(f"self.cls_head_{modality_name}")(fused_feature)
            reg_preds = eval(f"self.reg_head_{modality_name}")(fused_feature)
            dir_preds = eval(f"self.dir_head_{modality_name}")(fused_feature)
        
            preds_reverted_local_dict.update({
                                modality_name:
                                {'cls_preds': cls_preds,
                                'reg_preds': reg_preds,
                                'dir_preds': dir_preds,
                                'pyramid': 'collab',
                                'occ_single_list': occ_outputs}
                                })
            

        preds_adapted_local_dict = {}
        protocol_fusion_method = eval(f"self.fusion_method_m0")
        for modality_name in self.modality_name_list:
            """Fuse and output""" 
            
            feature = adapted_local_feat_dict[modality_name]
            if fusion_method == 'pyramid':
                fused_feature, occ_outputs = eval(f"self.fusion_net_m0")(
                                                        feature,
                                                        record_len, 
                                                        affine_matrix)       
            else:
                fused_feature = eval(f'self.fusion_net_m0')(feature, record_len, affine_matrix)
                occ_outputs = None            

            cls_preds = eval(f"self.cls_head_m0")(fused_feature)
            reg_preds = eval(f"self.reg_head_m0")(fused_feature)
            dir_preds = eval(f"self.dir_head_m0")(fused_feature)
        
            preds_adapted_local_dict.update({
                                modality_name:
                                {'cls_preds': cls_preds,
                                'reg_preds': reg_preds,
                                'dir_preds': dir_preds,
                                'pyramid': 'collab',
                                'occ_single_list': occ_outputs}
                                })


        output_dict.update({
            'protocol_fusion_name': protocol_fusion_method,
            'local_fusion_method': local_fusion_method_dict,
            'preds_reverted_protocol': preds_reverted_protocol_dict,
            'preds_reverted_local': preds_reverted_local_dict,
            'preds_adapted_local': preds_adapted_local_dict
        })

        return output_dict
