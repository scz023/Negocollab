""" Author: Yifan Lu <yifan_lu@sjtu.edu.cn>

HEAL: An Extensible Framework for Open Heterogeneous Collaborative Perception 
"""

import torch
import torch.nn as nn
import numpy as np
from icecream import ic
from collections import OrderedDict, Counter
from opencood.models.fuse_modules import build_fusion_net
from opencood.models.sub_modules.base_bev_backbone_resnet import ResNetBEVBackbone 
from opencood.models.sub_modules.domian_adapter_mpda import DomianAdapterMpda, DomianAdapterMpdaProtocol
from opencood.models.sub_modules.feature_alignnet import AlignNet
from opencood.models.sub_modules.downsample_conv import DownsampleConv
from opencood.models.sub_modules.naive_compress import NaiveCompressor
from opencood.models.fuse_modules.pyramid_fuse import PyramidFusion, regroup
from opencood.utils.transformation_utils import normalize_pairwise_tfm
from opencood.utils.model_utils import check_trainable_module, fix_bn, unfix_bn
import importlib
import torchvision

class HeterMpdaProtocol(nn.Module):
    def __init__(self, args):
        super(HeterMpdaProtocol, self).__init__()
        self.args = args
        modality_name_list = list(args.keys())
        modality_name_list = list(set(args['mapping_dict'].values())) 
        self.modality_name_list = modality_name_list

        self.new_agent_type = modality_name_list.copy()
        if 'm0' in self.new_agent_type:
            self.new_agent_type.remove('m0')

        self.cav_range = args['lidar_range']
        self.sensor_type_dict = OrderedDict()

        self.cam_crop_info = {} 
        
        # self.share_feat_dim = args['pub']['share_feat_dim']

        # setup each modality model
        for modality_name in self.modality_name_list:
            model_setting = args[modality_name]
            sensor_name = model_setting['sensor_type']
            self.sensor_type_dict[modality_name] = sensor_name
            fusion_metohd = model_setting['fusion_net']['method']

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


            """Modality Adapt"""
            adapter_args = model_setting["adapter_args"]
            adapter_args.update({"local_modality": modality_name})
            setattr(self, f"adapter_{modality_name}", DomianAdapterMpdaProtocol(adapter_args))


            """
            Fusion, by default multiscale fusion: 
            Note the input of PyramidFusion has downsampled 2x. (SECOND required)
            """
            setattr(self, f"fusion_net_{modality_name}", build_fusion_net(model_setting['fusion_net']))
            setattr(self, f"fusion_method_{modality_name}", fusion_metohd)


            """
            Shrink header, 在pyramid融合方法中, fusion_net会使特征升维, 在这里使用shrink对融合特征进行降维
            """
            # if fusion_metohd == 'pyramid':
            #     setattr(self, f"shrink_flag_{modality_name}", False)
            #     if 'shrink_header' in model_setting:
            #         setattr(self, f"shrink_flag_{modality_name}", True)
            #         setattr(self, f"shrink_conv_{modality_name}", DownsampleConv(model_setting['shrink_header']))

            """
            Shared Heads
            """
            setattr(self, f"cls_head_{modality_name}", nn.Conv2d(model_setting['in_head'], model_setting['anchor_number'],
                                    kernel_size=1))
            setattr(self, f"reg_head_{modality_name}", nn.Conv2d(model_setting['in_head'], 7 * model_setting['anchor_number'],
                                    kernel_size=1))            
            setattr(self, f"dir_head_{modality_name}", nn.Conv2d(model_setting['in_head'], \
                            model_setting['dir_args']['num_bins'] * model_setting['anchor_number'],
                                    kernel_size=1))
        
        # compressor will be only trainable
        self.compress = False
        # if 'compressor' in args:
        #     self.compress = True
        #     self.compressor = NaiveCompressor(args['compressor']['input_dim'],
        #                                       args['compressor']['compress_ratio'])
        
        """For feature transformation"""
        self.H = (self.cav_range[4] - self.cav_range[1])
        self.W = (self.cav_range[3] - self.cav_range[0])
        self.fake_voxel_size = 1

        self.model_train_init()
        # check again which module is not fixed.
        check_trainable_module(self)


    def model_train_init(self):
        # 首先设置所有参数均不更新
        for name, p in self.named_parameters():
            p.requires_grad = False
        for name, module in self.named_modules():
            module.eval()

        # 只更新adapter的参数
        # for name, p in self.named_parameters():
        #     if 'adapter' in name:
        #         p.requires_grad = True       
        # for name, module in self.named_modules():
        #     if 'adapter' in name:
        #         module.train()

        # 只调整adaper中, 非enhancer模型的参数
        for name, p in self.named_parameters():
            if 'adapter_' in name:
                p.requires_grad = True
                
        for name, module in self.named_children():
            if 'adapter_' in name:
                for name_sub_modu, sub_module in module.named_modules():
                    sub_module.train()

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
        agent_modality_list = data_dict['agent_modality_list'] 
        affine_matrix = normalize_pairwise_tfm(data_dict['pairwise_t_matrix'], self.H, self.W, self.fake_voxel_size)
        record_len = data_dict['record_len'] 
        # print(agent_modality_list)
        modality_count_dict = Counter(agent_modality_list)
        modality_feature_dict = {}

        for modality_name in self.modality_name_list:
            if modality_name not in modality_count_dict:
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

                    # 对齐特征尺寸
                    crop_func = torchvision.transforms.CenterCrop((target_H, target_W))
                    modality_feature_dict[modality_name] = crop_func(feature)
                    if eval(f"self.depth_supervision_{modality_name}"):
                        output_dict.update({
                            f"depth_items_{modality_name}": eval(f"self.encoder_{modality_name}").depth_items
                        })        

        """
        Align to common space.
        """
        protocol_feature_dict = {}
        for modality_name in self.modality_name_list:
            if modality_name not in modality_count_dict:
                continue
            protocol_feature_dict[modality_name] = \
                eval(f'self.adapter_{modality_name}.align')(modality_feature_dict[modality_name])

        """
        Assemble protocol features
        """
        protocol_feature_list = self.assemble_heter_features(agent_modality_list, protocol_feature_dict)
        protocol_feature = torch.stack(protocol_feature_list)

        
        ego_modality = agent_modality_list[0]
        record_len_modality = data_dict['record_len_modality'][ego_modality]
        reverted_feature, da_cls_pred, source_idx = eval(f'self.adapter_{ego_modality}.revert')\
                                                (protocol_feature, \
                                                 agent_modality_list, \
                                                 record_len, \
                                                 record_len_modality, \
                                                 affine_matrix)

        #  = domain_classify(reverted_feature)
        # if self.compress:
        #     adapted_heter_feature_2d = self.compressor(adapted_heter_feature_2d)

        # heter_feature_2d is downsampled 2x
        # add croping information to collaboration module
        

        fusion_method = eval(f"self.fusion_method_{ego_modality}")
        output_dict.update({'fusion_method': fusion_method})
        
        if fusion_method == 'pyramid':
            fused_feature, occ_outputs = eval(f"self.fusion_net_{ego_modality}")(
                                                    reverted_feature,
                                                    record_len, 
                                                    affine_matrix, 
                                                    agent_modality_list, 
                                                    self.cam_crop_info,
                                                )  
            # fused_feature, occ_outputs = eval(f"self.fusion_net_{ego_modality}.forward_collab")(
            #                                         adapted_heter_feature_2d,
            #                                         record_len, 
            #                                         affine_matrix, 
            #                                         agent_modality_list, 
            #                                         self.cam_crop_info,
            #                                     )
            # fused_feature = eval(f"self.shrink_conv_{ego_modality}")(fused_feature)
        else:
            fused_feature = eval(f'self.fusion_net_{ego_modality}')(reverted_feature, record_len, affine_matrix)
            occ_outputs = None
            

        cls_preds = eval(f'self.cls_head_{ego_modality}')(fused_feature)
        reg_preds = eval(f'self.reg_head_{ego_modality}')(fused_feature)
        dir_preds = eval(f'self.dir_head_{ego_modality}')(fused_feature)

        output_dict.update({'cls_preds': cls_preds,
                            'reg_preds': reg_preds,
                            'dir_preds': dir_preds})
        
        output_dict.update({'occ_single_list': 
                            occ_outputs})
        
        output_dict.update({'da_cls_preds': da_cls_pred,
                            'source_idx': source_idx
                            })
        
        output_dict.update({'ego_modality': ego_modality})

        return output_dict
