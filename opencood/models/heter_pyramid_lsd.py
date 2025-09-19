""" Author: Yifan Lu <yifan_lu@sjtu.edu.cn>

HEAL: An Extensible Framework for Open Heterogeneous Collaborative Perception 
"""

import torch
import torch.nn as nn
import numpy as np
from icecream import ic
import torchvision
from collections import OrderedDict, Counter
from opencood.models.sub_modules.base_bev_backbone_resnet import ResNetBEVBackbone 
from opencood.models.fuse_modules.pyramid_fuse import PyramidFusion
from opencood.models.sub_modules.comm import Comm
from opencood.models.sub_modules.feature_alignnet import AlignNet
from opencood.models.sub_modules.downsample_conv import DownsampleConv
import importlib
from opencood.models.sub_modules.focuser import Focuser
from opencood.utils.model_utils import check_trainable_module, fix_bn, unfix_bn
from opencood.tools.feature_show import feature_show

class HeterPyramidLSD(nn.Module):
    def __init__(self, args):
        super(HeterPyramidLSD, self).__init__()
        modality_name_list = list(args.keys())
        modality_name_list = [x for x in modality_name_list if x.startswith("m") and x[1:].isdigit()] 
        self.modality_name_list = modality_name_list
        self.cav_range = args['lidar_range']
        self.sensor_type_dict = OrderedDict()
        # self.adjust_modules = ['comm']
        
        
        # setup each modality model
        for modality_name in self.modality_name_list:
            model_setting = args[modality_name]
            sensor_name = model_setting['sensor_type']
            self.sensor_type_dict[modality_name] = sensor_name

            # import model
            encoder_filename = "opencood.models.heter_encoders"
            encoder_lib = importlib.import_module(encoder_filename)
            encoder_class = None
            target_model_name = model_setting['core_method'].replace('_', '')

            for name, cls in encoder_lib.__dict__.items():
                if name.lower() == target_model_name.lower():
                    encoder_class = cls

            # build encoder
            setattr(self, f"encoder_{modality_name}", encoder_class(model_setting['encoder_args']))
            # depth supervision for camera
            if model_setting['encoder_args'].get("depth_supervision", False) :
                setattr(self, f"depth_supervision_{modality_name}", True)
            else:
                setattr(self, f"depth_supervision_{modality_name}", False)

            # setup backbone (very light-weight)
            setattr(self, f"backbone_{modality_name}", ResNetBEVBackbone(model_setting['backbone_args']))
            if sensor_name == "camera":
                camera_mask_args = model_setting['camera_mask_args']
                setattr(self, f"crop_ratio_W_{modality_name}", (self.cav_range[3]) / (camera_mask_args['grid_conf']['xbound'][1]))
                setattr(self, f"crop_ratio_H_{modality_name}", (self.cav_range[4]) / (camera_mask_args['grid_conf']['ybound'][1]))

            setattr(self, f"aligner_{modality_name}", AlignNet(model_setting['aligner_args']))

            # Only select important feature to comunicate, used to improve reconstruction efficiency
            # setattr(self, f"focuser_{modality_name}", Focuser(model_setting['focuser_args']))

            # build comm module, include sender and receiver
            setattr(self, f"comm_{modality_name}", Comm(model_setting['comm_args']))

            # if args.get("fix_encoder", False):
                # self.fix_modules += [f"encoder_{modality_name}", f"backbone_{modality_name}"]

            """
            Would load from pretrain base.
            """
            setattr(self, f"pyramid_backbone_{modality_name}", PyramidFusion(model_setting['fusion_backbone']))
            # self.pyramid_backbone = PyramidFusion(args['fusion_backbone'])
            """
            Shrink header, Would load from pretrain base.
            """
            # self.shrink_flag = False
            # if 'shrink_header' in args:
            #     self.shrink_flag = True
            #     self.shrink_conv = DownsampleConv(args['shrink_header'])
                # self.fix_modules.append('shrink_conv')

            setattr(self, f"shrink_flag_{modality_name}", False)
            if 'shrink_header' in model_setting:
                setattr(self, f"shrink_flag_{modality_name}", True)
                setattr(self, f"shrink_conv_{modality_name}", DownsampleConv(model_setting['shrink_header']))
            
            """
            Shared Heads, Would load from pretrain base.
            """
            # self.cls_head = nn.Conv2d(args['in_head'], args['anchor_number'],
            #                         kernel_size=1)
            # self.reg_head = nn.Conv2d(args['in_head'], 7 * args['anchor_number'],
            #                         kernel_size=1)
            # self.dir_head = nn.Conv2d(args['in_head'], args['dir_args']['num_bins'] * args['anchor_number'],
            #                         kernel_size=1) # BIN_NUM = 2
            
            setattr(self, f"cls_head_{modality_name}", nn.Conv2d(model_setting['in_head'], model_setting['anchor_number'],
                                    kernel_size=1))
            setattr(self, f"reg_head_{modality_name}", nn.Conv2d(model_setting['in_head'], 7 * model_setting['anchor_number'],
                                    kernel_size=1))            
            setattr(self, f"dir_head_{modality_name}", nn.Conv2d(model_setting['in_head'], \
                            model_setting['dir_args']['num_bins'] * model_setting['anchor_number'],
                                    kernel_size=1))
        
        
        """
        设置通信模块中所有参数为可更新
        """
        for modality_name in self.modality_name_list:
            self.model_train_init(modality_name)
        # check again which module is not fixed.
        check_trainable_module(self)

    def model_train_init(self, modality_name):
        # for module in self.adjust_modules:
        # for name, p in self.named_parameters():
        #     if (not ('comm' in name)) and (not ('head' in name)):
        #         p.requires_grad = False
                
        # eval(f"self.pyramid_backbone_{modality_name}").train()
        # if  eval(f"self.shrink_flag_{modality_name}"):
        #     eval(f"self.shrink_conv_{modality_name}").train()
        # eval(f"self.cls_head_{modality_name}").train()
        # eval(f"self.reg_head_{modality_name}").train()
        # eval(f"self.dir_head_{modality_name}").train()
        
        
        for name, p in self.named_parameters():
            # 只训练通信模块和特征选择模块的参数
            if (not ('comm' in name)) and (not ('focuser' in name)):
                p.requires_grad = False
                
        eval(f"self.comm_{modality_name}").train()
            

    def forward(self, data_dict):
        output_dict = {'pyramid': 'single'}
        modality_name = [x for x in list(data_dict.keys()) if x.startswith("inputs_")]
        assert len(modality_name) == 1
        modality_name = modality_name[0].lstrip('inputs_')

        feature = eval(f"self.encoder_{modality_name}")(data_dict, modality_name)
        feature = eval(f"self.backbone_{modality_name}")({"spatial_features": feature})['spatial_features_2d']
        feature = eval(f"self.aligner_{modality_name}")(feature)

        if self.sensor_type_dict[modality_name] == "camera":
            # should be padding. Instead of masking
            _, _, H, W = feature.shape
            feature = torchvision.transforms.CenterCrop(
                    (int(H*eval(f"self.crop_ratio_H_{modality_name}")), int(W*eval(f"self.crop_ratio_W_{modality_name}")))
                )(feature)

            if eval(f"self.depth_supervision_{modality_name}"):
                output_dict.update({
                    f"depth_items_{modality_name}": eval(f"self.encoder_{modality_name}").depth_items
                })
        
        # select important feature to transmit
        # mask, sg_map = eval(f"self.focuser_{modality_name}")(feature)
        # feature_show(mask[0], '/home/scz/HEAL/analysis/masked', type = 'mean')
        # feature_show(feature[0], '/home/scz/HEAL/analysis/feature', type = 'mean')
        # feature = feature * mask    
        # feature_show(feature[0], '/home/scz/HEAL/analysis/feature_masked', type = 'mean')
        
        output_dict.update({'fc_before_send':feature})
        
        # Add sematic decompsition process here.
        weights, fc_before_rbdc = \
            eval(f"self.comm_{modality_name}.sender")(feature)
        
        feature, fc_after_rbdc = eval(f"self.comm_{modality_name}.receiver")(weights)
        
        """
        Save codebook, 
             feature after/before send by decomposed codebooks
             feature after/before reconstruct by decomposed codebooks \
             for Loss computation
        """

        output_dict.update({
                            # 'sg_map': sg_map,
                            'fc_after_send': feature,
                            'fc_before_rbdc': fc_before_rbdc,
                            'fc_after_rbdc': fc_after_rbdc,
                            'codebook': eval(f"self.comm_{modality_name}.codebook")
                            })
        
        # multiscale fusion. 
        feature, occ_map_list = eval(f"self.pyramid_backbone_{modality_name}.forward_single")(feature)

        if eval(f"self.shrink_flag_{modality_name}"):
            feature = eval(f"self.shrink_conv_{modality_name}")(feature)

        cls_preds = eval(f"self.cls_head_{modality_name}")(feature)
        reg_preds = eval(f"self.reg_head_{modality_name}")(feature)
        dir_preds = eval(f"self.dir_head_{modality_name}")(feature)

        output_dict.update({'cls_preds': cls_preds,
                            'reg_preds': reg_preds,
                            'dir_preds': dir_preds})
        output_dict.update({'occ_single_list': 
                            occ_map_list})
        return output_dict
    