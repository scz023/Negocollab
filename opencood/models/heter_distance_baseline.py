""" Author: Yifan Lu <yifan_lu@sjtu.edu.cn>

HEAL: An Extensible Framework for Open Heterogeneous Collaborative Perception 
"""

import torch
import torch.nn as nn
import numpy as np
from icecream import ic
from collections import OrderedDict, Counter
# from opencood.models.comm_modules.comm_for_stamp import CommForStamp
from opencood.models.comm_modules.comm_for_stamp import CommForStamp
from opencood.models.comm_modules.comm_in_pub import ComminPub
import torch.nn.functional as F
from opencood.models.fuse_modules import build_fusion_net
from opencood.models.sub_modules.base_bev_backbone_resnet import ResNetBEVBackbone 
from opencood.models.sub_modules.domian_adapter_mpda import DomianAdapterMpdaProtocol
from opencood.models.sub_modules.domian_adapter_pnpda import DomianAdapter, DomianAdapterProtocol, ResizeNet
from opencood.models.sub_modules.feature_alignnet import AlignNet
from opencood.models.sub_modules.downsample_conv import DownsampleConv
from opencood.models.sub_modules.keyfeat_modules import Negotiator
from opencood.models.sub_modules.naive_compress import NaiveCompressor
from opencood.models.fuse_modules.pyramid_fuse import PyramidFusion, regroup
from opencood.tools.feature_show import feature_show
from opencood.utils.transformation_utils import normalize_pairwise_tfm
from opencood.utils.model_utils import check_trainable_module, fix_bn, unfix_bn
from opencood.loss.distribution_distance import MMD, CrossEntropyDistance, EMDSinkhornDistance, EMD, KLDivergence
import importlib
import torchvision

class HeterDistanceBaseline(nn.Module):
    def __init__(self, args):
        super(HeterDistanceBaseline, self).__init__()
        
        self.args = args
        modality_name_list = list(set(args['modality_setting'].keys())) 
        
        self.modality_name_list = modality_name_list
        self.cav_range = args['lidar_range']
        self.sensor_type_dict = OrderedDict()
        self.cam_crop_info = {} 
        
        
        self.test_modality_list = modality_name_list.copy()
        if 'm0' in self.test_modality_list:
            self.test_modality_list.remove('m0')
        
        adapter_name = args['adapter_name']
        
        if adapter_name == 'nego':
            self.ori_modality_name_list = args['ori_modality']
        # print(self.ori_modality_name_list)
        
        self.adapter_name = adapter_name
        
        # setup each modality model
        for modality_name in self.modality_name_list:
            model_setting = args[modality_name]
            sensor_name = model_setting['sensor_type']
            self.sensor_type_dict[modality_name] = sensor_name
            # fusion_metohd = model_setting['fusion_net']['method']

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
            if adapter_name == 'pnpda':
                adapter_args = model_setting["adapter_args"]
                adapter_args.update({"local_modality": modality_name,
                                    'stage': 'distance'})
                setattr(self, f"adapter_{modality_name}", DomianAdapterProtocol(adapter_args))
            elif adapter_name == 'mpda':
                adapter_args = model_setting["adapter_args"]
                adapter_args.update({"local_modality": modality_name,
                                    'stage': 'distance'})
                setattr(self, f"adapter_{modality_name}", DomianAdapterMpdaProtocol(adapter_args))
            elif adapter_name == 'stamp':
                if modality_name == 'm0':
                    continue
                adapter_args = model_setting["comm_args"]
                setattr(self, f"comm_{modality_name}", CommForStamp(adapter_args))
            elif adapter_name == 'nego':
                adapter_args = model_setting["comm_args"]
                model_setting['comm_args'].update({'local_range': self.cav_range,
                                    'modality': modality_name
                                    })
                setattr(self, f"comm_{modality_name}", ComminPub(adapter_args))
                setattr(self, f"nego_reziser_{modality_name}", ResizeNet(model_setting['local_dim'], args['pub']['C_uni']))
        
        distance_method = args['distance']['method']

        
        if 'pub' in args.keys():
            args_pub = args['pub']
            # if (self.allied_num) == 0 and (not self.pc_grad):
            # if not self.pc_grad:
            self.negotiator = Negotiator(args_pub['negotiator']) if 'negotiator' in args_pub.keys() else None
        
        if distance_method == 'emd_sinkhorn':
            self.distance_func = EMDSinkhornDistance(args['distance']['args'])
        elif distance_method == 'emd':
            self.distance_func = EMD(args['distance']['args'])
        elif distance_method == 'mmd':
            self.distance_func = MMD(args['distance']['args'])
        elif distance_method == 'kl':
            self.distance_func = KLDivergence()
        elif distance_method == 'ce':
            self.distance_func = CrossEntropyDistance()
        
        for name, p in self.named_parameters():
            p.requires_grad = False
        for name, module in self.named_modules():
            module.eval()
        

        kernel_size = 5
        c_sigma = 1
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

    def crop_and_mask(self, feature, target_dict, crop_ratio):

        _, _, H, W = feature.shape
        targets = target_dict['pos_equal_one']
        negatives = target_dict['neg_equal_one']  
        targets = torch.logical_or(targets[...,0], targets[...,1]).unsqueeze(1)
        negatives = torch.logical_and(negatives[...,0], negatives[...,1]).unsqueeze(1)
        # print(targets.size())
        # print(negatives.size())
        mask = torch.logical_or(targets, negatives).float()
               
        mask = F.interpolate(mask, (H, W), mode='bilinear')
        mask = self.gaussian_filter(mask)       
        mask = torch.where(mask > 0, 1, 0)

        # feature = feature * mask
        
        target_H = int(H*crop_ratio)
        target_W = int(W*crop_ratio)
        crop_func = torchvision.transforms.CenterCrop((target_H, target_W))
        
        feature = crop_func(feature)

        return feature    
        
    def forward(self, data_dict, target_dict):        
        
        modality_feature_dict = {}
        for modality_name in self.test_modality_list:
            feature = eval(f"self.encoder_{modality_name}")(data_dict, modality_name)
            feature = eval(f"self.backbone_{modality_name}")({"spatial_features": feature})['spatial_features_2d']
            feature = eval(f"self.aligner_{modality_name}")(feature)
            modality_feature_dict[modality_name] = feature
             
        """
        Crop/Padd camera feature map.
        """
        for modality_name in self.modality_name_list:
            # if modality_name in modality_count_dict:
            if self.sensor_type_dict[modality_name] == "camera":
                # should be padding. Instead of masking
                feature = modality_feature_dict[modality_name]
                _, _, H, W = feature.shape
                target_H = int(H*eval(f"self.crop_ratio_H_{modality_name}"))
                target_W = int(W*eval(f"self.crop_ratio_W_{modality_name}"))

                # 对齐特征尺寸
                crop_func = torchvision.transforms.CenterCrop((target_H, target_W))
                modality_feature_dict[modality_name] = crop_func(feature)
                # if eval(f"self.depth_supervision_{modality_name}"):
                #     output_dict.update({
                #         f"depth_items_{modality_name}": eval(f"self.encoder_{modality_name}").depth_items
                #     })


        """ Align to protocol feature space"""
        protocol_feature_dict = {}
        for modality_name in self.test_modality_list:
            # if modality_name not in modality_count_dict:
            #     continue
            if self.adapter_name == 'stamp':
                protocol_feature_dict[modality_name] = eval(f'self.comm_{modality_name}.adapt')(modality_feature_dict[modality_name])
            elif self.adapter_name == 'nego':
                protocol_feature_dict[modality_name] = eval(f'self.comm_{modality_name}.send')(modality_feature_dict[modality_name])
            else:
                protocol_feature_dict[modality_name] = eval(f'self.adapter_{modality_name}.align')(modality_feature_dict[modality_name])
        
        """ Get protocol feature"""
        if self.adapter_name != 'nego':
            protocol_feature = eval(f"self.encoder_m0")(data_dict, 'm0')
            protocol_feature = eval(f"self.backbone_m0")({"spatial_features": protocol_feature})['spatial_features_2d']
            protocol_feature = eval(f"self.aligner_m0")(protocol_feature)
        
        else:
            standard_size = list(protocol_feature_dict.values())[0][0].size()
            standard_feature_list = []

            for modality_name in self.ori_modality_name_list:
                feature = modality_feature_dict[modality_name]
                # standard_feature_list.append(feature)
                standard_feature_list.append(eval(f"self.nego_reziser_{modality_name}")\
                    (feature, standard_size))     
            protocol_feature = self.negotiator(standard_feature_list)



        """
        Analyse domain gap of sub features.
        """
        crop_ratio = 0.3
        for modality_name in self.test_modality_list:
            feature = protocol_feature_dict[modality_name]
            feature = self.crop_and_mask(feature, target_dict[modality_name], crop_ratio)
            protocol_feature_dict[modality_name] = feature

            feature = modality_feature_dict[modality_name]
            feature = self.crop_and_mask(feature, target_dict[modality_name], crop_ratio)
            modality_feature_dict[modality_name] = feature

        if self.adapter_name == 'nego':
            protocol_feature = self.crop_and_mask(protocol_feature, target_dict['nego'], crop_ratio)
        else:
            protocol_feature = self.crop_and_mask(protocol_feature, target_dict['m0'], crop_ratio)


        """Compute domain distance"""
        modality_dis_ori_dict = {}
        modality_dis_aligned_dict = {}
        for modality_name in self.test_modality_list:
            modality_feature_ori = modality_feature_dict[modality_name]
            modality_dis_ori_dict[modality_name] = self.distance_func(protocol_feature, modality_feature_ori)
            
            # print(torch.equal(protocol_feature, modality_feature_ori))
            
            modality_feature_aligned = protocol_feature_dict[modality_name]
            modality_dis_aligned_dict[modality_name] = self.distance_func(protocol_feature, modality_feature_aligned)
    
        return modality_dis_ori_dict, modality_dis_aligned_dict