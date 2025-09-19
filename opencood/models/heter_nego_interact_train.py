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
from opencood.models.comm_modules.comm_in_pub import ComminPub
from opencood.models.fuse_modules import build_fusion_net
from opencood.models.sub_modules.base_bev_backbone_resnet import ResNetBEVBackbone 
from opencood.models.comm_modules.comm_in_pub_decom import ComminPub
# from opencood.models.sub_modules.comm_in_local import ComminLocal
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

class HeterNegoInteractTrain(nn.Module):
    def __init__(self, args):
        super(HeterNegoInteractTrain, self).__init__()
       
        self.modality_name_list = list(set(args['mapping_dict'].values()))

        self.cav_range = args['lidar_range']
        self.sensor_type_dict = OrderedDict()

        self.cam_crop_info = {} 
        
        self.allied = {}
        
        self.allied_num = 0
        
        self.allied_modality_list = []

        self.stage = args['stage'] if 'stage' in args.keys() else 'inf'

        self.pc_grad = args['pub']['pc_grad'] if 'pc_grad' in args['pub'].keys() else False
        
        # setup each modality model
        for modality_name in self.modality_name_list:
            model_setting = args[modality_name]
            sensor_name = model_setting['sensor_type']
            self.sensor_type_dict[modality_name] = sensor_name
            fusion_name = model_setting['fusion_net']["method"]
            
            """
            若在联盟中, 固定comm模块参数; 若不在联盟中, 调整comm模块参数
            """
            self.allied[modality_name] = model_setting['allied']
            if model_setting['allied']:
                self.allied_num += 1
                self.allied_modality_list.append(modality_name)

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
            Communication module from codebook
            """
            model_setting['comm_args'].update({'local_range': self.cav_range,
                                               'comm_space': args['comm_space'],
                                               'modality': modality_name
                                               })
            setattr(self, f"comm_{modality_name}", ComminPub(model_setting['comm_args']))
            # setattr(self, f"comm_{modality_name}", ComminPubVanilla(model_setting['comm_args'], self.pub_codebook, init_codebook))
            
            
            """
            Fusion, by default multiscale fusion: 
            Note the input of PyramidFusion has downsampled 2x. (SECOND required)
            """
            setattr(self, f"fusion_net_{modality_name}", build_fusion_net(model_setting['fusion_net']))
            setattr(self, f"fusion_method_{modality_name}", fusion_name)


            """
            Shrink header
            """
            if fusion_name == "pyramid":
                setattr(self, f"shrink_flag_{modality_name}", False)
                if 'shrink_header' in model_setting:
                    setattr(self, f"shrink_flag_{modality_name}", True)
                    setattr(self, f"shrink_conv_{modality_name}", DownsampleConv(model_setting['shrink_header']))

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
            
            setattr(self, f"nego_reziser_{modality_name}", ResizeNet(model_setting['local_dim'], args['pub']['C_uni']))
            
            """梯度手术方法时, 通过为每个模态备份单独的negotiator获得关于模态重构损失的专属梯度, 以此权衡各模态信息"""
            if self.pc_grad:
                setattr(self, f"negotiator_bak_{modality_name}",  Negotiator(args['pub']['negotiator'])) # nego: 物理上的两个用于产生不同梯度, 逻辑上的一个
            
        
        # if (self.allied_num) == 0 and (not self.pc_grad):
        if not self.pc_grad:
            self.negotiator = Negotiator(args['pub']['negotiator'])
        
        # if self.stage == 'ft':
        #     common_rep_head = args['common_rep_head']
        #     self.cls_head_common_rep = nn.Conv2d(common_rep_head['in_head'], common_rep_head['anchor_number'],
        #                             kernel_size=1)
        #     self.reg_head_common_rep = nn.Conv2d(common_rep_head['in_head'], 7 * common_rep_head['anchor_number'],
        #                             kernel_size=1)
        #     self.dir_head_common_rep = nn.Conv2d(common_rep_head['in_head'], common_rep_head['anchor_number'],
        #                             kernel_size=1)

        # self.
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
    
    
        """ 预设置全部模型参数冻结 """    
        for name, p in self.named_parameters():
            p.requires_grad = False
        for name, module in self.named_modules():
            module.eval()
    
        """ 仅更新未加入联盟的comm模块的参数 """
        # 已加入联盟的智能体模态类别
        for modality_name in self.modality_name_list:
            if not self.allied[modality_name]:
                self.set_mode_comm_trainable(modality_name)
                print(f"New agent type: {modality_name}")
                
        self.newtype_modality_list = list(set(self.modality_name_list) - set(self.allied_modality_list))
        
        if self.allied_num == 0:
            if self.stage == 'nego': # 全为未在联盟内的代理时, 初始化各模态备份的协商器参数为相同值
                self.set_nego_trainable()   # 这个会影响inference的feature的可视化效果！
                if self.pc_grad:
                    self._negotiator_init(self.newtype_modality_list[0])
            # self.set_nego_trainable()
            # elif self.stage == 'unify': # 在发送器对齐阶段, 使用备份nego的参数初始化ori nego
            #     self.nego_bak_combine()
        # else: # 新代理加入时, 同样使用nego辅助训练
            # self.set_nego_trainable()
                        
        # check again which module is not fixed.
        check_trainable_module(self)

    def set_mode_comm_trainable(self, modality_name):
        for name, p in self.named_parameters():
            if f'comm_{modality_name}' in name:
                p.requires_grad = True
        for name, module in self.named_modules():
            if f'comm_{modality_name}' in name:
                 module.train()

    def set_nego_trainable(self):
        for name, p in self.named_parameters():
            if f'nego' in name:
                p.requires_grad = True

        for name, module in self.named_modules():
            if f'nego' in name:
                 module.train()
                 
    def _negotiator_init(self, nego_ref_mode):
        # negotiator 共享参数神经网络初始化为相同的参数
        for modality_name in self.newtype_modality_list:
            for para_sign, para_mode in zip(eval(f'self.negotiator_bak_{nego_ref_mode}.parameters()'), \
                                            eval(f'self.negotiator_bak_{modality_name}.parameters()')):
                para_mode.data.copy_(para_sign.data)
                
                # 若已有联盟内代理, 不再更新negotiator参数
                if len(self.allied_modality_list) > 0 or self.stage == 'ft':
                    para_mode.requires_grad = False
                    

    """设置仅更新comm或者nego模块的参数。由于协商器只在stage1的初步协商阶段使用, 此时所有类型参数均参与训练, 因此不用区分modality
       New Type加入时, 直接使用sender辅助训练, 无需再使用协商器"""
    def set_only_comm_trainable(self):
        for name, p in self.named_parameters():                
            if f'comm' in name:
                p.requires_grad = True
            else:
                p.requires_grad = False
                
        for name, module in self.named_modules():
            if f'comm' in name:
                 module.train()
            else:
                module.eval()

    def set_only_nego_trainable(self):
        for name, p in self.named_parameters():
            if f'nego' in name:
                p.requires_grad = True
            else:
                p.requires_grad = False
        for name, module in self.named_modules():
            if f'nego' in name:
                 module.train()
            else:
                module.eval()    
    
    def nego_bak_combine(self):
        ref_mode = self.modality_name_list[0]
        for p, bak_p in zip(self.negotiator.parameters(), \
                            eval(f'self.negotiator_bak_{ref_mode}.parameters()')): 
            p.data = bak_p.data.clone()
        

    def comm_same_init(self):
        # 通信模块网络初始化为相同的参数
        
        sign_m = self.newtype_modality_list[0]

        for modality_name in self.newtype_modality_list:
            for para_sign, para_mode in zip(eval(f'self.comm_{sign_m}.parameters()'), \
                                            eval(f'self.comm_{modality_name}.parameters()')):
                para_mode.data.copy_(para_sign.data)


    def forward(self, data_dict):
        
        # 联盟内无代理, 联合更新comm 中 negotiator的参数
        # if len(self.allied_modality_list) <= 0:
        #     self._negotiator_para_update()
        
        output_dict = {'pyramid': 'single', 'modality_name_list': self.modality_name_list, 
                       'newtype_modality_list': self.newtype_modality_list}        
        
        # modality_type_list = data_dict['modality_type_list'] 
        # agent_modality_list = data_dict['agent_modality_list'] 
        affine_matrix = normalize_pairwise_tfm(data_dict['pairwise_t_matrix'], self.H, self.W, self.fake_voxel_size)
        record_len = data_dict['record_len'] 
        # print('\ninputbatch:', sum(record_len))
        modality_feature_dict = {}
        
        for modality_name in self.modality_name_list:
            feature = eval(f"self.encoder_{modality_name}")(data_dict, modality_name)
            feature = eval(f"self.backbone_{modality_name}")({"spatial_features": feature})['spatial_features_2d']
            feature = eval(f"self.aligner_{modality_name}")(feature)
            modality_feature_dict[modality_name] = feature
            
        """
        Crop/Padd camera feature map.
        """
        for modality_name in self.modality_name_list:
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
        
        output_dict.update({"fc_before_send": deep_clone_dict(modality_feature_dict)})
        
        
        """Feature Send, local to pub space"""
        num_allied = len(self.allied_modality_list)
        self.num_allied = num_allied 
        
        pub_modality_feature_dict = {}
        # pub_feature_list = []
        consensus_feature_list = []
        pub_feature_list = []
        for modality_name in self.modality_name_list:
            feature = modality_feature_dict[modality_name]
            pub_feature = eval(f"self.comm_{modality_name}.sender")(feature)
            pub_modality_feature_dict[modality_name] = pub_feature
            pub_feature_list.append(pub_feature)
            if num_allied > 0:    # 参与训练的智能体中有已allied的智能体, 公共表征为allied模态产生表征的平均值
                # consensus_feature_list.append(pub_feature)
                if modality_name in self.allied_modality_list:
                    consensus_feature_list.append(pub_feature)
                # 新代理加入时, 同样使用negotiator辅助训练
                # if modality_name in self.newtype_modality_list:
                #     consensus_feature_list.append(eval(f"self.nego_reziser_{modality_name}")\
                #         (feature, pub_feature[0].size()))    

            if num_allied == 0:   # 参与训练的智能体中全部都没有allied, 通过negotiator计算公共表征
                if modality_name in self.newtype_modality_list: # 将初始特征对齐到标准尺寸
                    consensus_feature_list.append(eval(f"self.nego_reziser_{modality_name}")\
                        (feature, pub_feature[0].size()))     
        
        
        if num_allied > 0: # 所有allied智能体标准标准的平均值作为公式
            consensus_feature = torch.mean(torch.stack(consensus_feature_list), dim=0)
            
            # if self.pc_grad: # 新代理加入时, 同样使用negotiator辅助训练
            #     # 物理推理两份consen feat, 分别输入对应模态的receiver, 用以梯度手术
            #     consensus_feature_for_mode_dict = {}
            #     for modality_name in self.modality_name_list:
            #         consensus_feature_for_mode_dict[modality_name] = \
            #             eval(f'self.negotiator_bak_{modality_name}')(consensus_feature_list)
            #     consensus_feature = torch.mean(torch.stack(list(consensus_feature_for_mode_dict.values())), dim=0)
            # else:
            #     consensus_feature = self.negotiator(consensus_feature_list)
            
        if num_allied == 0: # 通过negotiator协商标准表征
            # if self.pc_grad and self.stage == 'nego':
            if self.pc_grad:
                if self.stage == 'nego':
                # 物理推理两份consen feat, 分别输入对应模态的receiver, 用以梯度手术
                    consensus_feature_for_mode_dict = {}
                    for modality_name in self.modality_name_list:
                        consensus_feature_for_mode_dict[modality_name] = \
                            eval(f'self.negotiator_bak_{modality_name}')(consensus_feature_list)
                    consensus_feature = torch.mean(torch.stack(list(consensus_feature_for_mode_dict.values())), dim=0)
                    # feature_show(consensus_feature[0], './analysis/ori.png')
                    # feature_show(consensus_feature_list[0][0], './analysis/ori_list.png')
                elif self.stage == 'unify':
                    consensus_feature =  eval(f'self.negotiator_bak_{modality_name}')(consensus_feature_list)
                    # feature_show(consensus_feature[0], './analysis/unify.png')
                    # feature_show(consensus_feature_list[0][0], './analysis/unify_list.png')
                # feature_show(consensus_feature_for_mode_dict['m3'][0], './analysis/ori_m3.png')
                # feature_show(consensus_feature_for_mode_dict['m4'][0], './analysis/ori_m4.png')
            else:
                # for p, bak_p in zip(self.negotiator.parameters(), \
                #     eval(f'self.negotiator_bak_m3.parameters()')): 
                #     print(torch.equal(p.data, bak_p.data))
                consensus_feature = self.negotiator(consensus_feature_list)
                
                # feature_show(consensus_feature[0], './analysis/nego.png')
                
                # consensus_feature = self.negotiator_bak_m3(consensus_feature_list)
                # feature_show(consensus_feature[0], './analysis/nego_bak_m3.png')
                # consensus_feature = eval(f'self.negotiator_bak_m4')(consensus_feature_list)
                # feature_show(consensus_feature[0], './analysis/nego_bak_m4.png')
                
        
        output_dict.update({"avg_pub_feature": consensus_feature,
                            "pub_modality_feature_dict": deep_clone_dict(pub_modality_feature_dict)})
        

        avg_pub_feature = torch.mean(torch.stack(pub_feature_list), dim=0)
        # avg_pub_feature = consensus_feature
        """Feature Receive, pub to local space"""
        for modality_name in self.modality_name_list:            
            if num_allied == 0:
                if self.pc_grad:
                    if self.stage == 'nego':
                        receiver_feature = consensus_feature_for_mode_dict[modality_name]
                        # receiver_feature = torch.stack((consensus_feature_for_mode_dict[modality_name], pub_modality_feature_dict[modality_name]), dim=0)
                        # receiver_feature = receiver_feature.mean(dim=0)
                    elif self.stage == 'unify':
                        receiver_feature = torch.stack((consensus_feature_for_mode_dict[modality_name], pub_modality_feature_dict[modality_name]), dim=0)
                        receiver_feature = receiver_feature.mean(dim=0)
                        
                    modality_feature_dict[modality_name] = eval(f"self.comm_{modality_name}.receiver")\
                        (receiver_feature, record_len, affine_matrix)
                else:
                    receiver_feature = torch.stack((consensus_feature, pub_modality_feature_dict[modality_name]), dim=0)
                    receiver_feature = receiver_feature.mean(dim=0)
                    modality_feature_dict[modality_name] = eval(f"self.comm_{modality_name}.receiver")\
                    (receiver_feature, record_len, affine_matrix)
                    
            elif num_allied > 0:
                modality_feature_dict[modality_name] = eval(f"self.comm_{modality_name}.receiver")\
                (avg_pub_feature, record_len, affine_matrix)
            
        output_dict.update({"fc_after_receive": deep_clone_dict(modality_feature_dict)})
        
        fc_af_recombine_send = {}
        fc_bf_recombine_receive = {}
        for modality_name in self.newtype_modality_list:
            fc_af_recombine_send[modality_name] = eval(f"self.comm_{modality_name}.fc_af_recombine_send")
            fc_bf_recombine_receive[modality_name] = eval(f"self.comm_{modality_name}.fc_bf_recombine_receive")
        
        output_dict.update({"fc_af_recombine_send": fc_af_recombine_send,
                            "fc_bf_recombine_receive": fc_bf_recombine_receive
                            })
        
        # import os
        # save_dir = "analysis/vis_feats"
        # fc_before_send = output_dict['fc_before_send']
        # fc_after_receive = output_dict['fc_after_receive']
        # pub_modality_feat = output_dict["pub_modality_feature_dict"]
        
        # # avg_pub_feature = output_dict["avg_pub_feature"]
        # # feature_show(avg_pub_feature[0], os.path.join(save_dir, "consensus"))
        # for modality in self.newtype_modality_list:
        #     feature_show(fc_before_send[modality][0], os.path.join(save_dir, f"{modality}_bf_send"))
        #     feature_show(fc_after_receive[modality][0], os.path.join(save_dir, f"{modality}_af_receive"))
        #     feature_show(pub_modality_feat[modality][0], os.path.join(save_dir, f"{modality}_pub"))
        # analysis/vis_feats
        


        """适应下游任务时, 分别将公共表征、本地表征接入检测头, 得到监督信号"""
        # if self.stage == 'ft':
        #     output_dict = {
        #         "avg_pub_feature": output_dict['avg_pub_feature'],
        #         "pub_modality_feature_dict": output_dict['pub_modality_feature_dict'],
        #         'newtype_modality_list': self.newtype_modality_list}   

        #     for modality_name in self.modality_name_list:
        #         modality_preds_dict = {}
        #         modality_feature = modality_feature_dict[modality_name]
                
        #         """Fuse and output""" 
        #         fusion_method = eval(f"self.fusion_method_{modality_name}")
                
        #         if fusion_method == 'pyramid':
        #             modality_feature, occ_outputs = eval(f"self.fusion_net_{modality_name}.forward_single")(modality_feature)
        #         else:
        #             # 设置为每个batch只有一辆车, 以适应fusion_net的输入, 这样等价于分别用每个智能体执行推理融合
        #             B = modality_feature.shape[0]
        #             L = affine_matrix.shape[1]
        #             record_len = [1] * B
        #             affine_matrix = torch.ones((B, L, L, 2, 3)).to(modality_feature.device)
        #             modality_feature = eval(f'self.fusion_net_{modality_name}')(modality_feature, record_len, affine_matrix)

        #         if fusion_method == 'pyramid' and eval(f"self.shrink_flag_{modality_name}"):
        #             modality_feature = eval(f"self.shrink_conv_{modality_name}")(modality_feature)

        #         modality_preds_dict['cls_preds_single'] = eval(f"self.cls_head_{modality_name}")(modality_feature)
        #         modality_preds_dict['reg_preds_single'] = eval(f"self.reg_head_{modality_name}")(modality_feature)
        #         modality_preds_dict['dir_preds_single'] = eval(f"self.dir_head_{modality_name}")(modality_feature)
                
        #         output_dict[f'preds_{modality_name}'] = modality_preds_dict



        return output_dict
