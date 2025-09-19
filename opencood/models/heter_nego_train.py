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
from opencood.models.fuse_modules import build_fusion_net
from opencood.models.sub_modules.base_bev_backbone_resnet import ResNetBEVBackbone 
from opencood.models.comm_modules.comm_in_pub import ComminPub
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

class HeterNegoTrain(nn.Module):
    def __init__(self, args):
        super(HeterNegoTrain, self).__init__()
       
        self.modality_name_list = list(set(args['modality_setting'].keys()))

        self.cav_range = args['lidar_range']
        self.sensor_type_dict = OrderedDict()

        self.cam_crop_info = {} 
        
        self.allied = {}
        
        self.allied_num = 0
        
        self.allied_modality_list = []

        self.stage = args['stage'] if 'stage' in args.keys() else 'inf'

        self.pc_grad = args['pub']['pc_grad'] if 'pc_grad' in args['pub'].keys() else False


        self.supervise_single = False
        if args.get("supervise_single", False):
            self.supervise_single = True
        
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
                                            #    'comm_space': args['comm_space'],
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
            Heads
            """
            setattr(self, f"cls_head_{modality_name}", nn.Conv2d(model_setting['in_head'], model_setting['anchor_number'],
                                    kernel_size=1))
            setattr(self, f"reg_head_{modality_name}", nn.Conv2d(model_setting['in_head'], 7 * model_setting['anchor_number'],
                                    kernel_size=1))            
            setattr(self, f"dir_head_{modality_name}", nn.Conv2d(model_setting['in_head'], \
                            model_setting['dir_args']['num_bins'] * model_setting['anchor_number'],
                                    kernel_size=1))

            # setattr(self, f"occ_head_single_{modality_name}", nn.Conv2d(model_setting['in_head'], 1,
            #                         kernel_size=1))
            
            setattr(self, f"nego_reziser_{modality_name}", ResizeNet(model_setting['local_dim'], args['pub']['C_uni']))
            
            """梯度手术方法时, 通过为每个模态备份单独的negotiator获得关于模态重构损失的专属梯度, 以此权衡各模态信息"""
            if self.pc_grad:
                setattr(self, f"negotiator_bak_{modality_name}",  Negotiator(args['pub']['negotiator'])) # nego: 物理上的两个用于产生不同梯度, 逻辑上的一个
                
            
            """
            Single Head
            """
            if self.supervise_single and fusion_name != 'pyramid': # pyramid的single监督器直接包含在融合网络pyramid中, 无需重新定义
                setattr(self, f"cls_head_single_{modality_name}", nn.Conv2d(model_setting['in_head_sem'], model_setting['anchor_number'],
                                        kernel_size=1))
                setattr(self, f"reg_head_single_{modality_name}", nn.Conv2d(model_setting['in_head_sem'], 7 * model_setting['anchor_number'],
                                        kernel_size=1))            
                setattr(self, f"dir_head_single_{modality_name}", nn.Conv2d(model_setting['in_head_sem'], \
                                model_setting['dir_args']['num_bins'] * model_setting['anchor_number'],
                                        kernel_size=1))
            
        args_pub = args['pub']
        # if (self.allied_num) == 0 and (not self.pc_grad):
        # if not self.pc_grad:
        self.negotiator = Negotiator(args_pub['negotiator']) if 'negotiator' in args_pub.keys() else None
        
        # self.fusion_net_nego = build_fusion_net(args_pub['fusion_net'])
        self.fusion_method_nego = args_pub['fusion_net']['method']

        # self.cls_head_nego = nn.Conv2d(args_pub['in_head'], args_pub['anchor_number'],
        #                             kernel_size=1)
        # self.reg_head_nego = nn.Conv2d(args_pub['in_head'], 7 * args_pub['anchor_number'],
        #                             kernel_size=1)
        # self.dir_head_nego = nn.Conv2d(args_pub['in_head'], args_pub['dir_args']['num_bins'] * args_pub['anchor_number'],
        #                             kernel_size=1)

        self.occ_head_nego = nn.Conv2d(args_pub['in_head'], 1, kernel_size=1)

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

        self.newtype_modality_list = list(set(self.modality_name_list) - set(self.allied_modality_list))

        """ nego阶段, 仅更新未加入联盟的comm模块的参数 """
        if self.stage == 'nego':
            # 已加入联盟的智能体模态类别
            for modality_name in self.modality_name_list:
                if not self.allied[modality_name]:
                    self.set_mode_comm_trainable(modality_name)
                    # print(f"New agent type: {modality_name}")            
            
            if self.allied_num == 0:
                # 训练协商器阶段, 全为未在联盟内的代理时, 初始化各模态备份的协商器参数为相同值
                self.set_nego_trainable()   # 这个会影响inference的feature的可视化效果！
                if self.pc_grad:
                    self._negotiator_init(nego_ref_mode = self.newtype_modality_list[0])
        
        elif self.stage == 'ft': 
            """下游任务微调阶段, 仅训练接收器"""
            for modality_name in self.modality_name_list:
                if not self.allied[modality_name]:
                    self.set_mode_receiver_trainable(modality_name)
        else:
            AssertionError('Only nego and ft stage')

        check_trainable_module(self)

    def set_mode_receiver_trainable(self, modality_name):
        for name, p in self.named_parameters():
            if (modality_name in name) and \
                (('receiver' in name) or ('local_prompt_generator' in name)
                 or ('unify2local' in name)):
                p.requires_grad = True

        for name, module in self.named_modules():
            if (modality_name in name) and \
                (('receiver' in name) or ('local_prompt_generator' in name)
                 or ('unify2local' in name)):
                 module.train()
    
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
        # 各模态对应的negotiator备份初始化为相同的参数
        for modality_name in self.newtype_modality_list:
            for para_sign, para_mode in zip(eval(f'self.negotiator_bak_{nego_ref_mode}.parameters()'), \
                                            eval(f'self.negotiator_bak_{modality_name}.parameters()')):
                para_mode.data.copy_(para_sign.data)
                
                # 若已有联盟内代理, 不再更新negotiator参数
                if len(self.allied_modality_list) > 0 or self.stage == 'ft':
                    para_mode.requires_grad = False  
    
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
        
        output_dict = {'modality_name_list': self.modality_name_list, 
                       'newtype_modality_list': self.newtype_modality_list}        
        
        # modality_type_list = data_dict['modality_type_list'] 
        agent_modality_list = data_dict['agent_modality_list'] 
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
        
        modality_feat_common_dict = {}        
        modality_feat_common_list = []
        for modality_name in self.modality_name_list:       
                 
            modality_feat_common = eval(f"self.comm_{modality_name}.send")(modality_feature_dict[modality_name])

            modality_feat_common_list.append(modality_feat_common)
            modality_feat_common_dict[modality_name] = modality_feat_common

        stage = data_dict['stage']
        
        if stage == 'ft':
            """Compute det loss when heterogeneous cavs collaborate"""
            agent_modality_list = data_dict['agent_modality_list'] 
            ego_modality = agent_modality_list[0]
            fusion_method = eval(f"self.fusion_method_{ego_modality}")
            
            modality_fusion_name = {}
            modality_fusion_name[ego_modality] = eval(f"self.fusion_method_{ego_modality}")                
            modality_preds = {}

            received_feature = []
            for cav_id in range(len(agent_modality_list)):
                modality_name = agent_modality_list[cav_id]
                received_feature.append(modality_feat_common_dict[modality_name][cav_id])
                # if modality_name == ego_modality:
                    # received_feature.append(modality_feat_common_dict[modality_name][cav_id])
                # else:
                    # received_feature.append(common_rep[cav_id])

            received_feature = torch.stack(received_feature)
            
            restored_feature = eval(f'self.comm_{ego_modality}.receive')(received_feature, record_len, affine_matrix)
            # """Fuse and output""" 
            if fusion_method == 'pyramid':
                fused_feature, occ_outputs = eval(f"self.fusion_net_{ego_modality}")\
                                                    (restored_feature, record_len, affine_matrix)       
            else:
                fused_feature = eval(f'self.fusion_net_{ego_modality}')\
                                        (restored_feature, record_len, affine_matrix)
                occ_outputs = None            
            # print(ego_modality)
            cls_preds = eval(f"self.cls_head_{ego_modality}")(fused_feature)
            reg_preds = eval(f"self.reg_head_{ego_modality}")(fused_feature)
            dir_preds = eval(f"self.dir_head_{ego_modality}")(fused_feature)

            # cls_preds_single = eval(f"self.cls_head_single_{ego_modality}")(restored_feature)
            # reg_preds_single = eval(f"self.reg_head_single_{ego_modality}")(restored_feature)
            # dir_preds_single = eval(f"self.dir_head_single_{ego_modality}")(restored_feature)
            
            # feature_show(cls_preds[0], '/home/scz/HEAL/preds_show/cls.png')        
            
            # feature_show(cls_preds_single[0], '/home/scz/HEAL/preds_show/cls_single1.png')
            # feature_show(cls_preds_single[1], '/home/scz/HEAL/preds_show/cls_single2.png')
            # feature_show(cls_preds_single[2], '/home/scz/HEAL/preds_show/cls_single3.png')
            
            # modality_preds_single.update({
            #                     ego_modality:
            #                         {'cls_preds_single': cls_preds_single,
            #                         'reg_preds_single': reg_preds_single,
            #                         'dir_preds_single': dir_preds_single,
            #                         'pyramid': 'single'}
            #                     })
            
            modality_preds.update({
                                ego_modality:
                                    {'cls_preds': cls_preds,
                                    'reg_preds': reg_preds,
                                    'dir_preds': dir_preds,
                                    'pyramid': 'collab',
                                    'occ_single_list': occ_outputs}
                                })
            
            output_dict.update({
                                # 'modality_preds_single': modality_preds_single,
                                'modality_preds': modality_preds,
                                'modality_fusion_name': modality_fusion_name,
                                })


            """ Compute det loss of all cavs """
            
            # standard_size = modality_feat_common[0].size()
            # standard_feature_list = []

            # for modality_name in self.modality_name_list:
            #     if num_allied > 0:    # 参与训练的智能体中有已allied的智能体, 公共表征为allied模态产生表征的平均值
            #         modality_feat_common = modality_feat_common_dict[modality_name]
            #         if modality_name not in self.newtype_modality_list:
            #             standard_feature_list.append(modality_feat_common)

            #     elif num_allied == 0:   # 参与训练的智能体中全部都没有allied, 通过negotiator计算公共表征
            #         feature = modality_feature_dict[modality_name]
            #         standard_feature_list.append(eval(f"self.nego_reziser_{modality_name}")\
            #             (feature, standard_size))     
            
            # # 从标准表示的特征协商公共表征
            # if num_allied > 0: # 所有allied智能体标准特征的平均值作为公共表征
            #     """是否考虑联盟内代理与新代理协商, 得到新旧联合的公共表征？"""
            #     common_rep = torch.mean(torch.stack(standard_feature_list), dim=0)

            # elif num_allied == 0: # 通过negotiator协商标准表征
            #     common_rep = self.negotiator(standard_feature_list)

            # modality_fusion_name = {}
            # modality_preds = {}
            # # modality_preds_single = {}
            
            # for modality_name in self.modality_name_list:  
            #     modality_fusion_name[modality_name] = eval(f"self.fusion_method_{modality_name}")
            #     # heter_feature_2d_list = self.assemble_heter_features(agent_modality_list, modality_feature_dict)
            #     received_feature = torch.stack([common_rep, modality_feat_common_dict[modality_name]])
            #     received_feature = received_feature.mean(dim=0)

            #     received_feature = modality_feat_common_dict[modality_name]
            #     restored_feature = eval(f"self.comm_{modality_name}.receive")\
            #         (received_feature, record_len, affine_matrix)                 
                
            #     """Fuse and output""" 
            #     fusion_method = eval(f"self.fusion_method_{modality_name}")
                
            #     if fusion_method == 'pyramid':
            #         fused_feature, occ_outputs = eval(f"self.fusion_net_{modality_name}")(
            #                                                 restored_feature,
            #                                                 record_len, 
            #                                                 affine_matrix)       
            #     else:
            #         fused_feature = eval(f'self.fusion_net_{modality_name}')(restored_feature, record_len, affine_matrix)
            #         occ_outputs = None            

            #     cls_preds = eval(f"self.cls_head_{modality_name}")(fused_feature)
            #     reg_preds = eval(f"self.reg_head_{modality_name}")(fused_feature)
            #     dir_preds = eval(f"self.dir_head_{modality_name}")(fused_feature)

                # cls_preds_single = eval(f"self.cls_head_single_{modality_name}")(restored_feature)
                # reg_preds_single = eval(f"self.reg_head_single_{modality_name}")(restored_feature)
                # dir_preds_single = eval(f"self.dir_head_single_{modality_name}")(restored_feature)
            
                # modality_preds.update({
                #                     modality_name:
                #                     {'cls_preds': cls_preds,
                #                     'reg_preds': reg_preds,
                #                     'dir_preds': dir_preds,
                #                     'pyramid': 'collab',
                #                     'occ_single_list': occ_outputs}
                #                     })
                
                # modality_preds_single.update({
                #         modality_name:
                #             {'cls_preds_single': cls_preds_single,
                #             'reg_preds_single': reg_preds_single,
                #             'dir_preds_single': dir_preds_single,
                #             'pyramid': 'single'}
                #         })
            
            output_dict.update({
                        # 'modality_preds_single': modality_preds_single,
                        'modality_preds': modality_preds,
                        'modality_fusion_name': modality_fusion_name,
                        })

        elif stage == 'nego':
            """ 协商公共表征 """
            # 尺寸对齐到标准表示
            standard_size = modality_feat_common[0].size()
            standard_feature_list = []

            for modality_name in self.modality_name_list:
                feature = modality_feature_dict[modality_name]
                if num_allied > 0:    # 参与训练的智能体中有已allied的智能体, 公共表征为allied模态产生表征的平均值
                    modality_feat_common = modality_feat_common_dict[modality_name]
                    if modality_name not in self.newtype_modality_list:
                        standard_feature_list.append(modality_feat_common)

                elif num_allied == 0:   # 参与训练的智能体中全部都没有allied, 通过negotiator计算公共表征
                    standard_feature_list.append(eval(f"self.nego_reziser_{modality_name}")\
                        (feature, standard_size))     
            
            # 从标准表示的特征协商公共表征
            if num_allied > 0: # 所有allied智能体标准特征的平均值作为公共表征
                """是否考虑联盟内代理与新代理协商, 得到新旧联合的公共表征？"""
                common_rep = torch.mean(torch.stack(standard_feature_list), dim=0)

            elif num_allied == 0: # 通过negotiator协商标准表征
                # if self.pc_grad and self.stage == 'nego':
                if self.pc_grad:
                    # if self.stage == 'nego':
                    # 物理推理两份consen feat, 分别输入对应模态的receiver, 计算两个损失, 以此为梯度手术提供参考
                    mode_common_rep_dict = {}
                    for modality_name in self.modality_name_list:
                        mode_common_rep_dict[modality_name] = \
                            eval(f'self.negotiator_bak_{modality_name}')(standard_feature_list)
                    common_rep = torch.mean(torch.stack(list(mode_common_rep_dict.values())), dim=0)
                        # feature_show(consensus_feature[0], './analysis/ori.png')
                        # feature_show(consensus_feature_list[0][0], './analysis/ori_list.png')
                else:
                    # for p, bak_p in zip(self.negotiator.parameters(), \
                    #     eval(f'self.negotiator_bak_m3.parameters()')): 
                    #     print(torch.equal(p.data, bak_p.data))
                    common_rep = self.negotiator(standard_feature_list)
                    
                    # feature_show(consensus_feature[0], './analysis/nego.png')
                    
                    # consensus_feature = self.negotiator_bak_m3(consensus_feature_list)
                    # feature_show(consensus_feature[0], './analysis/nego_bak_m3.png')
                    # consensus_feature = eval(f'self.negotiator_bak_m4')(consensus_feature_list)
                    # feature_show(consensus_feature[0], './analysis/nego_bak_m4.png')
            output_dict.update({"common_rep": common_rep,
                                "modality_feat_common_dict": deep_clone_dict(modality_feat_common_dict)})
                            

            preds_nego = self.unify_pragma_loss(common_rep, modality_feat_common_dict)
            
            output_dict.update({"preds_nego": preds_nego})
            
            
            """对齐阶段, 只训练sender"""
            # if self.stage == 'align':
            #     return output_dict



            """ Feature Receive, common space to local"""
            for modality_name in self.modality_name_list:            
                if num_allied == 0:
                    # if self.stage == 'nego':
                    if self.pc_grad:
                        received_feature = mode_common_rep_dict[modality_name]
                    else:
                        received_feature = common_rep

                    
                    modality_feature_dict[modality_name] = eval(f"self.comm_{modality_name}.receive")\
                    (received_feature, record_len, affine_matrix)
                        
                elif num_allied > 0:
                    common_rep = torch.mean(torch.stack([common_rep, modality_feat_common_dict[modality_name]]), dim=0)
                    modality_feature_dict[modality_name] = eval(f"self.comm_{modality_name}.receive")\
                    (common_rep, record_len, affine_matrix)


            output_dict.update({"fc_after_receive": deep_clone_dict(modality_feature_dict)})
            
        
        # Find ego idxs
        # idx = 0
        # ego_idxs = [idx]
        # # ego_mode = agent_modality_list[idx]
        # for batch_n_cavs in record_len[:-1]:
        #     idx = idx + batch_n_cavs
        #     ego_idxs.append(idx)

        
        
                 
        

        """Only compute det loss of ego"""
        # agent_modality_list = data_dict['agent_modality_list'] 
        # ego_modality = agent_modality_list[0]
        # fusion_method = eval(f"self.fusion_method_{ego_modality}")
        
        # modality_fusion_name = {}
        # modality_fusion_name[ego_modality] = eval(f"self.fusion_method_{ego_modality}")                
        # modality_preds = {}
        # # modality_preds_single = {}

        # restored_feature = output_dict["fc_after_receive"][ego_modality]

        # """Fuse and output""" 
        # if fusion_method == 'pyramid':
        #     fused_feature, occ_outputs = eval(f"self.fusion_net_{ego_modality}")\
        #                                         (restored_feature, record_len, affine_matrix)       
        # else:
        #     fused_feature = eval(f'self.fusion_net_{ego_modality}')\
        #                             (restored_feature, record_len, affine_matrix)
        #     occ_outputs = None            
        # # print(ego_modality)
        # cls_preds = eval(f"self.cls_head_{ego_modality}")(fused_feature)
        # reg_preds = eval(f"self.reg_head_{ego_modality}")(fused_feature)
        # dir_preds = eval(f"self.dir_head_{ego_modality}")(fused_feature)

        # # cls_preds_single = eval(f"self.cls_head_single_{ego_modality}")(restored_feature)
        # # reg_preds_single = eval(f"self.reg_head_single_{ego_modality}")(restored_feature)
        # # dir_preds_single = eval(f"self.dir_head_single_{ego_modality}")(restored_feature)
        
        # # feature_show(cls_preds[0], '/home/scz/HEAL/preds_show/cls.png')        
        
        # # feature_show(cls_preds_single[0], '/home/scz/HEAL/preds_show/cls_single1.png')
        # # feature_show(cls_preds_single[1], '/home/scz/HEAL/preds_show/cls_single2.png')
        # # feature_show(cls_preds_single[2], '/home/scz/HEAL/preds_show/cls_single3.png')
        
        # # modality_preds_single.update({
        # #                     ego_modality:
        # #                         {'cls_preds_single': cls_preds_single,
        # #                         'reg_preds_single': reg_preds_single,
        # #                         'dir_preds_single': dir_preds_single,
        # #                         'pyramid': 'single'}
        # #                     })
        
        # modality_preds.update({
        #                     ego_modality:
        #                         {'cls_preds': cls_preds,
        #                         'reg_preds': reg_preds,
        #                         'dir_preds': dir_preds,
        #                         'pyramid': 'collab',
        #                         'occ_single_list': occ_outputs}
        #                     })
        
        # output_dict.update({
        #                     # 'modality_preds_single': modality_preds_single,
        #                     'modality_preds': modality_preds,
        #                     'modality_fusion_name': modality_fusion_name,
        #                     })
        

                
        # for b in range(cls_preds.shape[0]):
        #     feature_show(cls_preds[b], f'preds_show/view{b}_cls_{ego_modality}')
        #     feature_show(reg_preds[b], f'preds_show/view{b}_reg_{ego_modality}')
        #     feature_show(dir_preds[b], f'preds_show/view{b}_dir_{ego_modality}')
        # fc_af_recombine_send = {}
        # fc_bf_recombine_receive = {}
        # for modality_name in self.newtype_modality_list:
        #     fc_af_recombine_send[modality_name] = eval(f"self.comm_{modality_name}.fc_af_recombine_send")
        #     fc_bf_recombine_receive[modality_name] = eval(f"self.comm_{modality_name}.fc_bf_recombine_receive")
        
        # output_dict.update({"fc_af_recombine_send": fc_af_recombine_send,
        #                     "fc_bf_recombine_receive": fc_bf_recombine_receive
        #                     })

        return output_dict
    

    def unify_pragma_loss(self, common_rep, modality_feat_common_dict):
        """
        通过二维占用预测任务,
        使协商器得到的公共表征和各模态发送器转换得到的公共表征, 语用空间一致
        """


        preds_nego = dict()
        fusion_name_nego = self.fusion_method_nego
        preds_nego['fusion_name'] = fusion_name_nego

        occ_preds_nego = self.occ_head_nego(common_rep)

        preds_nego['nego'] = {"occ_preds_single": occ_preds_nego,}
        

        # for b in range(cls_preds_nego.shape[0]):
        #     feature_show(cls_preds_nego[b], f'preds_show/view{b}_cls_nego_common')
        #     feature_show(reg_preds_nego[b], f'preds_show/view{b}_reg_nego_common')
        #     feature_show(dir_preds_nego[b], f'preds_show/view{b}_dir_nego_common')
        
        for modality_name in self.modality_name_list:
            modality_feat_common = modality_feat_common_dict[modality_name]

            occ_preds_modality = self.occ_head_nego(modality_feat_common)

            preds_nego[modality_name] = {
                                    "occ_preds_single": occ_preds_modality}
            
            # for b in range(cls_preds_nego.shape[0]):
            #     feature_show(cls_preds_nego[b], f'preds_show/view{b}_cls_nego_{modality_name}')
            #     feature_show(reg_preds_nego[b], f'preds_show/view{b}_reg_nego_{modality_name}')
            #     feature_show(dir_preds_nego[b], f'preds_show/view{b}_dir_nego_{modality_name}')

        return preds_nego