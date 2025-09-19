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
from opencood.models.comm_modules.comm_in_pub_decom import ComminPub
# from opencood.models.sub_modules.comm_in_local import ComminLocal
from opencood.models.comm_modules.comm_in_pub_vanilla import ComminPubVanilla
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

class HeterNego(nn.Module):
    def __init__(self, args):
        super(HeterNego, self).__init__()
        self.args = args
        # modality_name_list = list(args.keys())
        # modality_name_list = [x for x in modality_name_list if x.startswith("m") and x[1:].isdigit()] 
        # self.modality_name_list = modality_name_list
        
        self.modality_name_list = list(set(args['mapping_dict'].values()))

        self.cav_range = args['lidar_range']
        self.sensor_type_dict = OrderedDict()

        self.cam_crop_info = {} 
        
        self.fixed = {}

        self.cosim_the = 0.0
        self.cosim_the_sup = args['pub']['sup_cosim_threhold'] 
        self.cosim_the_cor = args['pub']['cor_cosim_threhold'] 
        self.the_change_epoch = args['pub']['the_change_epoch']
        
        self.pub_codebook = None
        self.init_pub_codebook = None
        self.use_alliance = args['pub']['use_alliance']
        self.pub_cb_path = args['pub']['cb_path']
        
        # self.convert_phase = 'bf_decom'
        # if 'convert_phase' in args.keys() and args['convert_phase'] == 'af_decom':
        #     self.convert_phase = 'af_decom'
        
        # self.max_codes = 32
        self.pub_query_emb_path = args['pub']['query_emb_path']
        self.pub_query_embeddings = None
        
        self.local_pub_change = False
        if 'local_pub_change' in args.keys():
            self.local_pub_change = args['local_pub_change']

        if args['mode_cb_same_init']:
            """为不同模态的codebook初始化相同的参数"""
            max_code_num = 0
            for modality_name in self.modality_name_list:
                code_num = args[modality_name]['comm_args']['num_codes']
                if code_num > max_code_num:
                    max_code_num = code_num
            dim_code = args['pub']['C_uni']
            init_codebook = nn.Parameter(torch.randn(max_code_num, dim_code))
        else:
            init_codebook = None  
            
        
        # setup each modality model
        for modality_name in self.modality_name_list:
            model_setting = args[modality_name]
            sensor_name = model_setting['sensor_type']
            self.sensor_type_dict[modality_name] = sensor_name
            
            """
            若在联盟中, 固定comm模块参数; 若不在联盟中, 调整comm模块参数
            """
            self.fixed[modality_name] = model_setting['allied']

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
                                               'fixed':self.fixed[modality_name],
                                               'comm_space': args['comm_space'],
                                               'modality': modality_name
                                            #    'convert_phase': self.convert_phase
                                            #    'use_alliance': self.
                                               })
            setattr(self, f"comm_{modality_name}", ComminPub(model_setting['comm_args'], self.pub_codebook, init_codebook))
            # setattr(self, f"comm_{modality_name}", ComminPubVanilla(model_setting['comm_args'], self.pub_codebook, init_codebook))
            
            
            """
            Fusion, by default multiscale fusion: 
            Note the input of PyramidFusion has downsampled 2x. (SECOND required)
            """
            setattr(self, f"fusion_net_{modality_name}", build_fusion_net(model_setting['fusion_net']))
            setattr(self, f"fusion_method_{modality_name}", model_setting['fusion_net']['method'])


            """
            Shrink header
            """
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
    
    
        """设置仅更新comm模块的参数"""    
        for name, p in self.named_parameters():
            p.requires_grad = False
        for name, module in self.named_modules():
            module.eval()
    
        # 已加入联盟的智能体模态类别
        self.allied_modality_list = []
        for modality_name in self.modality_name_list:
            if not self.fixed[modality_name]:
                self.set_trainable(modality_name)
                print(f"New agent type: {modality_name}")
            else:
                # 不固定参数, 即已加入联盟, 记录下来
                self.allied_modality_list.append(modality_name)
                print(f"Allied agent type: {modality_name}")
                
        self.newtype_modality_list = list(set(self.modality_name_list) - set(self.allied_modality_list))
        # check again which module is not fixed.
        check_trainable_module(self)

        if args['comm_same_init']:
            self.comm_same_init()

    def comm_same_init(self):
        # 神经网络初始化为相同的参数
        
        sign_m = self.newtype_modality_list[0]

        for modality_name in self.newtype_modality_list:
            for para_sign, para_mode in zip(eval(f'self.comm_{sign_m}.parameters()'), \
                                            eval(f'self.comm_{modality_name}.parameters()')):
                para_mode.data.copy_(para_sign.data)
        # for modality_name in self.newtype_modality_list:
        #     for para_sign, para_mode in zip(eval(f'self.comm_{sign_m}.parameters()'), \
        #                                     eval(f'self.comm_{modality_name}.parameters()')):
        #         # print(torch.equal(para_sign, para_mode))

        
    def set_trainable(self, modality_name):
        for name, p in self.named_parameters():
            if f'comm_{modality_name}' in name:
                p.requires_grad = True
        for name, module in self.named_modules():
            if f'comm_{modality_name}' in name:
                 module.train()
           

    def forward(self, data_dict):
        output_dict = {'pyramid': 'single', 'modality_name_list': self.modality_name_list, 
                       'fixed_mode':self.fixed, 'newtype_modality_list': self.newtype_modality_list}
        
        """Setup public code and find mapping dict from local to pub"""
        if data_dict['epoch'] <= self.the_change_epoch:
            self.cosim_the = self.cosim_the_sup
        else: # 公共向量初步训练完成后, 启用更严格的限制, 使receiver的补全能力聚焦于测试中, 缺失的维度
            self.cosim_the = self.cosim_the_cor
        self.setup_pubcb_local2pub()
        output_dict.update({"pub_codebook": self.pub_codebook, "local2pub_dict":self.local2pub_dict})
        
        codebook_dict = {}
        for modality_name in self.modality_name_list:
            eval(f'self.comm_{modality_name}.pub_change')(self.local2pub_dict[modality_name], \
                                                    self.pub2local_dict[modality_name], self.pub_codebook)
            
            if not self.fixed[modality_name]:
                codebook_dict[modality_name] = eval(f'self.comm_{modality_name}.codebook')
        
        output_dict.update({"codebook_dict": codebook_dict})           
        
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
        
        # for mode in self.modality_name_list:
        #     feature_show(modality_feature_dict[modality_name][0], f'analysis/direct_unify/fc_before_send_{mode}', type='dim', dim=1)
        
        # add_clone_modality_dict(output_dict, "fc_before_send", modality_feature_dict)                
        output_dict.update({"fc_before_send": modality_feature_dict})
        
        
        # mode_pub_weight_dict = {}
        # fc_before_rbdc = {}
        # fc_after_rbdc = {}
        # keyfeat_bf_align = {}
        # keyfeat_af_align = {}
        # for modality_name in self.modality_name_list:
        #     mode_pub_weight_dict[modality_name] = \
        #       eval(f"self.comm_{modality_name}.sender")(modality_feature_dict[modality_name])
            
        #     keyfeat_bf_align.update({modality_name: eval(f"self.comm_{modality_name}.keyfeat_bf_align")})
        #     keyfeat_af_align.update({modality_name: eval(f"self.comm_{modality_name}.keyfeat_af_align")})
        #     if not self.fixed[modality_name]:
        #         fc_before_rbdc.update({modality_name: eval(f"self.comm_{modality_name}.fc_before_rbdc")})    
        #         fc_after_rbdc.update({modality_name: eval(f"self.comm_{modality_name}.fc_after_rbdc")})
                
                
        pub_keyfeat_for_negotiate_dict = {} # keyfeat before align2 in pub channel (idx sort by pub order)
        # keyfeat_for_negotiate = {}
        pub_keyfeat_for_negotiate_list = []
        for modality_name in self.modality_name_list:
            pub_keyfeat_for_negotiate_dict[modality_name] = \
                eval(f"self.comm_{modality_name}.sender")(modality_feature_dict[modality_name])
            
    
            # keyfeat_for_negotiate.update({modality_name: eval(f"self.comm_{modality_name}.keyfeat_for_negotiate")})
    
            if modality_name in self.allied_modality_list: # 若该模态已加入联盟, 使用所有维度的特征
                used_pub_keyfeat_for_negotiate = pub_keyfeat_for_negotiate_dict[modality_name]
            elif modality_name in self.newtype_modality_list: # 若该模态未加入联盟, 仅使用未allied维度的特征
                used_pub_keyfeat_for_negotiate = torch.zeros_like(pub_keyfeat_for_negotiate_dict[modality_name])
                used_pub_keyfeat_for_negotiate[:, self.pub_idx_noallied] = pub_keyfeat_for_negotiate_dict[modality_name][:, self.pub_idx_noallied]
            
            pub_keyfeat_for_negotiate_list.append(used_pub_keyfeat_for_negotiate)
                
        sum_pub_keyfeat_for_negotiate = torch.sum(torch.stack(pub_keyfeat_for_negotiate_list), dim=0)
        avg_pub_keyfeat_for_negotiate = sum_pub_keyfeat_for_negotiate / self.ref_allied_or_new_modes_per_pubcode.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

        # print('\n outputbatch:', weight_pub.shape[0])
        # output_dict.update({
        #     'avg_keyfeat_for_negotiate': avg_pub_keyfeat_for_negotiate,
        #     'keyfeat_for_negotiate': keyfeat_for_negotiate
        #     })


        mode_pub_weight_dict = {}
        pub_weight_list = []
        fc_before_rbdc = {}
        fc_after_rbdc = {}
        # keyfeat_bf_align = {}
        keyfeat_af_align = {}
        for modality_name in self.modality_name_list:
            mode_pub_weight_dict[modality_name] = \
              eval(f"self.comm_{modality_name}.sender_align")(avg_pub_keyfeat_for_negotiate)
            
            keyfeat_af_align.update({modality_name: eval(f"self.comm_{modality_name}.keyfeat_af_align")})
            if not self.fixed[modality_name]:
                # keyfeat_bf_align.update({modality_name: eval(f"self.comm_{modality_name}.keyfeat_bf_align")})
                fc_before_rbdc.update({modality_name: eval(f"self.comm_{modality_name}.fc_before_rbdc")})    
                fc_after_rbdc.update({modality_name: eval(f"self.comm_{modality_name}.fc_after_rbdc")})
            
            
            if modality_name in self.allied_modality_list: # 若该模态已加入联盟, 使用所有维度的特征
                used_mode_pub_weight = mode_pub_weight_dict[modality_name]
            elif modality_name in self.newtype_modality_list: # 若该模态未加入联盟, 仅使用未allied维度的特征
                used_mode_pub_weight = torch.zeros_like(mode_pub_weight_dict[modality_name])
                used_mode_pub_weight[:, self.pub_idx_noallied] = mode_pub_weight_dict[modality_name][:, self.pub_idx_noallied]
            
            pub_weight_list.append(used_mode_pub_weight)
    

        output_dict.update({
            "fc_before_rbdc": fc_before_rbdc,
            "fc_after_rbdc": fc_after_rbdc,
            # "keyfeat_bf_align": keyfeat_bf_align,
            "keyfeat_af_align": keyfeat_af_align,
            # "ref_num_modes_per_pubcode": self.ref_num_modes_per_pubcode,
            # "shared_weights_bf_trans": shared_weights_bf_trans,
            # "shared_weights_af_trans": shared_weights_af_trans,    
        })
            
            # ref_pub_idxs[modality_name] = (eval(f"self.comm_{modality_name}.ref_pub_idxs"))
          
        # 计算公共向量的平均值
        # weight_pub = torch.sum(torch.stack(list(mode_pub_weight_dict.values())), dim=0)
        # weight_pub = weight_pub / self.ref_num_modes_per_pubcode.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        
        # torch.max(weight_pub)
        
        # add_clone_modality_dict(output_dict, "mode_weight_dict", mode_pub_weight_dict)
        # output_dict.update({"weigh_pub": weight_pub.clone()})

        sum_pub_weight = torch.sum(torch.stack(pub_weight_list), dim=0)
        avg_pub_weight = sum_pub_weight / self.ref_allied_or_new_modes_per_pubcode.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)  
        
        # print('\n outputbatch:', weight_pub.shape[0])
        output_dict.update({'avg_pub_weight': avg_pub_weight})

        # shared_weights_bf_trans = {}
        # shared_weights_af_trans = {}
        fc_af_recombine_send = {}
        fc_bf_recombine_receive = {}
        receive_feature_dict = {}
        # keyfeat_bf_align_reverse = {}
        for modality_name in self.modality_name_list:
                # receive_feature_dict[modality_name] = \
                #     eval(f"self.comm_{modality_name}.receiver")(mode_pub_weight_dict[modality_name], record_len, affine_matrix)
                receive_feature_dict[modality_name] = \
                    eval(f"self.comm_{modality_name}.receiver")(avg_pub_weight, record_len, affine_matrix)
                
                
                # if data_dict['epoch'] <= self.the_change_epoch and self.local_pub_change: # 小于epoch12时, 输入本地权重, 训练基本重建能力
                #     receive_feature_dict[modality_name] = \
                #     eval(f"self.comm_{modality_name}.receiver")(mode_pub_weight_dict[modality_name], record_len, affine_matrix)
                # else: # 大于epoch12时, 语义锚点映射关系收敛, 输入平均权重, 训练对混合信息的重建能力
                #     receive_feature_dict[modality_name] = \
                #     eval(f"self.comm_{modality_name}.receiver")(avg_pub_weight, record_len, affine_matrix)
                if not self.fixed[modality_name]:
                    fc_af_recombine_send.update({modality_name: eval(f"self.comm_{modality_name}.fc_af_recombine_send")})
                    fc_bf_recombine_receive.update({modality_name: eval(f"self.comm_{modality_name}.fc_bf_recombine_receive")})
                    # keyfeat_bf_align_reverse.update({modality_name: eval(f"self.comm_{modality_name}.keyfeat_bf_align_reverse")})
                # mode_pub_weight和avg_pub_weight可以使用相同的发送器/映射器, 因为mode_pub_weight的码元数与pub_weight相同, sender/receiver会自动选择出本地的信息
            

                # shared_weights_bf_trans.update({modality_name: eval(f"self.comm_{modality_name}.shared_weights_bf_trans")})
                # shared_weights_af_trans.update({modality_name: eval(f"self.comm_{modality_name}.shared_weights_af_trans")})
        
        # for mode in self.modality_name_list:
        #     feature_show(modality_feature_dict[modality_name][0], f'analysis/direct_unify/fc_after_receive_{mode}', type='dim', dim=1)
        
        # add_clone_modality_dict(output_dict, "fc_after_receive", modality_feature_dict)
        output_dict.update({
                            "fc_bf_recombine_receive": fc_bf_recombine_receive,
                            "fc_af_recombine_send": fc_af_recombine_send,
                            "fc_after_receive":receive_feature_dict,
                            # "keyfeat_bf_align_reverse": keyfeat_bf_align_reverse
                            })
        
        return output_dict


    def setup_pubcb_local2pub(self):
        cosim_the = self.cosim_the
        modality_name_list = self.modality_name_list
        self.pub_codebook = self.init_pub_codebook
        codebook_dict = {}  # str: modality_name -> Tensor: codebook
        local2pub_dict = {} # str: modality_name -> dict: {(idx_local:idx_pub)}
        rescode_dict = {} # str: modality_name -> int num of codes
        for modality_name in modality_name_list:
            codebook_dict[modality_name] = eval(f'self.comm_{modality_name}.codebook').clone()
            rescode_dict[modality_name] = len(codebook_dict[modality_name])
            local2pub_dict[modality_name] = {}
        
        
        pub_codebook_list = []
        # 若存在公共码本, 首先将本地码本与公共码本匹配
        if self.pub_codebook is not None: 
            pub_codebook_list.append(self.pub_codebook)
            for modality_name in modality_name_list:
                if rescode_dict[modality_name] != 0:
                    norm_pub = F.normalize(pub_codebook_list[0], dim=1)
                    norm_mode = F.normalize(codebook_dict[modality_name], dim=1)
                    cosim = torch.mm(norm_mode, norm_pub.t())
                    
                    while rescode_dict[modality_name] > 0:
                        cosim_max_per_mode, idxs_pub = torch.max(cosim, dim=1)
                        cosim_max, idx_mode = torch.max(cosim_max_per_mode.unsqueeze(0), dim=1)
                        if cosim_max < cosim_the:
                            break
                        idx_pub = idxs_pub[idx_mode]
                        local2pub_dict[modality_name].update({idx_mode.item(): idx_pub.item()})
                        
                        rescode_dict[modality_name] = rescode_dict[modality_name] - 1
                        
                        # 一个本地基向量只匹配一个公共基向量
                        cosim[idx_mode,:] = -float('inf')
                        cosim[:,idx_pub] = -float('inf')
                        codebook_dict[modality_name][idx_mode] = 0 # 对应维度的code置为0, 表示已参与匹配
        
        # 剩余模态的向量自行匹配   
        total_code = []
        mode_idx_max = []
        cur = 0
        for modality_name in modality_name_list:
            # mode_idx_offset[modality_name] = cur
            total_code.append(codebook_dict[modality_name])
            cur = cur + len(codebook_dict[modality_name])
            mode_idx_max.append(cur)
        
        total_code = torch.cat(total_code, dim=0)     
        total_code = F.normalize(total_code, dim=1)
        cosim_mat = torch.mm(total_code, total_code.t())
        
        # 与自身模态的相关度为0
        cur = 0
        for i in range(len(modality_name_list)):
            len_mode = mode_idx_max[i]
            cosim_mat[cur: cur+len_mode, cur:cur+len_mode] = -float('inf')
            cur  = cur + len_mode 
        # pub_codebook的码元编号从其中已有码元开始计算
        pub_idx_max = len(pub_codebook_list[0]) if len(pub_codebook_list) > 0 else 0
        
        """循环寻找各模态中最接近的码元对, 直到所有大于阈值的码元对均被选出"""
        while torch.max(cosim_mat) > cosim_the:
            # 找到cosim最大的ego和neb对, 并保留ego的idx
            cosim_max_per_neb, idxs_neb = torch.max(cosim_mat, dim=1)
            _, idx_ego = torch.max(cosim_max_per_neb.unsqueeze(0), dim=1)
            
            idx_ego = idx_ego.item()
            
            # 从每个模态中寻找与ego_idx最接近向量的idx, 作为本轮匹配的码元
            matched_code_idx_list = [idx_ego]
            start_idx_mode = 0
            for mode_idx in range(len(modality_name_list)):
                _, neb_idx_mode = torch.max(cosim_mat[idx_ego, start_idx_mode: mode_idx_max[mode_idx]], dim=0)
                neb_idx = neb_idx_mode + start_idx_mode
                if cosim_mat[idx_ego, neb_idx] > cosim_the:
                    matched_code_idx_list.append(neb_idx)
                start_idx_mode = mode_idx_max[mode_idx]
                
        
            # 从匹配的码元计算pub_code, pub到local的映射字典, 将cosim_mat相应的位置置为负无穷, 不再参与匹配
            pub_code = []
            for code_idx in matched_code_idx_list:
                code_mode_idx = 0
                while True:
                    if code_idx < mode_idx_max[code_mode_idx]:
                        break
                    code_mode_idx = code_mode_idx + 1
                code_mode = modality_name_list[code_mode_idx]
                
                # code 在本地码本种的索引
                code_idx_in_local = code_idx if code_mode_idx ==0 else code_idx - mode_idx_max[code_mode_idx - 1]
                code_idx_in_local = int(code_idx_in_local)
                
                pub_code.append(codebook_dict[code_mode][code_idx_in_local].clone())
                local2pub_dict[code_mode].update({code_idx_in_local: pub_idx_max})
                    
                # 选择出的码元置为负无穷, 不再参与匹配
                cosim_mat[code_idx,:] = -float('inf')
                cosim_mat[:, code_idx] = -float('inf')
                codebook_dict[code_mode][code_idx_in_local] = 0
                rescode_dict[code_mode] = rescode_dict[code_mode] - 1
                
            # pub_code_list = torch.stack(pub_code_list)
            pub_code = F.normalize(torch.stack(pub_code), p=2, dim=1) # 正则化本地码元为单位基向量
            pub_code = torch.mean(pub_code, dim=0, keepdim=True) # 对单位基向量取平均值, 得到角平分线向量
            pub_code = F.normalize(pub_code, p=2, dim=1) # 角平分线向量进一步正则化为单位基向量
            # a = torch.sum(pub_code**2)
            pub_codebook_list.append(pub_code)

            pub_idx_max = pub_idx_max + 1
                    
        
        # 将剩余未匹配的基向量加入码本
        for mode in self.modality_name_list:
            if rescode_dict[mode] == 0:
                continue
            code_sum = torch.sum(codebook_dict[mode], dim=1)
            res_idxs = torch.where(code_sum != 0)[0]
            for idx in res_idxs:
                local2pub_dict[mode].update({idx.item():pub_idx_max})
                pub_idx_max = pub_idx_max + 1
            res_codes = codebook_dict[mode][res_idxs]
            res_codes = F.normalize(res_codes, p=2, dim=1)
            pub_codebook_list.append(res_codes)
        
        pub_codebook = torch.cat(pub_codebook_list, dim=0).detach()
        
        # assert pub_codebook.shape[0] == torch.abs
        self.pub_codebook = pub_codebook 
        self.local2pub_dict = local2pub_dict
        
        self.pub2local_dict = {}
        
        # 统计与每个公共码元关联的模态数, 为计算avg_pub_keyfeat做准备;
        # 若该码元及对应的特征已allied, 统计对应的allied的码元的数量
        # 若该码元及对应的特征未allied, 统计new_type的数量
        len_pub = pub_codebook.shape[0]
        ref_allied_or_new_modes_per_pubcode = torch.zeros(len_pub).to(pub_codebook.device)
        # ref_num_allied_modes_per_pubcode = torch.zeros(len_pub).to(pub_codebook.device)
        pub_idx_allied = []
        self.pub_ref_modes = {pub_id: [] for pub_id in range(len_pub)}

        for mode in self.modality_name_list:
            self.pub2local_dict[mode] = {}
            for local_idx, pub_idx in self.local2pub_dict[mode].items():
                self.pub2local_dict[mode].update({pub_idx: local_idx})
                self.pub_ref_modes[pub_idx].append(mode)
            
                if mode in self.allied_modality_list:
                    ref_allied_or_new_modes_per_pubcode[pub_idx] +=1
                    pub_idx_allied.append(pub_idx)
                elif mode in self.newtype_modality_list:
                    if pub_idx not in pub_idx_allied:
                        ref_allied_or_new_modes_per_pubcode[pub_idx] +=1
        
        self.ref_allied_or_new_modes_per_pubcode = ref_allied_or_new_modes_per_pubcode
        self.pub_idx_noallied = list(set([pub_idx for pub_idx in range(len_pub)]) - set(pub_idx_allied))
        # self.ref_num_modes_per_pubcode = torch.where(ref_num_modes_per_pubcode==0, 1e+3, ref_num_modes_per_pubcode)
        # self.ref_num_allied_modes_per_pubcode = torch.where(ref_num_allied_modes_per_pubcode==0, 1e+3, ref_num_allied_modes_per_pubcode)
        
        
        
        
        
        
       
        # len_pub = pub_codebook.shape[0]
        # ref_num_modes_per_pubcode = torch.zeros(len_pub).to(pub_codebook.device)
        # ref_num_allied_modes_per_pubcode = torch.zeros(len_pub).to(pub_codebook.device)
        # pub_idx_allied = []
        # self.pub_ref_modes = {pub_id: [] for pub_id in range(len_pub)}

        # for mode in self.modality_name_list:
        #     self.pub2local_dict[mode] = {}
        #     for local_idx, pub_idx in self.local2pub_dict[mode].items():
        #         self.pub2local_dict[mode].update({pub_idx: local_idx})
        #         self.pub_ref_modes[pub_idx].append(mode)
                
        #         ref_num_modes_per_pubcode[pub_idx] +=1

        #         # pub_idx分为allied或者no allied
        #         # 若已allied, 统计对应allied模态的数量; 
        #         # 若未allied, 统计对应new_type的模态的数量
        #         if mode in self.allied_modality_list:
        #             ref_num_allied_modes_per_pubcode[pub_idx] += 1
        #             pub_idx_allied.append(pub_idx)

        #         if (pub_idx not in pub_idx_allied) and (mode in self.newtype_modality_list):
        #             ref_num_allied_modes_per_pubcode[pub_idx] += 1
                
        # self.ref_num_modes_per_pubcode = torch.where(ref_num_modes_per_pubcode==0, 1e+3, ref_num_modes_per_pubcode)
        # self.ref_num_allied_modes_per_pubcode = torch.where(ref_num_allied_modes_per_pubcode==0, 1e+3, ref_num_allied_modes_per_pubcode)