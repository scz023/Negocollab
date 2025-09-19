# -*- coding: utf-8 -*-
# Author: Yifan Lu <yifan_lu@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib

import argparse
import os
import statistics
import threading
import time

import numpy as np
import torch

from opencood.tools.pcgrad import PCGrad
torch.autograd.set_detect_anomaly(True)
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tensorboardX import SummaryWriter
import tqdm

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils
from opencood.data_utils.datasets import build_dataset

import wandb
os.environ["WANDB_API_KEY"] = 'b058297d8947bc34e6e11764cffa6a3a94671dc6'
os.environ["WANDB_MODE"] = "offline"


def train_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument("--hypes_yaml", "-y", type=str,
                        help='data generation yaml file needed ')
    parser.add_argument('--eval_epoch', type=int, help='use epoch', default=-1)
    parser.add_argument('--model_dir', default='', required=True,
                        help='Continued training path')
    parser.add_argument('--fusion_method', '-f', default="intermediate",
                        help='passed to inference.')
    opt = parser.parse_args()
    return opt


def main():
    opt = train_parser()
    hypes = yaml_utils.load_yaml(opt.hypes_yaml, opt)
        
    train_setting = ['train_params', 'optimizer', 'lr_scheduler']
    for st in train_setting:
        hypes[st] = hypes['train_setting'][st]
    
    print('Dataset Building')
    opencood_train_dataset = build_dataset(hypes, visualize=False, train=True)
    opencood_validate_dataset = build_dataset(hypes,
                                              visualize=False,
                                              train=False)

    val_loader = DataLoader(opencood_validate_dataset,
                            batch_size=hypes['train_params']['batch_size'],
                            num_workers=8,
                            collate_fn=opencood_train_dataset.collate_batch_train,
                            shuffle=True,
                            pin_memory=True,
                            drop_last=True,
                            prefetch_factor=2)

    print('Creating Model')
    model = train_utils.create_model(hypes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # define the loss
    # criterion = train_utils.create_loss(hypes)

    # optimizer setup
    optimizer = train_utils.setup_optimizer(hypes, model)
    # lr scheduler setup
    num_steps = len(val_loader)

    modality_type_list = set(hypes['heter']['mapping_dict'].values())

    # if we want to train from last checkpoint.
    
    if opt.model_dir:
        saved_path = opt.model_dir
        resume_epoch, model = train_utils.load_saved_model(saved_path, model, opt.eval_epoch)

        if resume_epoch == 0: # 若无预训练的协作模型, 从保存在stage0_mx_collab的local模型加载
            modality_type_list = set(hypes['heter']['mapping_dict'].values())
            for modality_name in modality_type_list:
                _, model = train_utils.load_modality_saved_model( \
                    os.path.join(opt.model_dir, hypes['model']['args'][modality_name]['model_dir']), \
                    model, modality_name, True)
            if hasattr(model, 'negotiator'):
                model = train_utils.load_saved_model_negotiator(saved_path, model)
            
            resume_epoch = 'Val_distance' # individual combine 

    else:
        assert opt.model_dir

    
    if torch.cuda.is_available():
        model.to(device)
        # criterion.to(device)

    print('Validate start')
     
    test_modality_list = model.test_modality_list


    result_file = os.path.join(opt.model_dir, 'result.txt')
    if not os.path.exists(result_file):
        f = open(result_file, 'w')
        f.close()

    # max_ap50, max_ap50_epoch = find_best_pub_ap50(result_file, modes)

    valid_ave_dis_ori_dict = dict()
    valid_ave_dis_aligned_dict = dict()
    for modality_name in test_modality_list:
        valid_ave_dis_ori_dict[modality_name] = [0]
        valid_ave_dis_aligned_dict[modality_name] = [0]
        
    np.random.seed(40)
    # epoch = init_epoch
    model.eval()
    with torch.no_grad():
        for i, batch_data in enumerate(val_loader):
            # break
            if batch_data is None:
                continue
            model.zero_grad()
            optimizer.zero_grad()
            model.eval()

            batch_data = train_utils.to_device(batch_data, device)
            modality_dis_ori_dict, modality_dis_aligned_dict = \
                model(batch_data['ego'], batch_data['ego']['label_dict_single'])
            
            for modality_name in test_modality_list:
                modality_dis_ori = modality_dis_ori_dict[modality_name]
                print(f'{modality_name} val distance_ori {modality_dis_ori:.3f}')
                valid_ave_dis_ori_dict[modality_name].append(modality_dis_ori.item())
                
                modality_dis_aligned = modality_dis_aligned_dict[modality_name]
                print(f'{modality_name} val distance_aligned {modality_dis_aligned:.3f}')
                valid_ave_dis_aligned_dict[modality_name].append(modality_dis_aligned.item())
            print('\n')


    f = open(os.path.join(opt.model_dir, 'val.txt'), 'a+')
    for modality_name in test_modality_list:   
        f.write(f'Diatance of {modality_name}:\n')
        valid_ave_dis_ori = statistics.mean(valid_ave_dis_ori_dict[modality_name])
        valid_ave_dis_aligned = statistics.mean(valid_ave_dis_aligned_dict[modality_name])

        print('The validation dis of ori is %f' % (valid_ave_dis_ori))
        print('The validation dis of aligned is %f' % (valid_ave_dis_aligned))
        
        f.write(f'Ori: {valid_ave_dis_ori}\n')
        f.write(f'Aligned: {valid_ave_dis_aligned}\n')
        f.write(f'\n') 
    f.close()

    opencood_validate_dataset.reinitialize()
    torch.cuda.empty_cache()

if __name__ == '__main__':
    main()