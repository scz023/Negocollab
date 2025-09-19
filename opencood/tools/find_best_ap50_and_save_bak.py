# -*- coding: utf-8 -*-
# Author: Yifan Lu <yifan_lu@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib

import argparse
import os
import statistics
import threading
import time

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
    # parser.add_argument("--hypes_yaml", "-y", type=str,
    #                     help='data generation yaml file needed ')
    parser.add_argument('--model_dir', default='', required=True,
                        help='Continued training path')
    parser.add_argument('--eval_epoch', type=int, help='use epoch', default=-1)

    opt = parser.parse_args()
    return opt


def main():
    """
    寻找从result.txt中寻找最大ap对应的epoch, 并将comm模块的参数保存回stage0_mx_collab
    
    """
    opt = train_parser()
    hypes = yaml_utils.load_yaml(None, opt)

    if isinstance(hypes['model']['core_method'], dict):
        if 'train_nego' in hypes['model']['core_method'].keys():
            hypes['model']['core_method'] = hypes['model']['core_method']['train_nego']
        else:
            hypes['model']['core_method'] = hypes['model']['core_method']['train']

    print('Creating Model')
    model = train_utils.create_model(hypes)
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    modality_type_list = set(hypes['heter']['mapping_dict'].values())

    # if we want to train from last checkpoint.
    if opt.model_dir:
        saved_path = opt.model_dir
        
        for modality_name in modality_type_list:
            # Add local path to pub_codebook path and local modality model path, and get full path
            modality_hypes = hypes['model']['args'][modality_name]
            modality_hypes['model_dir'] = os.path.join(opt.model_dir, modality_hypes['model_dir'])
        
        """加载训练好的整体模型"""
        if opt.eval_epoch == -1:
            init_epoch, model = train_utils.load_saved_model_ap(saved_path, model)
        else:
            init_epoch, model = train_utils.load_saved_model_ap(saved_path, model, opt.eval_epoch)

        load_epoch = init_epoch

    else:
        assert opt.model_dir

    
    # if torch.cuda.is_available():
    #     model.to(device)


    result_file = os.path.join(opt.model_dir, 'result.txt')
    if not os.path.exists(result_file):
        f = open(result_file, 'w')
        f.close()
    
    # max_ap50, max_ap50_epoch = find_best_pub_ap50(result_file, modes)
    if not hasattr(opt, 'eval_epoch'):
        ap50, ap50_epoch = find_best_pub_ap50(os.path.join(opt.model_dir, 'result.txt'), modality_type_list)
        print(f"Best ap50 at epoch {ap50_epoch}, with {ap50}")

        load_epoch = ap50_epoch


    _, model = train_utils.load_saved_model_ap(saved_path, model, load_epoch) # 重新加载最高ap对应的模型

    if not hasattr(opt, 'eval_epoch'):
        torch.save(model.state_dict(),
            os.path.join(saved_path,
                            'net_epoch_bestap_at%d.pth' % (ap50_epoch)))
        if os.path.exists(os.path.join(saved_path,
                            'net_epoch_bestap_at%d.pth' % (init_epoch))):
            os.remove(os.path.join(saved_path,
                            'net_epoch_bestap_at%d.pth' % (init_epoch)))            
    

    """ 将总模型中的所有参数保存回本地(自然包含comm和adapter) """
    for modality_name in modality_type_list:                    
        hete_component = train_utils.save_model_back_to_local(\
            hypes['model']['args'][modality_name]['model_dir'], model, modality_name)
        
        print(f'Save {hete_component} module in best ap at epoch {load_epoch} back to model of {modality_name}')
    
    """将nego参数保存到negotiator文件夹下"""

    if 'nego' in hypes['model']['core_method']:
        train_utils.save_model_negotiator(saved_path, model)

    print(f'Negotiator of epoch {load_epoch} saved')



def find_best_pub_ap50(file_name, mode_list):
    """
    从文件目录下的result.txt文件中提取最大AP @0.5值和对应的epoch
    :param file_name: 文件名称
    :return: 包含模型名和对应AP值的字典
    """
    f = open(file_name, 'r', encoding='utf-8')
    file_content = f.read()
    f.close()
    
    epoch_results = len(mode_list) + 1
    
    # 将文件内容按行分割
    lines = file_content.strip().split('\n')

    max_ap50, best_ap50_epoch = 0, -1
    if len(lines) < epoch_results:
        return max_ap50, best_ap50_epoch
    
    max_ap50 = 0
    best_ap50_epoch = -1
    lidx = 0
    while lidx < len(lines):
        # mode = mode_list[idx]
        msg_mode = lines[lidx]
        
        if 'ind' in msg_mode: 
            lidx = lidx + epoch_results + 1
            continue

        locate_ap50 = msg_mode.index('@0.5:')
        ap50 = float(msg_mode[locate_ap50+7 : locate_ap50+13])

        if ap50 >  max_ap50:
            max_ap50 = ap50
            locate_epoch = msg_mode.index('Epoch: ')
            best_ap50_epoch = int(msg_mode[locate_epoch+7 : locate_epoch + 9])
            # print(msg_mode[locate_epoch+7 : locate_epoch + 9])
        lidx = lidx + epoch_results + 1

    return max_ap50, best_ap50_epoch


if __name__ == '__main__':
    main()
