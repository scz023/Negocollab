# -*- coding: utf-8 -*-
# Author: Yifan Lu <yifan_lu@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib

import argparse
from collections import OrderedDict
import os
import statistics
import threading
import time

import torch

from opencood.tools.pcgrad import PCGrad
from opencood.tools.pcgrad_fast import PCGradFast
from torch.utils.data import DataLoader, DistributedSampler
from opencood.tools.pcgrad_ori import PCGradOri
torch.autograd.set_detect_anomaly(True)
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tensorboardX import SummaryWriter
import tqdm

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import multi_gpu_utils, train_utils
from opencood.data_utils.datasets import build_dataset

import wandb
os.environ["WANDB_API_KEY"] = 'b058297d8947bc34e6e11764cffa6a3a94671dc6'
os.environ["WANDB_MODE"] = "offline"


def train_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument("--hypes_yaml", "-y", type=str,
                        help='data generation yaml file needed ')
    parser.add_argument('--eval_epoch', type=int, help='use epoch', default=-1)
    parser.add_argument("--stage", "-s", type=str,
                        help='Traning stage. \
                        nego: negotiate common feature space. \
                        ft: Downstream task adaptation. \
                        nego: nsd and nsd.')
    parser.add_argument('--model_dir', default='', required=True,
                        help='Continued training path')
    parser.add_argument('--fusion_method', '-f', default="intermediate",
                        help='passed to inference.')
    parser.add_argument('--run_test', type=str, default='True',
                        help='whether run inference.')
    parser.add_argument('--use_cb', type=str, default='False',
                        help='whether collaborate in public or local sematic space')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    opt = parser.parse_args()
    return opt


def main():
    opt = train_parser()
    hypes = yaml_utils.load_yaml(opt.hypes_yaml, opt)
    multi_gpu_utils.init_distributed_mode(opt)
    
    """opt.stage: nego, add, ft"""
    

    if opt.stage == 'nego' or opt.stage == 'unify':
        hypes['model']['args']['comm_space'] = 'pub'
        hypes['model']['core_method'] = hypes['model']['core_method']['train']
        hypes['loss']['core_method'] = hypes['loss']['core_method']['nego']
        hypes['model']['args'].update({'stage': 'nego'})
        
    # if opt.stage == 'ft':
    #     hypes['model']['args']['comm_space'] = 'pub'
    #     hypes['model']['args'].update({'stage': 'ft'})

    #     hypes['model']['core_method'] = hypes['model']['core_method']['train']
    #     hypes['loss']['core_method'] = hypes['loss']['core_method']['ft']

        

        # hypes['fusion']['core_method'] = 'intermediateheter'
        # hypes['model']['core_method'] = hypes['model']['core_method']['inf']
        
        # if  hypes['model']['args']['m1']['fusion_net']['method'] == 'pyramid':
        #     hypes['loss']['core_method'] = 'point_pillar_pyramid_loss'
        #     supervise_single_flag = True
        #     # supervise_single_flag = False
        # else:
        #     hypes['loss']['core_method'] = 'point_pillar_depth_loss'
        
    train_setting = ['train_params', 'optimizer', 'lr_scheduler']
    for st in train_setting:
        hypes[st] = hypes['train_setting'][opt.stage][st]
    
    print('Dataset Building')
    opencood_train_dataset = build_dataset(hypes, visualize=False, train=True)
    # opencood_validate_dataset = build_dataset(hypes,
    #                                           visualize=False,
    #                                           train=False)
    if opt.distributed:
        sampler_train = DistributedSampler(opencood_train_dataset)
        batch_sampler_train = torch.utils.data.BatchSampler(
            sampler_train, hypes['train_params']['batch_size'], drop_last=True)

        train_loader = DataLoader(opencood_train_dataset,
                                  batch_sampler=batch_sampler_train,
                                  num_workers=8,
                                  collate_fn=opencood_train_dataset.collate_batch_train)
    
    else:
        train_loader = DataLoader(opencood_train_dataset,
                                batch_size=hypes['train_params']['batch_size'],
                                num_workers=4,
                                collate_fn=opencood_train_dataset.collate_batch_train,
                                shuffle=True,
                                pin_memory=True,
                                drop_last=True,
                                prefetch_factor=2)
    # val_loader = DataLoader(opencood_validate_dataset,
    #                         batch_size=hypes['train_params']['batch_size'],
    #                         num_workers=4,
    #                         collate_fn=opencood_train_dataset.collate_batch_train,
    #                         shuffle=True,
    #                         pin_memory=True,
    #                         drop_last=True,
    #                         prefetch_factor=2)

    print('Creating Model')
    model = train_utils.create_model(hypes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 除comm模块均为推理状态
    # model.eval()
    # for name, sub_model in model.named_children():
        # if 'comm' in name:
            # sub_model.train()
    # model.backbone_m1.training
    # record lowest validation loss checkpoint.
    lowest_val_loss = 1e5
    lowest_val_epoch = -1
    
    #   lowest_val_loss = 1e5

    # define the loss
    criterion = train_utils.create_loss(hypes)

    # optimizer setup
    optimizer = train_utils.setup_optimizer(hypes, model)
    # lr scheduler setup
    num_steps = len(train_loader)

    modality_type_list = set(hypes['heter']['mapping_dict'].values())

    # if we want to train from last checkpoint.
    
    if opt.model_dir:
        saved_path = opt.model_dir
        if opt.eval_epoch == -1:
            init_epoch, model = train_utils.load_saved_model_ap(saved_path, model)
        else:
            init_epoch, model = train_utils.load_saved_model_ap(saved_path, model, opt.eval_epoch)
        
        
        # Add local path to pub_codebook path and local modality model path, and get full path
        for modality_name in modality_type_list:
            modality_hypes = hypes['model']['args'][modality_name]
            modality_hypes['model_dir'] = os.path.join(opt.model_dir, modality_hypes['model_dir'])
                
        # modality_checkpoint_name = {}
        if init_epoch == 0:
            # 若文件夹下无预训练模型, 从每个模态的文件夹加载预训练参数, 从pub_codebook加载公共码本            
            for modality_name in modality_type_list:
                load_comm = hypes['model']['args'][modality_name]['allied']
                _, model = train_utils.load_modality_saved_model_ap(\
                    hypes['model']['args'][modality_name]['model_dir'], \
                    model, modality_name, load_comm)
            
                
        elif init_epoch == 1: # 参与协商智能体中, 有已经allied的时(即stage1联盟内协商已完成), 训练从epoch1开始
            for modality_name in modality_type_list:
                # 若未在联盟中, 加载该模态编码器预训练参数
                if not hypes['model']['args'][modality_name]['allied']: 
                    _, model = train_utils.load_modality_saved_model_ap(\
                    hypes['model']['args'][modality_name]['model_dir'], \
                    model, modality_name, False)

        lowest_val_epoch = init_epoch
        scheduler = train_utils.setup_lr_schedular(hypes, optimizer, num_steps, init_epoch=init_epoch)
        # print(f"resume from {init_epoch} epoch.")

    else:
        assert opt.model_dir

    
    if torch.cuda.is_available():
        model.to(device)
        criterion.to(device)



    # model.
    # record training
    # In train, record every step in every epoch
    # In validation, record mean loss of all steps in every epoch 
    writer = SummaryWriter(saved_path)    
    
    wandb.init(
        # set the wandb project where this run will be logged
        project=hypes["name"]+time.strftime('%m_%d_%H_%M_%S'),
        # track hyperparameters and run metadata
        config={
        "learning_rate": hypes["optimizer"]["lr"],
        "architecture": hypes["name"],
        "dataset": hypes["root_dir"],
        "epochs": hypes["train_params"]["epoches"],
        }
    )

    print('Training start')
    #  epoches = hypes['train_params']['epoches']
    # 从init_epoch开始, 继续训练epoches轮
    epoches = hypes['train_params']['epoches'] + init_epoch
    # supervise_single_flag = False if not hasattr(opencood_train_dataset, "supervise_single") else opencood_train_dataset.supervise_single
    # used to help schedule learning rate

    if opt.stage == 'nego' or opt.stage == 'unify':
        assert len(model.allied_modality_list) == 0, "nego stage no allied agent!"
    elif opt.stage == 'add':
        assert ((len(model.allied_modality_list) > 0) and (len(model.newtype_modality_list) > 0)), "add stage must have allied agent!"

    if hypes['train_setting']['pc_grad'] and model.allied_num == 0:
        # pc_grader = PCGrad(model, hypes)
        pc_grader = PCGradFast(model, hypes)
        # pc_grader = PCGradOri(optimizer, model)

    model_without_ddp = model

    if opt.distributed:
        model = \
            torch.nn.parallel.DistributedDataParallel(model,
                                                      device_ids=[opt.gpu],
                                                      find_unused_parameters=True) # True
        model_without_ddp = model.module

    for epoch in range(init_epoch, max(epoches, init_epoch)):
        if hypes['lr_scheduler']['core_method'] != 'cosineannealwarm':
            scheduler.step(epoch)
        if hypes['lr_scheduler']['core_method'] == 'cosineannealwarm':
            scheduler.step_update(epoch * num_steps + 0)
            
        for param_group in optimizer.param_groups:
            print('learning rate %f' % param_group["lr"])
            
        if opt.distributed:
                    sampler_train.set_epoch(epoch)
        # if opt.stage == 'nego':
        #     if epoch % 2 == 1: # 奇数, 训练nego
        #         model.set_only_nego_trainable()
        #     elif epoch % 2 == 0: # 偶数次, 训练s&r
        #         model.set_only_comm_trainable()
            
        pbar2 = tqdm.tqdm(total=len(train_loader), leave=True)
        
        train_ave_loss = []
        for i, batch_data in enumerate(train_loader):
            if batch_data is None or batch_data['ego']['object_bbx_mask'].sum()==0:
                continue
            model.zero_grad()
            optimizer.zero_grad()
            batch_data = train_utils.to_device(batch_data, device)
            batch_data['ego']['epoch'] = epoch
            ouput_dict = model(batch_data['ego'])
            
            # reset loss computation logic
            ouput_dict['epoch'] = epoch
            final_loss = criterion(ouput_dict, batch_data['ego']['label_dict_single'])
            # final_loss = criterion(ouput_dict, batch_data['ego']['label_dict'])
            criterion.logging(epoch, i, len(train_loader), writer, pbar=pbar2)
            
            
            train_ave_loss.append(final_loss.item())
            
            # if opt.stage == 'nego':
            #     # if epoch == 0:
            #     if epoch % 2 == 1: # 训练nego, 使用梯度手术消解各模态优化冲突
            #         pc_grader.pc_backward(criterion.modality_loss_dict.values())
            #     elif epoch % 2 == 0: # 训练s&r, 正常方式更新参数 
            #         final_loss.backward()
            #         optimizer.step()                    
            # else:
            #     final_loss.backward()
            #     optimizer.step()
            
            # back-propagatio
            # if hypes['train_setting']['pc_grad'] and model_without_ddp.allied_num == 0:
            #     pc_grader.pc_backward(criterion.modality_loss_dict.values())
            # else:
            #     final_loss.backward()


            final_loss.backward()

            if hypes['train_setting']['pc_grad'] and model_without_ddp.allied_num == 0:
                pc_grader.grad_surgery()

            optimizer.step()

            
            # 若negotiator共享参数, 单独更新各模态中negotiator的参数
            # if hypes['model']['args']['share_negotiator']:
                # model._negotiator_para_update()
            
            pbar2.update(1)
            # torch.cuda.empty_cache()  # it will destroy memory buffer
            if hypes['lr_scheduler']['core_method'] == 'cosineannealwarm':
                scheduler.step_update(epoch * num_steps + i)
            # break
        
        train_ave_loss = statistics.mean(train_ave_loss)      
        wandb.log({"train_loss": train_ave_loss})

        if epoch % hypes['train_params']['save_freq'] == 0:
            # nomodule_state_dict = OrderedDict()
            # for k,v in model.state_dict().items():
            #     new_state_dict[k[7:]] = v
            # net.load_state_dict(new_state_dict)

            # model_
            torch.save(model_without_ddp.state_dict(),
                    os.path.join(saved_path,
                                    'net_epoch%d.pth' % (epoch + 1)))


        # if epoch % hypes['train_params']['eval_freq'] == 0:                       
        #     thread_test = threading.Thread(target=start_val, args=(opt, epoch + 1))
        #     thread_test.start()

        torch.cuda.empty_cache() 

        opencood_train_dataset.reinitialize()

    print('Training Finished, checkpoints saved to %s' % saved_path)

    # run_test = True
    # if run_test:
    #     fusion_method = opt.fusion_method
    #     cmd = f"python opencood/tools/inference.py --model_dir {saved_path} --fusion_method {fusion_method}"
    #     print(f"Running command: {cmd}")
    #     os.system(cmd)


def start_val(opt, epoch):
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    cmd = f"python opencood/tools/train_nego_vanilla_val_inf.py \
        -y None \
        --model_dir {opt.model_dir} \
        -s {opt.stage} \
        --eval_epoch {epoch}"
    print(f"Running command: {cmd}")

    # print(f"Running command: {cmd}")
    os.system(cmd)
    # time.sleep(0.1)


if __name__ == '__main__':
    main()
