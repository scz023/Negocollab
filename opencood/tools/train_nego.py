# -*- coding: utf-8 -*-
# Author: Yifan Lu <yifan_lu@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib

import argparse
import os
import statistics
import threading
import time

import torch
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
    parser.add_argument("--stage", "-s", type=str,
                        help='Traning stage. \
                        nsd: negotiate semantic decompsition. \
                        consis: pub weights consistance, local weights cycle consistnace. \
                        nego: nsd and nsd.')
    parser.add_argument('--lsd_mode', '-lsd', type=int, default=0,
                        help='whether use lsd config')
    parser.add_argument('--model_dir', default='', required=True,
                        help='Continued training path')
    parser.add_argument('--fusion_method', '-f', default="intermediate",
                        help='passed to inference.')
    parser.add_argument('--run_test', type=str, default='True',
                        help='whether run inference.')
    parser.add_argument('--use_cb', type=str, default='False',
                        help='whether collaborate in public or local sematic space')
    opt = parser.parse_args()
    return opt


def main():
    opt = train_parser()
    hypes = yaml_utils.load_yaml(opt.hypes_yaml, opt)
    
    """opt.stage: nsd, consis, nego"""
    hypes['model']['core_method'] = hypes['model']['core_method'][f'train_{opt.stage}']
    hypes['loss']['args']['stage'] = opt.stage
    if opt.stage == 'consis':
        hypes['model']['args']['comm_space'] = 'pub'
    if opt.stage == 'nsd':
        # hypes['model']['args']['comm_space'] = 'local'
        hypes['model']['args']['comm_space'] = 'direct_pub'
    # if opt.stage == 'nego':
    #     hypes['model']['args']['train_space'] = 
        
    
    train_setting = ['train_params', 'optimizer', 'lr_scheduler']
    for st in train_setting:
        hypes[st] = hypes['train_setting'][opt.stage][st]
    
    print('Dataset Building')
    opencood_train_dataset = build_dataset(hypes, visualize=False, train=True)
    opencood_validate_dataset = build_dataset(hypes,
                                              visualize=False,
                                              train=False)

    train_loader = DataLoader(opencood_train_dataset,
                              batch_size=hypes['train_params']['batch_size'],
                              num_workers=4,
                              collate_fn=opencood_train_dataset.collate_batch_train,
                              shuffle=True,
                              pin_memory=True,
                              drop_last=True,
                              prefetch_factor=2)
    val_loader = DataLoader(opencood_validate_dataset,
                            batch_size=hypes['train_params']['batch_size'],
                            num_workers=4,
                            collate_fn=opencood_train_dataset.collate_batch_train,
                            shuffle=True,
                            pin_memory=True,
                            drop_last=True,
                            prefetch_factor=2)

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
            init_epoch, model = train_utils.load_saved_model(saved_path, model)
        else:
            init_epoch, model = train_utils.load_saved_model(saved_path, model, opt.eval_epoch)
        
        
        # Add local path to pub_codebook path and local modality model path, and get full path
        model.pub_cb_path = os.path.join(opt.model_dir, model.pub_cb_path)
        model.pub_query_emb_path = os.path.join(opt.model_dir, model.pub_query_emb_path)
        for modality_name in modality_type_list:
            modality_hypes = hypes['model']['args'][modality_name]
            modality_hypes['model_dir'] = os.path.join(opt.model_dir, modality_hypes['model_dir'])
                
        # modality_checkpoint_name = {}
        if init_epoch == 0:
            # 若文件夹下无预训练模型, 从每个模态的文件夹加载预训练参数, 从pub_codebook加载公共码本            
            for modality_name in modality_type_list:
                load_comm = hypes['model']['args'][modality_name]['allied']
                if opt.stage == 'consis':
                    load_comm = True
                _, model = train_utils.load_modality_saved_model(\
                    hypes['model']['args'][modality_name]['model_dir'], \
                    model, modality_name, load_comm)
            
            if model.use_alliance: # 若使用联盟中的智能体辅助训练, 加载联盟预训练好的pub_codebook和pub_query_emb_path
                model.init_pub_codebook = torch.load(model.pub_cb_path)['pub_codes']
                model.init_pub_codebook.requires_grad = False
                model.pub_query_embeddings = torch.load(model.pub_query_emb_path)
                
        elif init_epoch == 1: # 参与协商智能体中, 有已经allied的时(即stage1联盟内协商已完成), 训练从epoch1开始
            for modality_name in modality_type_list:
                # 若未在联盟中, 加载该模态编码器预训练参数
                if not hypes['model']['args'][modality_name]['allied']: 
                    _, model = train_utils.load_modality_saved_model(\
                    hypes['model']['args'][modality_name]['model_dir'], \
                    model, modality_name, False)
            if model.use_alliance:
                model = train_utils.load_pub_cb(model, init_epoch) 
            
        else:
            if model.use_alliance:
                model = train_utils.load_pub_cb(model, init_epoch)
            # model = train_utils.load_pub_query(model, init_epoch)
        
        # init_epoch, model = train_utils.load_saved_model_comm(saved_path, model)
        
        # 更新save_path为comm_module路径
        # saved_path = os.path.join(saved_path, 'comm_module')
        lowest_val_epoch = init_epoch
        scheduler = train_utils.setup_lr_schedular(hypes, optimizer, num_steps, init_epoch=init_epoch)
        # print(f"resume from {init_epoch} epoch.")

    else:
        assert opt.model_dir
            # modality_checkpoint_name[modality_name] = checkpoint_name
    # F.cosine_similarity(model.comm_m1.codebook, model.comm_m3.codebook, dim=1)
    # we assume gpu is necessary
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
    epoches = hypes['train_params']['epoches']
    supervise_single_flag = False if not hasattr(opencood_train_dataset, "supervise_single") else opencood_train_dataset.supervise_single
    # used to help schedule learning rate

    for epoch in range(init_epoch, max(epoches, init_epoch)):
        if hypes['lr_scheduler']['core_method'] != 'cosineannealwarm':
            scheduler.step(epoch)
        if hypes['lr_scheduler']['core_method'] == 'cosineannealwarm':
            scheduler.step_update(epoch * num_steps + 0)
            
        for param_group in optimizer.param_groups:
            print('learning rate %f' % param_group["lr"])
        # the model will be evaluation mode during validation
        model.train()
        try: # heter_model stage2
            model.model_train_init()
        except:
            print("No model_train_init function")
            
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
            criterion.logging(epoch, i, len(train_loader), writer, pbar=pbar2)
            
            train_ave_loss.append(final_loss.item())

            # back-propagation
            final_loss.backward()
            optimizer.step()
            
            pbar2.update(1)
            # torch.cuda.empty_cache()  # it will destroy memory buffer
            if hypes['lr_scheduler']['core_method'] == 'cosineannealwarm':
                scheduler.step_update(epoch * num_steps + i)
            # break
        
        train_ave_loss = statistics.mean(train_ave_loss)      
        wandb.log({"train_loss": train_ave_loss})

        if epoch % hypes['train_params']['save_freq'] == 0 or epoch == 1:
            torch.save(model.state_dict(),
                    os.path.join(saved_path,
                                    'net_epoch%d.pth' % (epoch + 1)))
            train_utils.save_pub_cb(model, epoch + 1)
            
            if opt.stage == 'consis':
                train_utils.save_pub_query_emb(model, epoch + 1)


        if epoch % hypes['train_params']['eval_freq'] == 0 or epoch == 1:
            
            # torch.cuda.empty_cache()  # it will destroy memory buffer
            
            valid_ave_loss = []
            # val_ave_rec_loss = []
            # val_ave_cycle_loss = []
            # val_ave_orth_loss = []
            # val_ave_unit_loss = []

            with torch.no_grad():
                for i, batch_data in enumerate(val_loader):
                    if batch_data is None:
                        continue
                    model.zero_grad()
                    optimizer.zero_grad()
                    model.eval()

                    batch_data = train_utils.to_device(batch_data, device)
                    batch_data['ego']['epoch'] = epoch
                    ouput_dict = model(batch_data['ego'])
                    
                    ouput_dict['epoch'] = epoch
                    final_loss = criterion(ouput_dict,
                                           batch_data['ego']['label_dict_single'])
                    print(f'val loss {final_loss:.3f}')
                    valid_ave_loss.append(final_loss.item())
                    
                    
                    for k ,v in criterion.loss_dict.items():
                        k = 'val_' + k
                        writer.add_scalar(k, v, epoch)
                        
                    # break


            valid_ave_loss = statistics.mean(valid_ave_loss)
            # val_ave_rec_loss = statistics.mean(val_ave_rec_loss)
            # val_ave_cycle_loss = statistics.mean(val_ave_cycle_loss)
            # val_ave_orth_loss = statistics.mean(val_ave_orth_loss)
            # val_ave_unit_loss = statistics.mean(val_ave_unit_loss)
            print('At epoch %d, the validation loss is %f' % (epoch,
                                                              valid_ave_loss))
            
            writer.add_scalar('Val_Loss', valid_ave_loss, epoch)
            # writer.add_scalar('val_rec_loss', val_ave_rec_loss, epoch)
            # writer.add_scalar('val_cycle_loss', val_ave_cycle_loss, epoch)
            # writer.add_scalar('val_orth_loss', val_ave_orth_loss, epoch)
            # writer.add_scalar('val_unit_loss', val_ave_unit_loss, epoch)
            wandb.log({"val_loss": valid_ave_loss})



            # lowest val loss
            if valid_ave_loss < lowest_val_loss:
                lowest_val_loss = valid_ave_loss
                torch.save(model.state_dict(),
                    os.path.join(saved_path,
                                    'net_epoch_bestval_at%d.pth' % (epoch + 1)))
                if lowest_val_epoch != -1 and os.path.exists(os.path.join(saved_path,
                                    'net_epoch_bestval_at%d.pth' % (lowest_val_epoch))):
                    os.remove(os.path.join(saved_path,
                                    'net_epoch_bestval_at%d.pth' % (lowest_val_epoch)))
                
                # save pub_code and local model parameters for each modality
                # if model.use_alliance:
                    # 若在联盟中, 更新默认的pub_codebook
                pub_codebook = {'pub_codes': model.pub_codebook, "pub2local_dict":model.pub2local_dict}
                torch.save(pub_codebook, model.pub_cb_path)
                
                if opt.stage == 'consis':
                    torch.save(model.pub_query_embeddings, model.pub_query_emb_path)                
                
                for modality_name in modality_type_list:                    
                    train_utils.save_model_back_to_local(\
                        hypes['model']['args'][modality_name]['model_dir'], model, modality_name)
                    # torch.save(modality_checkpoint_name[modality_name], modality_model_include_comm)
                lowest_val_epoch = epoch + 1
            
            run_test = True
            if (opt.run_test == 'True') or (opt.run_test == 'true') or (opt.run_test == '1'):
                run_test = True
            elif (opt.run_test == 'False') or (opt.run_test == 'false') or (opt.run_test == '0'):
                run_test = False
                
            if run_test:
                epoch_th = threading.Thread(target=epoch_test, args=(hypes, opt, epoch))
                epoch_th.start()
                # 从result.txt读入AP@0.5和AP@0.7的值并写入日志文件
                # ap50_dict, ap70_dict = load_ap_values(os.path.join(opt.model_dir, 'result.txt'), modes)
                # for mode in modes:
                #     writer.add_scalar(mode + '_AP@0.5', ap50_dict[mode], epoch + 1)
                #     writer.add_scalar(mode + '_AP@0.7', ap70_dict[mode], epoch + 1)

        opencood_train_dataset.reinitialize()

    print('Training Finished, checkpoints saved to %s' % saved_path)

    # run_test = True
    # if run_test:
    #     fusion_method = opt.fusion_method
    #     cmd = f"python opencood/tools/inference.py --model_dir {saved_path} --fusion_method {fusion_method}"
    #     print(f"Running command: {cmd}")
    #     os.system(cmd)


def epoch_test(hypes, opt, epoch):
    modes = list(set(hypes['heter']['mapping_dict'].values()))
    if opt.stage == "nsd":
        modes.insert(0, 'direct_pub')
    elif opt.stage == "consis":
        modes.insert(0, 'pub')
    
    for mode in modes:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        cmd = f"python opencood/tools/inference.py \
            --model_dir {opt.model_dir} \
            --collab_mode {mode} \
            --eval_epoch {epoch + 1} \
            --use_cb True"
        print(f"Running command: {cmd}")
        os.system(cmd)
        time.sleep(0.1)
    with open(os.path.join(opt.model_dir, 'result.txt'), 'a+') as f:
        f.write('\n')
    time.sleep(0.1)

def load_ap_values(file_name, mode_list):
    """
    从文件目录下的result.txt文件中提取AP @0.5和AP @0.7的值
    :param file_name: 文件名称
    :return: 包含模型名和对应AP值的字典
    """
    f = open(file_name, 'r', encoding='utf-8')
    file_content = f.read()
    f.close()
    
    # 将文件内容按行分割
    lines = file_content.strip().split('\n')

    # print(lines)
    
    ap50_dict = {}
    ap70_dict = {}
    for idx in range(len(mode_list)):
        mode = mode_list[idx]
        msg_mode = lines[idx-3]
        if '@0.5:' in msg_mode:   
            locate_ap50 = msg_mode.index('@0.5:')
            ap50_dict[mode] = float(msg_mode[locate_ap50+7 : locate_ap50+13])
            
            locate_ap70 = msg_mode.index('@0.7:')
            ap70_dict[mode] = float(msg_mode[locate_ap70+7 : locate_ap70+13])

    return ap50_dict, ap70_dict


if __name__ == '__main__':
    main()
