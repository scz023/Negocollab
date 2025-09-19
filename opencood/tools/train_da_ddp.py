import argparse
import os
import statistics
import glob
import threading
import time
import torch
from torch.utils.data import DataLoader, DistributedSampler
from tensorboardX import SummaryWriter
import wandb

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils
from opencood.data_utils.datasets import build_dataset
from opencood.tools import multi_gpu_utils
from icecream import ic
import tqdm
os.environ["WANDB_API_KEY"] = 'b058297d8947bc34e6e11764cffa6a3a94671dc6'
os.environ["WANDB_MODE"] = "offline"

# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --use_env opencood/tools/train_ddp.py --hypes_yaml ${CONFIG_FILE} [--model_dir  ${CHECKPOINT_FOLDER}

def train_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument("--hypes_yaml", "-y", type=str,
                        help='data generation yaml file needed ')
    parser.add_argument('--eval_epoch', type=int, help='use epoch', default=-1)
    parser.add_argument("--stage", "-s", type=str, default='contrastive',
                        help='Traning stage. \
                        nego: feature align. \
                        finetune: adapt to downstream tasks.')
    parser.add_argument('--run_test', type=str, default='True',
                        help='whether run inference.')
    parser.add_argument('--model_dir', default='',
                        help='Continued training path')
    parser.add_argument('--fusion_method', '-f', default="intermediate",
                        help='passed to inference.')
    parser.add_argument("--half", action='store_true',
                        help="whether train with half precision")
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    opt = parser.parse_args()
    return opt


def main():
    opt = train_parser()
    hypes = yaml_utils.load_yaml(opt.hypes_yaml, opt)
    
    """opt.stage: all, contrastive, finetune"""        
    
    train_setting = ['train_params', 'optimizer', 'lr_scheduler']
    for st in train_setting:
        if opt.stage == 'rep':
            set_key = 'representaion'
        elif opt.stage == 'ft':
            set_key = 'finetune'
        else:
            continue
        hypes[st] = hypes['train_setting'][set_key][st]

    hypes['loss']['args'].update({'stage': opt.stage})
    multi_gpu_utils.init_distributed_mode(opt)

    print('Dataset Building')
    opencood_train_dataset = build_dataset(hypes, visualize=False, train=True)
    opencood_validate_dataset = build_dataset(hypes,
                                              visualize=False,
                                              train=False)

    if opt.distributed:
        sampler_train = DistributedSampler(opencood_train_dataset)
        sampler_val = DistributedSampler(opencood_validate_dataset, shuffle=False)

        batch_sampler_train = torch.utils.data.BatchSampler(
            sampler_train, hypes['train_params']['batch_size'], drop_last=True)

        train_loader = DataLoader(opencood_train_dataset,
                                  batch_sampler=batch_sampler_train,
                                  num_workers=8,
                                  collate_fn=opencood_train_dataset.collate_batch_train)
        val_loader = DataLoader(opencood_validate_dataset,
                                sampler=sampler_val,
                                num_workers=8,
                                collate_fn=opencood_train_dataset.collate_batch_train,
                                drop_last=False)
    else:
        train_loader = DataLoader(opencood_train_dataset,
                                  batch_size=hypes['train_params'][
                                      'batch_size'],
                                  num_workers=8,
                                  collate_fn=opencood_train_dataset.collate_batch_train,
                                  shuffle=True,
                                  pin_memory=True,
                                  drop_last=True)
        val_loader = DataLoader(opencood_validate_dataset,
                                batch_size=hypes['train_params']['batch_size'],
                                num_workers=8,
                                collate_fn=opencood_train_dataset.collate_batch_train,
                                shuffle=True,
                                pin_memory=True,
                                drop_last=True)

    print('Creating Model')
    model = train_utils.create_model(hypes)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # record lowest validation loss checkpoint.
    lowest_val_loss = 1e5
    lowest_val_epoch = -1
    
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
        for modality_name in modality_type_list:
            modality_hypes = hypes['model']['args'][modality_name]
            modality_hypes['model_dir'] = os.path.join(opt.model_dir, modality_hypes['model_dir'])
                
        # modality_checkpoint_name = {}
        if init_epoch == 0:
            # 若文件夹下无预训练模型, 从每个模态的文件夹加载预训练参数, 从pub_codebook加载公共码本            
            for modality_name in modality_type_list:
                _, model = train_utils.load_modality_saved_model(\
                    hypes['model']['args'][modality_name]['model_dir'], \
                    model, modality_name)
                
        
        # 更新save_path为comm_module路径
        # saved_path = os.path.join(saved_path, 'comm_module')
        lowest_val_epoch = init_epoch
        # print(f"resume from {init_epoch} epoch.")

    else:
        assert opt.model_dir
            # modality_checkpoint_name[modality_name] = checkpoint_name
    # F.cosine_similarity(model.comm_m1.codebook, model.comm_m3.codebook, dim=1)
    # we assume gpu is necessary
    if torch.cuda.is_available():
        model.to(device)
        
    # ddp setting
    model_without_ddp = model

    if opt.distributed:
        model = \
            torch.nn.parallel.DistributedDataParallel(model,
                                                      device_ids=[opt.gpu],
                                                      find_unused_parameters=True) # True
        model_without_ddp = model.module

    # lr scheduler setup
    num_steps = len(train_loader)

    # define the loss
    criterion = train_utils.create_loss(hypes)

    # optimizer setup
    optimizer = train_utils.setup_optimizer(hypes, model_without_ddp)
    
    scheduler = train_utils.setup_lr_schedular(hypes, optimizer, num_steps, init_epoch=init_epoch)



    # record training
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

    # half precision training
    if opt.half:
        scaler = torch.cuda.amp.GradScaler()

    print('Training start')
    epoches = hypes['train_params']['epoches'] + init_epoch
    supervise_single_flag = False if not hasattr(opencood_train_dataset, "supervise_single") else opencood_train_dataset.supervise_single
    # used to help schedule learning rate

    for epoch in range(init_epoch, max(epoches, init_epoch)):
        if hypes['lr_scheduler']['core_method'] != 'cosineannealwarm':
            scheduler.step(epoch)
        if hypes['lr_scheduler']['core_method'] == 'cosineannealwarm':
            scheduler.step_update(epoch * num_steps + 0)
            
        for param_group in optimizer.param_groups:
            print('learning rate %f' % param_group["lr"])
        if opt.distributed:
            sampler_train.set_epoch(epoch)
        # the model will be evaluation mode during validation
        model.train()
        try: # heter_model stage2
            model_without_ddp.model_train_init()
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
            if not opt.half:
                ouput_dict = model(batch_data['ego'])
                ouput_dict['epoch'] = epoch
                final_loss = criterion(ouput_dict, batch_data['ego']['label_dict'], \
                                    batch_data['ego']['label_dict_single'])
            else:
                with torch.cuda.amp.autocast():
                    ouput_dict = model(batch_data['ego'])
                    ouput_dict['epoch'] = epoch
                    final_loss = criterion(ouput_dict, batch_data['ego']['label_dict'], \
                                        batch_data['ego']['label_dict_single'])

            criterion.logging(epoch, i, len(train_loader), writer)
            train_ave_loss.append(final_loss.item())
            final_loss.requires_grad_(True) 
            if not opt.half:    
                
                final_loss.backward()
                optimizer.step()
            else:
                scaler.scale(final_loss).backward()
                scaler.step(optimizer)
                scaler.update()

            pbar2.update(1)
            # torch.cuda.empty_cache()  # it will destroy memory buffer
            if hypes['lr_scheduler']['core_method'] == 'cosineannealwarm':
                scheduler.step_update(epoch * num_steps + i)
        train_ave_loss = statistics.mean(train_ave_loss)      
        wandb.log({"train_loss": train_ave_loss})
        if epoch % hypes['train_params']['save_freq'] == 0 or epoch == 1:
            torch.save(model_without_ddp.state_dict(),
                       os.path.join(saved_path,
                                    'net_epoch%d.pth' % (epoch + 1)))
            
        if epoch % hypes['train_params']['eval_freq'] == 0 or epoch == 1:
            valid_ave_loss = []

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
                    final_loss = criterion(ouput_dict, batch_data['ego']['label_dict'], \
                                   batch_data['ego']['label_dict_single'])
                    print(f'val loss {final_loss:.3f}')
                    valid_ave_loss.append(final_loss.item())

            valid_ave_loss = statistics.mean(valid_ave_loss)
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
                torch.save(model_without_ddp.state_dict(),
                       os.path.join(saved_path,
                                    'net_epoch_bestval_at%d.pth' % (epoch + 1)))
                if lowest_val_epoch != -1 and os.path.exists(os.path.join(saved_path,
                                    'net_epoch_bestval_at%d.pth' % (lowest_val_epoch))):
                    if opt.rank == 0:
                        os.remove(os.path.join(saved_path,
                                        'net_epoch_bestval_at%d.pth' % (lowest_val_epoch)))
                lowest_val_epoch = epoch + 1
            run_test = True
            if (opt.run_test == 'True') or (opt.run_test == 'true') or (opt.run_test == '1'):
                run_test = True
            elif (opt.run_test == 'False') or (opt.run_test == 'false') or (opt.run_test == '0'):
                run_test = False
                
            if run_test:
                epoch_th = threading.Thread(target=epoch_test, args=(opt, epoch))
                epoch_th.start()

        # scheduler.step(epoch)
        
        opencood_train_dataset.reinitialize()

    print('Training Finished, checkpoints saved to %s' % saved_path)

    if opt.rank == 0:
        run_test = True
        
        # ddp training may leave multiple bestval
        bestval_model_list = glob.glob(os.path.join(saved_path, "net_epoch_bestval_at*"))
        
        if len(bestval_model_list) > 1:
            import numpy as np
            bestval_model_epoch_list = [eval(x.split("/")[-1].lstrip("net_epoch_bestval_at").rstrip(".pth")) for x in bestval_model_list]
            ascending_idx = np.argsort(bestval_model_epoch_list)
            for idx in ascending_idx:
                if idx != (len(bestval_model_list) - 1):
                    os.remove(bestval_model_list[idx])

        # if run_test:
        #     fusion_method = opt.fusion_method
        #     cmd = f"python opencood/tools/inference.py --model_dir {saved_path} --fusion_method {fusion_method}"
        #     print(f"Running command: {cmd}")
        #     os.system(cmd)
def epoch_test(opt, epoch):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    cmd = f"python opencood/tools/inference.py \
        --model_dir {opt.model_dir} \
        --eval_epoch {epoch + 1}"
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
