import argparse
import os
import statistics
import glob
import time
import torch
from torch.utils.data import DataLoader, DistributedSampler
from tensorboardX import SummaryWriter

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils
from opencood.data_utils.datasets import build_dataset
from opencood.tools import multi_gpu_utils
from icecream import ic
import tqdm

# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --use_env opencood/tools/train_ddp.py --hypes_yaml ${CONFIG_FILE} [--model_dir  ${CHECKPOINT_FOLDER}
import wandb
os.environ["WANDB_API_KEY"] = 'b058297d8947bc34e6e11764cffa6a3a94671dc6'
os.environ["WANDB_MODE"] = "offline"
def train_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument("--hypes_yaml", "-y", type=str, required=True,
                        help='data generation yaml file needed ')
    parser.add_argument('--lsd_mode', '-lsd', type=int, default=1,
                        help='whether use lsd config')
    parser.add_argument('--model_dir', default='', required=True,
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
    opt.config_lsd = 1
    hypes = yaml_utils.load_yaml(opt.hypes_yaml, opt)
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
    
    # 除comm模块均为推理状态
    model.eval()
    
    # record lowest validation loss checkpoint.
    lowest_val_loss = 1e5
    lowest_val_epoch = -1

    # if we want to train from last checkpoint.
    if opt.model_dir:
        saved_path = opt.model_dir
        _, model = train_utils.load_saved_model(saved_path, model)
        
        init_epoch, model = train_utils.load_saved_model_comm(saved_path, model)
        
        # 更新save_path为comm_module路径
        saved_path = os.path.join(saved_path, 'comm_module')
        lowest_val_epoch = init_epoch
    else:
        assert opt.model_dir

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


    # define the loss
    criterion = train_utils.create_loss(hypes)

    # optimizer setup
    optimizer = train_utils.setup_optimizer(hypes, model_without_ddp)
    
    num_steps = len(train_loader)
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
        if opt.distributed:
            sampler_train.set_epoch(epoch)
        # the model will be evaluation mode during validation
        model.train()
        # try: # heter_model stage2
        #     model_without_ddp.model_train_init()
        # except:
        #     print("No model_train_init function")
            
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
                final_loss = criterion(ouput_dict, None)
            else:
                with torch.cuda.amp.autocast():
                    ouput_dict = model(batch_data['ego'])
                    final_loss = criterion(ouput_dict, None)

            criterion.logging(epoch, i, len(train_loader), writer, pbar=pbar2)
            train_ave_loss.append(final_loss.item())

            # if supervise_single_flag:
            #     if not opt.half:
            #         final_loss += criterion(ouput_dict, batch_data['ego']['label_dict_single'], suffix="_single") * hypes['train_params'].get("single_weight", 1)
            #     else:
            #         with torch.cuda.amp.autocast():
            #             final_loss += criterion(ouput_dict, batch_data['ego']['label_dict_single'], suffix="_single") * hypes['train_params'].get("single_weight", 1)
            #     criterion.logging(epoch, i, len(train_loader), writer, suffix="_single")

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
            # break
        train_ave_loss = statistics.mean(train_ave_loss)      
        wandb.log({"train_loss": train_ave_loss})

        if epoch % hypes['train_params']['save_freq'] == 0:
            if opt.config_lsd:
                model_comm_dict = train_utils.find_model_comm(model_without_ddp.load_state_dict())
                torch.save(model_comm_dict,
                    os.path.join(saved_path,
                            'comm_epoch%d.pth' % (epoch + 1)))
            else:
                torch.save(model_without_ddp.state_dict(),
                        os.path.join(saved_path,
                                        'net_epoch%d.pth' % (epoch + 1)))
            
        if epoch % hypes['train_params']['eval_freq'] == 0:
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

                    final_loss = criterion(ouput_dict,
                                           batch_data['ego']['label_dict'])
                    print(f'val loss {final_loss:.3f}')
                    valid_ave_loss.append(final_loss.item())
                    
                    writer.add_scalar('val_rec_loss', criterion.loss_dict['rec_loss'], epoch)
                    writer.add_scalar('val_cycle_loss', criterion.loss_dict['cycle_loss'], epoch)
                    writer.add_scalar('val_orth_loss', criterion.loss_dict['orth_loss'], epoch)
                    writer.add_scalar('val_unit_loss', criterion.loss_dict['unit_loss'], epoch)
                    writer.add_scalar('val_unify_loss', criterion.loss_dict['unify_loss'], epoch)
                    # val_ave_rec_loss.append(criterion.loss_dict['rec_loss'])
                    # val_ave_cycle_loss.append(criterion.loss_dict['cycle_loss'])
                    # val_ave_orth_loss.append(criterion.loss_dict['orth_loss'])
                    # val_ave_unit_loss.append(criterion.loss_dict['unit_loss'])

            valid_ave_loss = statistics.mean(valid_ave_loss)
            print('At epoch %d, the validation loss is %f' % (epoch,
                                                              valid_ave_loss))
            
            writer.add_scalar('Val_Loss_Comm', valid_ave_loss, epoch)
            # writer.add_scalar('val_rec_loss', val_ave_rec_loss, epoch)
            # writer.add_scalar('val_cycle_loss', val_ave_cycle_loss, epoch)
            # writer.add_scalar('val_orth_loss', val_ave_orth_loss, epoch)
            # writer.add_scalar('val_unit_loss', val_ave_unit_loss, epoch)
            wandb.log({"val_loss": valid_ave_loss})
            # wandb.log({"val_rec_loss": val_ave_rec_loss})
            # wandb.log({"val_cycle_loss": val_ave_cycle_loss})
            # wandb.log({"val_orth_loss": val_ave_orth_loss})
            # wandb.log({"val_unit_loss": val_ave_unit_loss}) 



            # lowest val loss
            if valid_ave_loss < lowest_val_loss:
                lowest_val_loss = valid_ave_loss
                if opt.config_lsd:
                    model_comm_dict = train_utils.find_model_comm(model_without_ddp.state_dict())
                    torch.save(model_comm_dict,
                        os.path.join(saved_path,
                                        'comm_epoch_bestval_at%d.pth' % (epoch + 1)))
                    if lowest_val_epoch != -1 and os.path.exists(os.path.join(saved_path,
                                        'comm_epoch_bestval_at%d.pth' % (lowest_val_epoch))):
                        os.remove(os.path.join(saved_path,
                                        'comm_epoch_bestval_at%d.pth' % (lowest_val_epoch)))
                               
                else:
                    torch.save(model_without_ddp.state_dict(),
                        os.path.join(saved_path,
                                        'net_epoch_bestval_at%d.pth' % (epoch + 1)))
                    if lowest_val_epoch != -1 and os.path.exists(os.path.join(saved_path,
                                        'net_epoch_bestval_at%d.pth' % (lowest_val_epoch))):
                        if opt.rank == 0:
                            os.remove(os.path.join(saved_path,
                                            'net_epoch_bestval_at%d.pth' % (lowest_val_epoch)))
                lowest_val_epoch = epoch + 1


        opencood_train_dataset.reinitialize()

    print('Training Finished, checkpoints saved to %s' % saved_path)

    # if opt.rank == 0:
    #     run_test = True
        
    #     # ddp training may leave multiple bestval
    #     if opt.config_lsd:
    #         bestval_model_list = glob.glob(os.path.join(saved_path, "comm_epoch_bestval_at*"))
    #     else:
    #         bestval_model_list = glob.glob(os.path.join(saved_path, "net_epoch_bestval_at*"))
        
    #     if len(bestval_model_list) > 1:
    #         import numpy as np
    #         if opt.config_lsd:
    #             bestval_model_epoch_list = [eval(x.split("/")[-1].lstrip("comm_epoch_bestval_at").rstrip(".pth")) for x in bestval_model_list]
    #         else:
    #             bestval_model_epoch_list = [eval(x.split("/")[-1].lstrip("net_epoch_bestval_at").rstrip(".pth")) for x in bestval_model_list]
    #         ascending_idx = np.argsort(bestval_model_epoch_list)
    #         for idx in ascending_idx:
    #             if idx != (len(bestval_model_list) - 1):
    #                 os.remove(bestval_model_list[idx])

    #     if run_test:
    #         fusion_method = opt.fusion_method
    #         cmd = f"python opencood/tools/inference.py --model_dir {saved_path} --fusion_method {fusion_method}"
    #         print(f"Running command: {cmd}")
    #         os.system(cmd)


if __name__ == '__main__':
    main()
