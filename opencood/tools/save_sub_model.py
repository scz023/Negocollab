import torch
import os

modality_name_list = ['m1', 'm2', 'm3', 'm4']
epoch_list = [31, 19, 25, 35]

for i in range(4):
    modality_name = modality_name_list[i]
    epoch = epoch_list[i]

    file = f'checkpoints/z-opv2v_pnpda_protocol_m1m2m3m4_pyramid/stage0_{modality_name}_collab/net_epoch{epoch}.pth'
    model = torch.load(file)
    adapter_dict = {}
    for k, v in model.items():
        if ('adapter' in k):
        # if ('adapter' in k) and ('calibrator' not in k):
            adapter_dict[k] = v
    torch.save(adapter_dict, f'adapters/adapter_pnpda_{modality_name}.pth')


    file = f'checkpoints/z-opv2v_mpda_m1m2m3m4_pyramid/stage0_{modality_name}_collab/net_epoch{epoch}.pth'
    model = torch.load(file)
    adapter_dict = {}
    for k, v in model.items():
        if ('adapter' in k) and ('classifier' not in k):
            adapter_dict[k] = v

    torch.save(adapter_dict, f'adapters/adapter_mpda_{modality_name}.pth')


    file = f'checkpoints/z-opv2v_nego_m1m2m3m4_pyramid/stage0_{modality_name}_collab/net_epoch{epoch}.pth'
    model = torch.load(file)
    adapter_dict = {}
    for k, v in model.items():
        if 'comm' in k:
            adapter_dict[k] = v
    torch.save(adapter_dict, f'adapters/comm_nego_{modality_name}.pth')

    file = f'checkpoints/z-opv2v_stamp_pyramid_m1m2m3m4/stage0_{modality_name}_collab/net_epoch{epoch}.pth'
    model = torch.load(file)
    adapter_dict = {}
    for k, v in model.items():
        if 'comm' in k:
            adapter_dict[k] = v
    torch.save(adapter_dict, f'adapters/comm_stamp_{modality_name}.pth')