import torch
import os


# model_s1_file = 'checkpoints/opv2v_pnpda_pyramid_m1m2_s2/stage0_m1_collab/net_epoch31.pth'
# model_s1_file = 'checkpoints/opv2v_pnpda_pyramid_m1m2_s2/stage0_m2_collab/net_epoch19.pth'
# model_s2_file = 'checkpoints/opv2v_pnpda_pyramid_m1m2_s2/net_epoch9.pth'

model_s1_file = 'checkpoints/z-opv2v_nego_m1m2m3m4_pyramid-ft/stage0_m2_collab/net_epoch19.pth'
model_s2_file = 'checkpoints/z-opv2v_nego_m1m2_pyramid-ft/stage0_m2_collab/net_epoch19.pth'

model_s1 = torch.load(model_s1_file)
model_s2 = torch.load(model_s2_file)

for name, para in model_s1.items():
    para_s1 = model_s1[name]
    if name not in model_s2.keys():
        continue
    
    para_s2 = model_s2[name]
    if not torch.equal(para_s1, para_s2):           
        print('changed', name)
    else:
        # continue
        if 'comm' in name:
            print('remain', name)
        
# stage = 'nego'
# with open('/home/scz/HEAL/checkpoints/opv2v_mdpa_pyramid_m1m2/result.txt', 'a+') as f:
#     f.write(f'Results of stage {stage}:\n')