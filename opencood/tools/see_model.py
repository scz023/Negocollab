import torch


# filename1 = 'checkpoints/opv2v_stamp_pyramid_m1m2/stage0_m1_collab/net_epoch31.pth'
# filename2 = 'checkpoints/opv2v_stamp_pyramid_m1m2/stage0_m2_collab/net_epoch19.pth'

# model_para = torch.load(filename2)

# a=0

# x = torch.Tensor(0.0)

filename = 'checkpoints/zz-opv2v_nego_from_m1/z-opv2v_nego_m0_guide_m1_pyramid-ft/stage0_m1_collab/net_epoch31.pth'
anchor_model = torch.load(filename)
dict_comm = dict()

for k, p in anchor_model.items():
    if 'comm' in k:
        dict_comm[k] = p


filname_epoch = 'checkpoints/zz-opv2v_nego_from_m1/z-opv2v_nego_m1m3_pyramid-ft/stage0_m1_collab/net_epoch31.pth'
epoch_model = torch.load(filname_epoch)
comm_equal = True
for k, p in dict_comm.items():
    cur_equal = torch.equal(p, epoch_model[k])
    if not cur_equal:
        print(f'Para of {k} not equal')
    comm_equal = comm_equal and torch.equal(p, epoch_model[k])
    
# if not comm_equal:
#     print(f'Model of {epoch} not equal')

# for epoch in range(2,8):
    # filname_epoch = f'checkpoints/opv2v_stamp_pyramid_m0m1/net_epoch{epoch}.pth'
    # epoch_model = torch.load(filname_epoch)
    # comm_equal = True
    # for k, p in dict_comm.items():
    #     cur_equal = torch.equal(p, epoch_model[k])
    #     if not cur_equal:
    #         print(f'Para of {k} not equal')
    #     comm_equal = comm_equal and torch.equal(p, epoch_model[k])
        
    # if not comm_equal:
    #     print(f'Model of {epoch} not equal')