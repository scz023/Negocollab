import glob
import re
import torch
import os

def load_modality_model(saved_path, modality, epoch=None):
    """
    Load saved model if exiseted

    Parameters
    __________
    saved_path : str
       model saved path
    model : opencood object
        The model instance.

    Returns
    -------
    model : opencood object
        The model instance loaded pretrained params.
    """
    assert os.path.exists(saved_path), '{} not found'.format(saved_path)

    def findLastCheckpoint(save_dir):
        file_list = glob.glob(os.path.join(save_dir, '*epoch*.pth'))
        if file_list:
            epochs_exist = []
            for file_ in file_list:
                result = re.findall(".*epoch(.*).pth.*", file_)
                epochs_exist.append(int(result[0]))
            initial_epoch_ = max(epochs_exist)
        else:
            initial_epoch_ = 0
        return initial_epoch_

    initial_epoch = findLastCheckpoint(saved_path)
    print('Loading epoch %d at modality %s' % (initial_epoch, modality))

    loaded_state_dict = torch.load(os.path.join(saved_path,
                    'net_epoch%d.pth' % initial_epoch), map_location='cpu')
    
    return initial_epoch, loaded_state_dict


def extract_modality(current_dir):
    modality_list = []

    # 遍历当前目录下的所有文件和子目录
    for dirname in os.listdir(current_dir):
        # 检查是否为符合条件的目录名称
        if dirname.startswith("stage0_") and dirname.endswith("_collab"):
            mx = dirname.split("_")[1]
            # 检查x是否为数字
            try:
                x = int(mx[1:])  # 提取x
            except ValueError:
                continue  # 跳过非数字的情况

            modality_list.append(mx)

    return modality_list


def add_adapter(modelpath_w_adapter, filepath_wo_adapter, comm = False):
    """把存在modelpath_w_adapter的model中的adapter参数添加到
    filepath_wo_adapter下的stage0_mx_collab中
    """
    
    modality_list = extract_modality(filepath_wo_adapter)
    print(modality_list)

    dict_model_wo_adapter = dict()
    dict_mode_epoch = dict()
    dict_modelpath = dict()

    for mode in modality_list:
        modepath = os.path.join(filepath_wo_adapter, 'stage0_{}_collab'.format(mode))
        epoch, mode_model = load_modality_model(modepath, mode)

        dict_model_wo_adapter[mode] = mode_model
        dict_mode_epoch[mode] = epoch
        dict_modelpath[mode] = modepath

    adapter_name = 'adapter_'
    if comm:
        adapter_name = 'comm_'
    model_w_adapter = torch.load(modelpath_w_adapter)
    for name, para in model_w_adapter.items():
        if adapter_name in name:
            if comm:
                mode = name[5:7] 
            else:
                mode = name[8:10] 
            if mode in modality_list:
                dict_model_wo_adapter[mode][name] = para
                print('{} para {} find!'.format(mode, name))

    
    for mode in modality_list:
        torch.save(dict_model_wo_adapter[mode], os.path.join(dict_modelpath[mode],
                                                             'net_epoch%d.pth' % (dict_mode_epoch[mode])))
        
        print('model of {} saved to {}!'.format(mode, dict_modelpath[mode] ))
        
# modelpath_w_adapter = '../../checkpoints/opv2v_mdpa_pyramid_m1m3/net_epoch2.pth'
# filepath_wo_adapter = '../../checkpoints/opv2v_mdpa_pyramid_m1m3(sd2)'
# add_adapter(modelpath_w_adapter, filepath_wo_adapter)

# modelpath_w_adapter = '../../checkpoints/opv2v_pnpda_pyramid_m1m3/net_epoch3.pth'
# filepath_wo_adapter = '../../checkpoints/opv2v_pnpda_pyramid_m1m3(sd2)'
# add_adapter(modelpath_w_adapter, filepath_wo_adapter)

# modelpath_w_adapter = './checkpoints/opv2v_nego_m1m3_pyramid/net_epoch1.pth'
# filepath_wo_adapter = './checkpoints/opv2v_nego_m1m3_gen_sd2'


# modelpath_w_adapter = './checkpoints/opv2v_pnpda_pyramid_m1m3/net_epoch1.pth'
# filepath_wo_adapter = './checkpoints/opv2v_pnpda_pyramid_m1m3(sd2)'
# add_adapter(modelpath_w_adapter, filepath_wo_adapter)


modelpath_w_adapter = '../../checkpoints/opv2v_pnpda_pyramid_m1m3_s2/net_epoch4.pth'
filepath_wo_adapter = '../../checkpoints/opv2v_pnpda_pyramid_m1m3(sd2)'
add_adapter(modelpath_w_adapter, filepath_wo_adapter)