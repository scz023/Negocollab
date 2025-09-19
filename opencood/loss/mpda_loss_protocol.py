import torch
import torch.nn as nn
import torch.nn.functional as F
from opencood.loss.contrastive_loss import ContrastiveLoss
from opencood.loss.point_pillar_depth_loss import PointPillarDepthLoss
from opencood.loss.point_pillar_pyramid_loss import PointPillarPyramidLoss
from opencood.tools.feature_show import feature_show
from opencood.tools.train_utils import to_device


class MpdaLossProtocol(PointPillarDepthLoss):
    def __init__(self, args):
        super().__init__(args)
        self.pyramid_loss = PointPillarPyramidLoss(args)
        self.loss_dict = {}

    def forward(self, output_dict, label_dict, label_dict_single=None, suffix=""):
        """
        Parameters
        ----------
        output_dict: 模型输出
        target_dict: 后处理输出
        """
        
        ego_modality = output_dict['ego_modality']
        fusion_method = output_dict['fusion_method']

        if suffix == '_single':
            if fusion_method == 'pyramid':
                sup_single_loss = self.pyramid_loss(output_dict, label_dict_single, suffix)                
            else:
                sup_single_loss = super().forward(output_dict, label_dict_single, suffix)
                
            self.loss_dict.update({"sup_single_loss": sup_single_loss})   
            return sup_single_loss
        

        da_cls_preds = output_dict['da_cls_preds']
        source_index = output_dict['source_idx']
        device = output_dict['da_cls_preds'].device
        da_loss = to_device(torch.zeros(1), device)

        N, C, H, W = da_cls_preds.shape
        da_cls_preds = da_cls_preds.permute(0, 2, 3, 1)
        da_targets = to_device(torch.zeros_like(da_cls_preds,
                                                dtype=torch.float32), device)
        for i in source_index:
            da_targets[i, :] = 1

        da_cls_preds = da_cls_preds.reshape(N, -1)
        da_targets = da_targets.reshape(N, -1)

        da_loss = F.binary_cross_entropy_with_logits(da_cls_preds,
                                                        da_targets)
        da_loss = torch.clamp(da_loss, min=-20, max=20)  # 根据需要调整范围


        self.loss_dict.update(
            {
                "da_loss": da_loss
            }
        )

        """协议模型只计算域分类损失"""
        # if ego_modality == 'm0':
        #     self.total_loss = da_loss
        #     return da_loss
        
        
        if fusion_method == 'pyramid':
            det_loss = self.pyramid_loss(output_dict, label_dict, suffix)               
        else:
            det_loss = super().forward(output_dict, label_dict)
        
        self.loss_dict.update(
            {
                "det_loss": det_loss
            }
        )
        
        total_loss = da_loss + det_loss

        self.total_loss = total_loss
        return total_loss

    def logging(self, epoch, batch_id, batch_len, writer = None, pbar=None, suffix=""):
        
        writer.add_scalar('Total_loss', self.total_loss.item(),
                            epoch*batch_len + batch_id)
        
        for k, v in self.loss_dict.items():
            if not isinstance(v, torch.Tensor):
                writer.add_scalar(k, v, epoch*batch_len + batch_id)
            else: 
                writer.add_scalar(k, v.item(), epoch*batch_len + batch_id)
            # print(k, v.item())
           
        print_msg ="[epoch %d][%d/%d]%s, || Loss: %.4f" % \
            (epoch, batch_id + 1, batch_len, suffix, self.total_loss.item())
        for k, v in self.loss_dict.items():
            k = k.replace("_loss", "").capitalize()
            print_msg = print_msg + f" || {k}: {v:.4f}"
        
        if pbar is None:
            print(print_msg)   
        else:
            pbar.set_description(print_msg)


if __name__ == "__main__":
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    ego = torch.rand(4, 256, 50, 176)  # .cuda()
    cav = torch.rand(4, 256, 50, 176)  # .cuda()
    mask = torch.rand(4, 50, 50, 176)>0.2
    
    args ={
        'tau': 0.1,
        'max_voxel': 40
    }
    data_dict = {"features_q": ego, "features_k": cav}
    target_dict = {"pos_region_ranges": mask}
    model = ContrastiveLoss(args)
    output = model(ego, cav, mask)
    # print(output)
    # print(output.shape)