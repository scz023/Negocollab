import copy
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import load_sample_images
from torchvision.ops import roi_align
from opencood.loss.torch_dist import reduce_mean
from opencood.tools.feature_show import feature_show
import matplotlib.pyplot as plt
import numpy as np
from torchvision.transforms.functional import to_pil_image

def corners_nd(dims, origin=0.5):
    """Generate relative box corners based on length per dim and origin point.

    Args:
        dims (np.ndarray, shape=[N, ndim]): Array of length per dim
        origin (list or array or float, optional): origin point relate to
            smallest point. Defaults to 0.5

    Returns:
        np.ndarray, shape=[N, 2 ** ndim, ndim]: Returned corners.
        point layout example: (2d) x0y0, x0y1, x1y0, x1y1;
            (3d) x0y0z0, x0y0z1, x0y1z0, x0y1z1, x1y0z0, x1y0z1, x1y1z0, x1y1z1
            where x0 < x1, y0 < y1, z0 < z1.
    """
    ndim = int(dims.shape[1])
    corners_norm = np.stack(np.unravel_index(np.arange(2**ndim), [2] * ndim), axis=1)
    # now corners_norm has format: (2d) x0y0, x0y1, x1y0, x1y1
    # (3d) x0y0z0, x0y0z1, x0y1z0, x0y1z1, x1y0z0, x1y0z1, x1y1z0, x1y1z1
    # so need to convert to a format which is convenient to do other computing.
    # for 2d boxes, format is clockwise start with minimum point
    # for 3d boxes, please draw lines by your hand.
    if ndim == 2:
        # generate clockwise box corners
        corners_norm = corners_norm[[0, 1, 3, 2]]
    elif ndim == 3:
        corners_norm = corners_norm[[0, 1, 3, 2, 4, 5, 7, 6]]
    corners_norm = corners_norm - np.array(origin)
    corners = dims.reshape([-1, 1, ndim]) * corners_norm.reshape([1, 2**ndim, ndim])
    return corners


def rotation_2d_reverse(points, angles):
    """Rotation 2d points based on origin point clockwise when angle positive.

    Args:
        points (np.ndarray): Points to be rotated with shape \
            (N, point_size, 2).
        angles (np.ndarray): Rotation angle with shape (N).

    Returns:
        np.ndarray: Same shape as points.
    """
    rot_sin = np.sin(angles)
    rot_cos = np.cos(angles)
    rot_mat_T = np.stack([[rot_cos, rot_sin], [-rot_sin, rot_cos]])
    return np.einsum("aij,jka->aik", points, rot_mat_T)

def center_to_corner_box2d(centers, dims, angles=None, origin=0.5):
    """Convert kitti locations, dimensions and angles to corners.
    format: center(xy), dims(xy), angles(clockwise when positive)

    Args:
        centers (np.ndarray): Locations in kitti label file with shape (N, 2).
        dims (np.ndarray): Dimensions in kitti label file with shape (N, 2).
        angles (np.ndarray, optional): Rotation_y in kitti label file with
            shape (N). Defaults to None.
        origin (list or array or float, optional): origin point relate to
            smallest point. Defaults to 0.5.

    Returns:
        np.ndarray: Corners with the shape of (N, 4, 2).
    """
    # 'length' in kitti format is in x axis.
    # xyz(hwl)(kitti label file)<->xyz(lhw)(camera)<->z(-x)(-y)(wlh)(lidar)
    # center in kitti format is [0.5, 1.0, 0.5] in xyz.
    corners = corners_nd(dims, origin=origin)
    # corners: [N, 4, 2]
    if angles is not None:
        corners = rotation_2d_reverse(corners, angles)
    corners += centers.reshape([-1, 1, 2])
    corners = torch.from_numpy(corners)
    return corners

def rel_loss(feature_stu_sample, feature_teach_sample, \
             gt_boxes_bev_coords_shape, gt_boxes_indices,\
             stu_rel='ss', teach_rel='tt', rel_mode='cosim'):
    """gt_boxes_sample_stu: n_sample, 9, c
       gt_boxes_sample_teach: n_sample, 9, c
       stu_rel: ss -relation of student and student; 
                st -relation of student and teacher; 
       teach_rel: tt - relation of teacher and teacher; 
                  ts -relation of teacher and student; 
       rel_mode: cosim or mse
    """
    
    criterion = nn.L1Loss(reduce=False)
    weight = gt_boxes_indices.float().sum()
    weight = reduce_mean(weight)
    
    
    gt_boxes_sample_stu = feature_stu_sample.contiguous().view(
        -1, feature_stu_sample.shape[-2], feature_stu_sample.shape[-1]
    )
    gt_boxes_sample_teach = feature_teach_sample.contiguous().view(
        -1, feature_teach_sample.shape[-2], feature_teach_sample.shape[-1]
    )

    if stu_rel == 'ss':
        gt_boxes_sample_stu_b = gt_boxes_sample_stu
    elif stu_rel == 'st':
        gt_boxes_sample_stu_b = gt_boxes_sample_teach
        
    if teach_rel == 'tt':
        gt_boxes_sample_teach_b = gt_boxes_sample_teach
    elif teach_rel == 'ts':
        gt_boxes_sample_teach_b = gt_boxes_sample_stu
    
    
    if rel_mode == 'cosim':
        """余弦相似度一致"""
        gt_boxes_sample_stu = gt_boxes_sample_stu / (
            torch.norm(gt_boxes_sample_stu, dim=-1, keepdim=True) + 1e-4
        )
        gt_boxes_sample_stu_b = gt_boxes_sample_stu_b / (
            torch.norm(gt_boxes_sample_stu_b, dim=-1, keepdim=True) + 1e-4
        )
        
        gt_boxes_sample_teach = gt_boxes_sample_teach / (
            torch.norm(gt_boxes_sample_teach, dim=-1, keepdim=True) + 1e-4
        )
        gt_boxes_sample_teach_b = gt_boxes_sample_teach_b / (
            torch.norm(gt_boxes_sample_teach_b, dim=-1, keepdim=True) + 1e-4
        )
        
        gt_boxes_stu_rel = torch.bmm(
            gt_boxes_sample_stu,
            torch.transpose(gt_boxes_sample_stu_b, 1, 2),
        )
        gt_boxes_teach_rel = torch.bmm(
            gt_boxes_sample_teach,
            torch.transpose(gt_boxes_sample_teach_b, 1, 2),
        ) 
    elif rel_mode == 'mse':
        n_sample = gt_boxes_sample_stu.shape[1]
        gt_boxes_stu_rel = torch.norm(
            gt_boxes_sample_stu.unsqueeze(1).expand(-1, n_sample, -1, -1) - 
            gt_boxes_sample_stu_b.unsqueeze(2).expand(-1, -1, n_sample, -1),
            dim = -1
        )
        gt_boxes_teach_rel = torch.norm(
            gt_boxes_sample_teach.unsqueeze(1).expand(-1, n_sample, -1, -1) - 
            gt_boxes_sample_teach_b.unsqueeze(2).expand(-1, -1, n_sample, -1),
            dim = -1
        )
        
        
    gt_boxes_stu_rel = gt_boxes_stu_rel.contiguous().view(
        gt_boxes_bev_coords_shape[0],
        gt_boxes_bev_coords_shape[1],
        gt_boxes_stu_rel.shape[-2],
        gt_boxes_stu_rel.shape[-1],
    )
    gt_boxes_teach_rel = gt_boxes_teach_rel.contiguous().view(
        gt_boxes_bev_coords_shape[0],
        gt_boxes_bev_coords_shape[1],
        gt_boxes_teach_rel.shape[-2],
        gt_boxes_teach_rel.shape[-1],
    )
    
    # criterion = nn.L1Loss(reduce=False)
    """相似度需要一致"""
    loss_rel = criterion(
        gt_boxes_stu_rel[gt_boxes_indices], gt_boxes_teach_rel[gt_boxes_indices]
    )
    
    loss_rel = torch.mean(loss_rel, 2)
    loss_rel = torch.mean(loss_rel, 1)
    loss_rel = torch.sum(loss_rel)
    loss_rel = loss_rel / (weight + 1e-4)
    
    return loss_rel
    

def BEVDistillLoss(bev_stu, bev_teach, gt_boxes_bev_coords, gt_boxes_indices):
    """
    bev_stu: Student Feature.
    bev_teach: Teacher Feature.
    """
    
    h, w = bev_stu.shape[-2:]
    gt_boxes_bev_center = torch.mean(gt_boxes_bev_coords, dim=2).unsqueeze(2)
    gt_boxes_bev_edge_1 = torch.mean(
        gt_boxes_bev_coords[:, :, [0, 1], :], dim=2
    ).unsqueeze(2)
    gt_boxes_bev_edge_2 = torch.mean(
        gt_boxes_bev_coords[:, :, [1, 2], :], dim=2
    ).unsqueeze(2)
    gt_boxes_bev_edge_3 = torch.mean(
        gt_boxes_bev_coords[:, :, [2, 3], :], dim=2
    ).unsqueeze(2)
    gt_boxes_bev_edge_4 = torch.mean(
        gt_boxes_bev_coords[:, :, [0, 3], :], dim=2
    ).unsqueeze(2)
    gt_boxes_bev_all = torch.cat(
        (
            gt_boxes_bev_coords,
            gt_boxes_bev_center,
            gt_boxes_bev_edge_1,
            gt_boxes_bev_edge_2,
            gt_boxes_bev_edge_3,
            gt_boxes_bev_edge_4,
        ),
        dim=2,
    )
    gt_boxes_bev_all[:, :, :, 0] = (gt_boxes_bev_all[:, :, :, 0] - w / 2) / (w / 2)
    gt_boxes_bev_all[:, :, :, 1] = (gt_boxes_bev_all[:, :, :, 1] - h / 2) / (h / 2)
    gt_boxes_bev_all[:, :, :, [0, 1]] = gt_boxes_bev_all[:, :, :, [1, 0]]
    feature_stu_sample = torch.nn.functional.grid_sample(bev_stu, gt_boxes_bev_all)
    feature_stu_sample = feature_stu_sample.permute(0, 2, 3, 1)
    feature_teach_sample = torch.nn.functional.grid_sample(bev_teach, gt_boxes_bev_all)
    feature_teach_sample = feature_teach_sample.permute(0, 2, 3, 1)
   
    # feature_fuse_sample_show = feature_stu_sample.permute(0, 1, 3, 2).reshape(-1, feature_stu_sample.shape[1], 64, 3, 3)
    # for n in range(feature_fuse_sample_show.shape[1]):
    #     feature_show(feature_fuse_sample_show[0][n], f'sample_show/sample_stru_{n}.png')
    # feature_show()
    
    
    # loss_tt_ss = rel_loss(feature_stu_sample, feature_teach_sample, gt_boxes_bev_coords.shape, gt_boxes_indices,\
    #                         stu_rel='ss', teach_rel='tt', rel_mode='cosim') + \
    #              rel_loss(feature_stu_sample, feature_teach_sample, gt_boxes_bev_coords.shape, gt_boxes_indices,\
    #                         stu_rel='ss', teach_rel='tt', rel_mode='mse')
                 
    # loss_tt_st = rel_loss(feature_stu_sample, feature_teach_sample, gt_boxes_bev_coords.shape, gt_boxes_indices,\
    #                     stu_rel='st', teach_rel='tt', rel_mode='cosim') + \
    #             rel_loss(feature_stu_sample, feature_teach_sample, gt_boxes_bev_coords.shape, gt_boxes_indices,\
    #                     stu_rel='st', teach_rel='tt', rel_mode='mse')
    # loss_distill = loss_tt_ss + loss_tt_st


    # loss_tt_ss = rel_loss(feature_stu_sample, feature_teach_sample, gt_boxes_bev_coords.shape, gt_boxes_indices,\
    #                         stu_rel='ss', teach_rel='tt', rel_mode='mse')
    
    
    loss_tt_ss = rel_loss(feature_stu_sample, feature_teach_sample, gt_boxes_bev_coords.shape, gt_boxes_indices,\
                            stu_rel='ss', teach_rel='tt', rel_mode='cosim')
    loss_tt_st = rel_loss(feature_stu_sample, feature_teach_sample, gt_boxes_bev_coords.shape, gt_boxes_indices,\
                        stu_rel='st', teach_rel='tt', rel_mode='cosim')
    loss_distill = loss_tt_ss + loss_tt_st
    
    return loss_distill


def FeatureDistillLoss(
    feature_lidar, feature_fuse, gt_boxes_bev_coords, gt_boxes_indices
):
    h, w = feature_lidar.shape[-2:]
    gt_boxes_bev_center = torch.mean(gt_boxes_bev_coords, dim=2).unsqueeze(2)
    gt_boxes_bev_edge_1 = torch.mean(
        gt_boxes_bev_coords[:, :, [0, 1], :], dim=2
    ).unsqueeze(2)
    gt_boxes_bev_edge_2 = torch.mean(
        gt_boxes_bev_coords[:, :, [1, 2], :], dim=2
    ).unsqueeze(2)
    gt_boxes_bev_edge_3 = torch.mean(
        gt_boxes_bev_coords[:, :, [2, 3], :], dim=2
    ).unsqueeze(2)
    gt_boxes_bev_edge_4 = torch.mean(
        gt_boxes_bev_coords[:, :, [0, 3], :], dim=2
    ).unsqueeze(2)
    gt_boxes_bev_all = torch.cat(
        (
            gt_boxes_bev_coords,
            gt_boxes_bev_center,
            gt_boxes_bev_edge_1,
            gt_boxes_bev_edge_2,
            gt_boxes_bev_edge_3,
            gt_boxes_bev_edge_4,
        ),
        dim=2,
    )
    gt_boxes_bev_all[:, :, :, 0] = (gt_boxes_bev_all[:, :, :, 0] - w / 2) / (w / 2)
    gt_boxes_bev_all[:, :, :, 1] = (gt_boxes_bev_all[:, :, :, 1] - h / 2) / (h / 2)
    gt_boxes_bev_all[:, :, :, [0, 1]] = gt_boxes_bev_all[:, :, :, [1, 0]]
    feature_lidar_sample = torch.nn.functional.grid_sample(
        feature_lidar, gt_boxes_bev_all
    )
    feature_lidar_sample = feature_lidar_sample.permute(0, 2, 3, 1)
    feature_fuse_sample = torch.nn.functional.grid_sample(
        feature_fuse, gt_boxes_bev_all
    )
    feature_fuse_sample = feature_fuse_sample.permute(0, 2, 3, 1)


    # feature_fuse_sample_show = feature_lidar_sample.permute(0, 1, 3, 2).reshape(-1, feature_lidar_sample.shape[1], 64, 3, 3)
    # for n in range(feature_fuse_sample_show.shape[1]):
    #     feature_show(feature_fuse_sample_show[0][n], f'sample_show/sample_sem_{n}.png')

    criterion = nn.L1Loss(reduce=False)
    loss_feature_distill = criterion(
        feature_lidar_sample[gt_boxes_indices], feature_fuse_sample[gt_boxes_indices]
    )
    loss_feature_distill = torch.mean(loss_feature_distill, 2)
    loss_feature_distill = torch.mean(loss_feature_distill, 1)
    loss_feature_distill = torch.sum(loss_feature_distill)
    weight = gt_boxes_indices.float().sum()
    weight = reduce_mean(weight)
    loss_feature_distill = loss_feature_distill / (weight + 1e-4)
    return loss_feature_distill


class SemDistillLoss(nn.Module):
    def __init__(self, args):
        super().__init__()
        
        self.cav_lidar_range = args['cav_lidar']
        self.voxel_size = args['voxel_size']
        self.feature_stride = args['feature_stride']
    
    def forward(self, bev_stu, bev_teach, target_dict_single):
        object_bbx_center = target_dict_single['object_bbx_center_single']
        object_bbx_mask = target_dict_single['object_bbx_mask_single']
        
        # gt_boxes = object_bbx_center
        
        batch_uni_samp_mask = torch.any(object_bbx_mask==1, dim=0)
        gt_boxes = object_bbx_center[:, batch_uni_samp_mask]
        
        
        # gt_labels = target_dict_single
        
        gt_boxes_indice = torch.zeros((gt_boxes.shape[0], gt_boxes.shape[1]))
        for i in range(gt_boxes.shape[0]):
            cnt = gt_boxes[i].__len__() - 1
            while cnt > 0 and gt_boxes[i][cnt].sum() == 0:
                cnt -= 1
            gt_boxes_indice[i][: cnt + 1] = 1
        gt_boxes_indice = gt_boxes_indice.bool()
        
        
        # gt_labels += 1  看起来是把类别也作为一个特征参与计算, 但opencood里只有一个类别, 这一步可以略去?
        # gt_boxes = torch.cat([gt_boxes, gt_labels.unsqueeze(dim=2)], dim=2)

        gt_boxes_bev_coords = torch.zeros((gt_boxes.shape[0], gt_boxes.shape[1], 4, 2))
        for i in range(gt_boxes.shape[0]):
            gt_boxes_tmp = gt_boxes[i]
            gt_boxes_tmp_bev = center_to_corner_box2d(
                gt_boxes_tmp[:, :2].cpu().detach().numpy(),
                gt_boxes_tmp[:, 3:5].cpu().detach().numpy(),
                gt_boxes_tmp[:, 6].cpu().detach().numpy(),
                origin=(0.5, 0.5),
            )
            gt_boxes_bev_coords[i] = gt_boxes_tmp_bev
            
        # 将bbox的坐标映射到特征上
        gt_boxes_bev_coords = gt_boxes_bev_coords.cuda()
        gt_boxes_indice = gt_boxes_indice.cuda()
        gt_boxes_bev_coords[:, :, :, 0] = (
            gt_boxes_bev_coords[:, :, :, 0] - self.cav_lidar_range[0]
        ) / (self.voxel_size[0] * self.feature_stride)
        gt_boxes_bev_coords[:, :, :, 1] = (
            gt_boxes_bev_coords[:, :, :, 1] - self.cav_lidar_range[1]
        ) / (self.voxel_size[1] * self.feature_stride)
        
        
        # points_tensor = gt_boxes_bev_coords[0].reshape(-1, 2)
        # image_tensor = bev_stu[0].mean(dim=0)
        # image_pil = to_pil_image(image_tensor)
        
        # # 创建一个matplotlib图形和一个轴
        # fig, ax = plt.subplots()
        # # 显示图像
        # ax.imshow(image_pil, cmap='viridis')
        # # 从张量中提取点的x和y坐标
        # x = points_tensor[:, 0].cpu()
        # y = points_tensor[:, 1].cpu()

        # # 在图像上绘制点，确保点的大小足够大
        # ax.scatter(x, y, s=1, color='red')  # s参数控制点的大小

        # # 设置坐标轴的范围与图像像素对应
        # ax.set_xlim(0, image_tensor.size(1))
        # ax.set_ylim(image_tensor.size(0), 0)  # 注意y轴是反向的
        # plt.axis('off')
        # plt.Normalize(vmin=0, vmax=2)
        # plt.savefig('sample_show/points_in_feat.png', bbox_inches='tight', pad_inches = -0.1)
        # plt.close()
        
        loss_sem = FeatureDistillLoss(bev_stu, bev_teach, gt_boxes_bev_coords, gt_boxes_indice)
        
        
        return loss_sem

class StruDistillLoss(nn.Module):
    def __init__(self, args):
        super().__init__()
        
        self.cav_lidar_range = args['cav_lidar']
        self.voxel_size = args['voxel_size']
        self.feature_stride = args['feature_stride']
    
    def forward(self, bev_stu, bev_teach, target_dict_single):
        object_bbx_center = target_dict_single['object_bbx_center_single']
        object_bbx_mask = target_dict_single['object_bbx_mask_single']
        
        # gt_boxes = object_bbx_center
        
        batch_uni_samp_mask = torch.any(object_bbx_mask==1, dim=0)
        gt_boxes = object_bbx_center[:, batch_uni_samp_mask]
        
        
        # gt_labels = target_dict_single
        
        gt_boxes_indice = torch.zeros((gt_boxes.shape[0], gt_boxes.shape[1]))
        for i in range(gt_boxes.shape[0]):
            cnt = gt_boxes[i].__len__() - 1
            while cnt > 0 and gt_boxes[i][cnt].sum() == 0:
                cnt -= 1
            gt_boxes_indice[i][: cnt + 1] = 1
        gt_boxes_indice = gt_boxes_indice.bool()
        
        
        # gt_labels += 1  看起来是把类别也作为一个特征参与计算, 但opencood里只有一个类别, 这一步可以略去?
        # gt_boxes = torch.cat([gt_boxes, gt_labels.unsqueeze(dim=2)], dim=2)

        gt_boxes_bev_coords = torch.zeros((gt_boxes.shape[0], gt_boxes.shape[1], 4, 2))
        for i in range(gt_boxes.shape[0]):
            gt_boxes_tmp = gt_boxes[i]
            gt_boxes_tmp_bev = center_to_corner_box2d(
                gt_boxes_tmp[:, :2].cpu().detach().numpy(),
                gt_boxes_tmp[:, 3:5].cpu().detach().numpy(),
                gt_boxes_tmp[:, 6].cpu().detach().numpy(),
                origin=(0.5, 0.5),
            )
            gt_boxes_bev_coords[i] = gt_boxes_tmp_bev
            
        # 将bbox的坐标映射到特征上
        gt_boxes_bev_coords = gt_boxes_bev_coords.cuda()
        gt_boxes_indice = gt_boxes_indice.cuda()
        gt_boxes_bev_coords[:, :, :, 0] = (
            gt_boxes_bev_coords[:, :, :, 0] - self.cav_lidar_range[0]
        ) / (self.voxel_size[0] * self.feature_stride)
        gt_boxes_bev_coords[:, :, :, 1] = (
            gt_boxes_bev_coords[:, :, :, 1] - self.cav_lidar_range[1]
        ) / (self.voxel_size[1] * self.feature_stride)
        
        
        # points_tensor = gt_boxes_bev_coords[0].reshape(-1, 2)
        # image_tensor = bev_stu[0].mean(dim=0)
        # image_pil = to_pil_image(image_tensor)

        # # 创建一个matplotlib图形和一个轴
        # fig, ax = plt.subplots()
        # # 显示图像
        # ax.imshow(image_pil, cmap='viridis')
        # # 从张量中提取点的x和y坐标
        # x = points_tensor[:, 0].cpu()
        # y = points_tensor[:, 1].cpu()

        # # 在图像上绘制点，确保点的大小足够大
        # ax.scatter(x, y, s=2, color='red')  # s参数控制点的大小

        # # 设置坐标轴的范围与图像像素对应
        # ax.set_xlim(0, image_tensor.size(1))
        # ax.set_ylim(image_tensor.size(0), 0)  # 注意y轴是反向的
        # plt.axis('off')
        # plt.Normalize(vmin=0, vmax=2)
        # plt.savefig('sample_show/points_in_feat.png', bbox_inches='tight', pad_inches = -0.1)
        
        loss_stru = BEVDistillLoss(bev_stu, bev_teach, gt_boxes_bev_coords, gt_boxes_indice)
        
        
        return loss_stru