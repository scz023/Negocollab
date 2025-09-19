import torch
import torch.nn as nn



import torch

import torch
import torch.nn as nn

class CrossEntropyDistance(nn.Module):
    def __init__(self):
        super(CrossEntropyDistance, self).__init__()
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, target, preds):           
        
        # print(target.size())
        
        target = torch.softmax(target, dim=1)
        preds = torch.softmax(preds, dim=1)

        # 计算交叉熵损失，不进行归约
        # b c h w -> b h w
        cross_entropy = self.cross_entropy_loss(preds, target)

        # print(cross_entropy.size())

        # 在高和宽维度上求平均，再在批次维度上求平均
        cross_entropy = cross_entropy.mean(dim=(1, 2)).mean()

        return cross_entropy
    


class KLDivergence(nn.Module):
    def __init__(self):
        super(KLDivergence, self).__init__()
        self.kl_loss = nn.KLDivLoss(reduction='none')

    def forward(self, target, preds):
       
        # 确保输入是概率分布，使用 softmax 函数
        # target = torch.softmax(target, dim=1)
        # preds = torch.softmax(preds, dim=1)

        # 计算 log_softmax
        preds_log = torch.log_softmax(preds, dim=1)

        # 计算 KL 散度，不进行归约
        kl_div = self.kl_loss(preds_log, target)

        # 在高和宽维度上求平均，在通道上求和, 在批次维度上求平均
        kl_div = kl_div.mean(dim=(2, 3)).mean(1).mean(0)

        return kl_div
    


class MMD(nn.Module):
    def __init__(self, args):
        super(MMD, self).__init__()
        kernel_mul = 2.0
        kernel_num = 5
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        return
    
    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[1])+int(target.size()[1])
        total = torch.cat([source, target], dim=1)

        total0 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(1)), int(total.size(1)), int(total.size(2)))
        total1 = total.unsqueeze(2).expand(int(total.size(0)), int(total.size(1)), int(total.size(1)), int(total.size(2)))
        L2_distance = ((total0-total1)**2).sum(3) 
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def forward(self, source, target):
        source = source.flatten(2).permute(0, 2, 1)
        target = target.flatten(2).permute(0, 2, 1)
        batch_size = int(source.size()[1])
        kernels = self.guassian_kernel(source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
        XX = kernels[:,:batch_size, :,:batch_size]
        YY = kernels[:,batch_size:, :,batch_size:]
        XY = kernels[:,:batch_size, :,batch_size:]
        YX = kernels[:,batch_size:, :,:batch_size]
        loss = torch.mean((XX + YY - XY -YX),dim=1)
        loss = loss.mean() 
        return loss

class EMD(nn.Module):
    def __init__(self, args):
        """
        Wasserstein 距离模块化版本

        参数:
            reduction (str): 'mean' 返回平均距离, 'none' 返回每个样本的距离
        """
        super(EMD, self).__init__()
        reduction = 'mean'
        assert reduction in ['mean', 'none'], "reduction 只能是 'mean' 或 'none'"
        self.reduction = reduction

    def forward(self, input1, input2):
        """
        计算两个 (B, C, H, W) 张量之间的 Wasserstein 距离

        返回:
            如果 reduction == 'mean': 返回单个标量
            如果 reduction == 'none': 返回 (B,) 张量
        """
        B = input1.size(0)
        assert input1.shape == input2.shape, "两个输入的尺寸必须相同"

        input1_flat = input1.contiguous().view(B, -1)
        # print(input1.size())
        # print(input2.size())
        input2_flat = input2.contiguous().view(B, -1)

        input1_sorted, _ = torch.sort(input1_flat, dim=1)
        input2_sorted, _ = torch.sort(input2_flat, dim=1)

        distances = torch.mean(torch.abs(input1_sorted - input2_sorted), dim=1)  # shape: (B,)

        if self.reduction == 'mean':
            return distances.mean()
        else:
            return distances



# Adapted from https://github.com/gpeyre/SinkhornAutoDiff
# class SinkhornDistance(nn.Module):
class EMDSinkhornDistance(nn.Module):
    r"""
    Given two empirical measures each with :math:`P_1` locations
    :math:`x\in\mathbb{R}^{D_1}` and :math:`P_2` locations :math:`y\in\mathbb{R}^{D_2}`,
    outputs an approximation of the regularized OT cost for point clouds.

    Args:
        eps (float): regularization coefficient
        max_iter (int): maximum number of Sinkhorn iterations
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of
            elements in the output, 'sum': the output will be summed. Default: 'none'

    Shape:
        - Input: :math:`(N, P_1, D_1)`, :math:`(N, P_2, D_2)`
        - Output: :math:`(N)` or :math:`()`, depending on `reduction`
    """
    def __init__(self, args):
        super(EMDSinkhornDistance, self).__init__()
        self.eps = args['eps']
        self.max_iter = args['max_iter']
        self.reduction = args['reduction']

    def forward(self, x, y):
        """
        x: b c h w
        y: b c h w
        """
        
        # b c h w -> b c h*w -> b h*w c 
        x = x.flatten(2).permute(0, 2, 1)
        y = y.flatten(2).permute(0, 2, 1)
        
        # The Sinkhorn algorithm takes as input three variables :
        C = self._cost_matrix(x, y)  # Wasserstein cost function
        x_points = x.shape[-2]
        y_points = y.shape[-2]
        if x.dim() == 2:
            batch_size = 1
        else:
            batch_size = x.shape[0]

        # both marginals are fixed with equal weights
        mu = torch.empty(batch_size, x_points, dtype=torch.float,
                         requires_grad=False).fill_(1.0 / x_points).squeeze()
        nu = torch.empty(batch_size, y_points, dtype=torch.float,
                         requires_grad=False).fill_(1.0 / y_points).squeeze()

        u = torch.zeros_like(mu)
        v = torch.zeros_like(nu)
        # To check if algorithm terminates because of threshold
        # or max iterations reached
        actual_nits = 0
        # Stopping criterion
        thresh = 1e-1

        # Sinkhorn iterations
        for i in range(self.max_iter):
            u1 = u  # useful to check the update
            u = self.eps * (torch.log(mu+1e-8) - torch.logsumexp(self.M(C, u, v), dim=-1)) + u
            v = self.eps * (torch.log(nu+1e-8) - torch.logsumexp(self.M(C, u, v).transpose(-2, -1), dim=-1)) + v
            err = (u - u1).abs().sum(-1).mean()

            actual_nits += 1
            if err.item() < thresh:
                break

        U, V = u, v
        # Transport plan pi = diag(a)*K*diag(b)
        pi = torch.exp(self.M(C, U, V))
        # Sinkhorn distance
        cost = torch.sum(pi * C, dim=(-2, -1))

        if self.reduction == 'mean':
            cost = cost.mean()
        elif self.reduction == 'sum':
            cost = cost.sum()

        return cost, pi, C

    def M(self, C, u, v):
        "Modified cost for logarithmic updates"
        "$M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$"
        return (-C + u.unsqueeze(-1) + v.unsqueeze(-2)) / self.eps

    @staticmethod
    def _cost_matrix(x, y, p=2):
        "Returns the matrix of $|x_i-y_j|^p$."
        x_col = x.unsqueeze(-2)
        y_lin = y.unsqueeze(-3)
        C = torch.sum((torch.abs(x_col - y_lin)) ** p, -1)
        return C

    @staticmethod
    def ave(u, u1, tau):
        "Barycenter subroutine, used by kinetic acceleration through extrapolation."
        return tau * u + (1 - tau) * u1