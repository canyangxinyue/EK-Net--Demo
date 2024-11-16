
import torch
from torch import nn

from models.losses.basic_loss import BalanceCrossEntropyLoss, MaskL1Loss, DiceLoss


class EK_loss(nn.Module):
    def __init__(self, alpha=1.0, beta=10, gamma=10, ohem_ratio=3, reduction='mean', eps=1e-6):
        """
        Implement PSE Loss.
        :param alpha: binary_map loss 前面的系数
        :param beta: threshold_map loss 前面的系数
        :param ohem_ratio: OHEM的比例
        :param reduction: 'mean' or 'sum'对 batch里的loss 算均值或求和
        """
        super().__init__()
        assert reduction in ['mean', 'sum'], " reduction must in ['mean','sum']"
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.bce_loss = BalanceCrossEntropyLoss(negative_ratio=ohem_ratio)
        self.dice_loss = DiceLoss(eps=eps)
        self.l1_loss = MaskL1Loss(eps=eps)
        self.loc_loss = SmoothL1Loss(beta=0.1,loss_weight=0.05)
        self.ohem_ratio = ohem_ratio
        self.reduction = reduction

    def forward(self, pred, batch):
        shrink_maps = pred[:, 0, :, :]
        threshold_maps = pred[:, 1, :, :]
        binary_maps = pred[:, 2, :, :]
        distances = pred[:, 3:5, :, :]
        
        threshold_ratio= batch['threshold_mask_ratio'].sum()
        shrink_ratio= batch['shrink_mask_ratio'].sum()
        normalize = 1 / (threshold_ratio + shrink_ratio)
        
        shrink_maps_weight = (threshold_ratio * normalize).detach()
        threshold_maps_weight = (shrink_ratio * normalize).detach()


        loss_shrink_maps = self.bce_loss(shrink_maps, batch['shrink_map'], batch['shrink_mask'])
        loss_threshold_maps = self.l1_loss(threshold_maps, batch['threshold_map'], batch['threshold_mask'])
        loss_loc, _ = self.loc_loss(distances, batch['gt_instances'], batch['gt_kernel_instances'], batch['training_mask_distances']\
                                , batch['gt_distances'], reduce=False)
        loss_loc = torch.mean(loss_loc)
        metrics = dict(loss_shrink_maps=loss_shrink_maps, loss_threshold_maps=loss_threshold_maps,loss_loc=loss_loc)

        loss_binary_maps = self.dice_loss(binary_maps, batch['shrink_map'], batch['shrink_mask'])
        metrics['loss_binary_maps'] = loss_binary_maps
        loss_all =  self.alpha * shrink_maps_weight *loss_shrink_maps +\
                    self.beta * threshold_maps_weight * loss_threshold_maps +\
                    shrink_maps_weight * loss_binary_maps +\
                    self.gamma * threshold_maps_weight * loss_loc
        metrics['loss'] = loss_all
        
        return metrics


class SmoothL1Loss(nn.Module):
    def __init__(self, beta=1.0, loss_weight=1.0):
        super(SmoothL1Loss, self).__init__()
        self.beta = beta
        self.loss_weight = loss_weight

        self.coord = nn.Parameter(torch.zeros([640, 640, 2]).long(), requires_grad=False)
        for i in range(640):
            for j in range(640):
                self.coord[i, j, 0] = j
                self.coord[i, j, 1] = i
        self.coord.data = self.coord.view(-1, 2) # (h*w, 2)

    def forward_single(self, input, target, mask, beta=1.0, eps=1e-6):
        batch_size = input.size(0)

        diff = torch.abs(input - target) * mask.unsqueeze(1)
        loss = torch.where(diff < beta, 0.5 * diff * diff / beta,
                           diff - 0.5 * beta)
        loss = loss.contiguous().view(batch_size, -1).float()
        mask = mask.contiguous().view(batch_size, -1).float()
        loss = torch.sum(loss, dim=-1)
        loss = loss / (mask.sum(dim=-1) + eps)

        return loss

    def select_single(self, distance, gt_instance, gt_kernel_instance, training_mask):

        with torch.no_grad():
            off_points = (self.coord.float() + 10 * distance[:, self.coord[:, 1], self.coord[:, 0]].transpose(1, 0)).long() # (h*w, 2)
            off_points = torch.clamp(off_points, 0, distance.size(-1) - 1)
            selected_mask = (gt_instance[self.coord[:, 1], self.coord[:, 0]] != gt_kernel_instance[off_points[:, 1], off_points[:, 0]])
            selected_mask = selected_mask.contiguous().view(1, -1, distance.shape[-1]).long()
            selected_training_mask = selected_mask * training_mask

            return selected_training_mask

    def forward(self, distances, gt_instances, gt_kernel_instances, training_masks, gt_distances, reduce=True):

        selected_training_masks = []
        for i in range(distances.size(0)):
            selected_training_masks.append(
                self.select_single(distances[i, :, :, :], gt_instances[i, :, :],
                                    gt_kernel_instances[i, :, :], training_masks[i, :, :])
            )
        selected_training_masks = torch.cat(selected_training_masks, 0).float()

        loss = self.forward_single(distances, gt_distances, selected_training_masks, self.beta)
        loss = self.loss_weight * loss

        with torch.no_grad():
            batch_size = distances.size(0)
            false_num = selected_training_masks.contiguous().view(batch_size, -1)
            false_num = false_num.sum(dim=-1)
            total_num = training_masks.contiguous().view(batch_size, -1).float()
            total_num = total_num.sum(dim=-1)
            iou_text = (total_num - false_num) / (total_num + 1e-6)

        if reduce:
            loss = torch.mean(loss)

        return loss, iou_text
