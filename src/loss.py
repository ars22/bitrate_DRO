import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import joint_dro
from src.utils import get_lp_norm


class LossComputer:
    def __init__(
        self,
        criterion,
        loss_type,
        dataset,
        alpha=None,
        gamma=0.1,
        adj=None,
        min_var_weight=0,
        step_size=0.01,
        normalize_loss=False,
        btl=False,
        joint_dro_alpha=None,
        lambd=0.05, 
        eta=0.05,
        bitrate_dro_adversary=None
    ):
        assert loss_type in ["group_dro", "erm", "joint_dro", "bitrate_dro"]

        self.criterion = criterion
        self.loss_type = loss_type
        self.gamma = gamma
        self.alpha = alpha
        self.min_var_weight = min_var_weight
        self.step_size = step_size
        self.normalize_loss = normalize_loss
        self.btl = btl
        self.lambd = lambd # l1 penalty for bitrate DRO
        self.eta = eta # learning rate for bitrate DRO
        self.bitrate_dro_adversary = bitrate_dro_adversary # adv for bitrate dro
        self.n_groups = dataset.n_groups
        
        if self.loss_type == 'bitrate_dro':
            self.opt_bitrate_dro_adversary = torch.optim.Adam(
                self.bitrate_dro_adversary.parameters(), lr=self.eta)

        self.group_counts = dataset.group_counts().cuda()
        self.group_frac = self.group_counts / self.group_counts.sum()
        self.group_str = dataset.group_str

        if self.loss_type == "joint_dro":
            # Joint DRO reg should be 0.
            assert joint_dro_alpha is not None
            self._joint_dro_loss_computer = joint_dro.RobustLoss(
                    joint_dro_alpha, 0, "cvar")

        if adj is not None:
            self.adj = torch.from_numpy(adj).float().cuda()
        else:
            self.adj = torch.zeros(self.n_groups).float().cuda()

        if loss_type == "group_dro":
            assert alpha, "alpha must be specified"

        # quantities maintained throughout training
        self.adv_probs = torch.ones(self.n_groups).cuda() / self.n_groups
        self.exp_avg_loss = torch.zeros(self.n_groups).cuda()
        self.exp_avg_initialized = torch.zeros(self.n_groups).byte().cuda()

        self.reset_stats()

    def loss(self, yhat, y, group_idx=None, is_training=False, bitrate_dro_adversary_inputs=None):
        # compute per-sample and per-group losses
        per_sample_losses = self.criterion(yhat, y)
        group_loss, group_count = self.compute_group_avg(
            per_sample_losses, group_idx)
        group_acc, group_count = self.compute_group_avg(
            (torch.argmax(yhat, 1) == y).float(), group_idx)

        # update historical losses
        self.update_exp_avg_loss(group_loss, group_count)

        # compute overall loss
        if self.loss_type == "group_dro":
            if not self.btl:
                actual_loss, weights = self.compute_robust_loss(
                    group_loss, group_count)
            else:
                actual_loss, weights = self.compute_robust_loss_btl(
                    group_loss, group_count)
        elif self.loss_type == "joint_dro":
            actual_loss = self._joint_dro_loss_computer(per_sample_losses)
            weights = None
        elif self.loss_type == "bitrate_dro":
            if is_training:
                self.bitrate_dro_adversary.train()
            else:
                self.bitrate_dro_adversary.eval()
            actual_loss, weights = self.compute_bitrate_dro_loss(
                bitrate_dro_adversary_inputs=bitrate_dro_adversary_inputs,
                group_idx = group_idx,
                targets=y,
                per_sample_losses=per_sample_losses,
                is_training=is_training,
                criterion=torch.nn.BCELoss(reduction='none'))
        else:
            assert self.loss_type == "erm"

            actual_loss = per_sample_losses.mean()
            weights = None

        # update stats
        self.update_stats(actual_loss, group_loss, group_acc, group_count,
                          weights)

        if self.loss_type == 'bitrate_dro':
            return actual_loss, weights
        else:
            return actual_loss
    
    
    
    
    
    def compute_bitrate_dro_loss(self, bitrate_dro_adversary_inputs, targets, group_idx, per_sample_losses, criterion, is_training):
        gamma = torch.quantile(per_sample_losses, 0.95).item()
        adv_targets = (per_sample_losses > gamma).float().detach()
        cwts = torch.ones_like(per_sample_losses)
        cwts[adv_targets > 0.5] = len(adv_targets) / sum(adv_targets > 0.5)
        cwts[adv_targets <= 0.5] = len(adv_targets) / sum(adv_targets <= 0.5)
        
        with torch.set_grad_enabled(is_training):
            wts = self.bitrate_dro_adversary(bitrate_dro_adversary_inputs.detach(), targets).squeeze()
            # adv_loss = (wts * (per_sample_losses - gamma).detach()).sum() - self.lambd * get_lp_norm(self.bitrate_dro_adversary, p=1)
            adv_loss =(criterion(wts, adv_targets) * cwts).mean() + self.lambd * get_lp_norm(self.bitrate_dro_adversary, p=1)
        if is_training:
            self.opt_bitrate_dro_adversary.zero_grad()
            (adv_loss).backward()    
            # print(adv_loss.item())
            # print(wts, adv_targets, per_sample_losses)
            self.opt_bitrate_dro_adversary.step()
            with torch.no_grad():
                wts = self.bitrate_dro_adversary.get_scaled_wts(bitrate_dro_adversary_inputs.detach(), targets).squeeze()
        
            # adv_probs_batch = torch.zeros_like(self.adv_probs)
            # for g in range(self.n_groups):
            #     if len(wts[group_idx==g]):
            #         adv_probs_batch[g] = wts[group_idx==g].mean()
            # adv_probs_batch = adv_probs_batch / (adv_probs_batch.sum() + 1e-3)
            # self.adv_probs = self.adv_probs * torch.exp(
            #     self.step_size * adv_probs_batch)
            # self.adv_probs = self.adv_probs / (self.adv_probs.sum())
            
        wts = wts.detach()
        return (wts * per_sample_losses).sum() / (wts.sum() + 1e-3), wts
        
        
        
        

    def compute_robust_loss(self, group_loss, group_count):
        adjusted_loss = group_loss
        if torch.all(self.adj > 0):
            adjusted_loss += self.adj / torch.sqrt(self.group_counts)
        if self.normalize_loss:
            adjusted_loss = adjusted_loss / (adjusted_loss.sum())
        self.adv_probs = self.adv_probs * torch.exp(
            self.step_size * adjusted_loss.data)
        self.adv_probs = self.adv_probs / (self.adv_probs.sum())

        robust_loss = group_loss @ self.adv_probs
        return robust_loss, self.adv_probs

    def compute_robust_loss_btl(self, group_loss, group_count):
        adjusted_loss = self.exp_avg_loss + self.adj / torch.sqrt(
            self.group_counts)
        return self.compute_robust_loss_greedy(group_loss, adjusted_loss)

    def compute_robust_loss_greedy(self, group_loss, ref_loss):
        sorted_idx = ref_loss.sort(descending=True)[1]
        sorted_loss = group_loss[sorted_idx]
        sorted_frac = self.group_frac[sorted_idx]

        mask = torch.cumsum(sorted_frac, dim=0) <= self.alpha
        weights = mask.float() * sorted_frac / self.alpha
        last_idx = mask.sum()
        weights[last_idx] = 1 - weights.sum()
        weights = sorted_frac * self.min_var_weight + weights * (
            1 - self.min_var_weight)

        robust_loss = sorted_loss @ weights

        # sort the weights back
        _, unsort_idx = sorted_idx.sort()
        unsorted_weights = weights[unsort_idx]
        return robust_loss, unsorted_weights

    def compute_group_avg(self, losses, group_idx):
        # compute observed counts and mean loss for each group
        group_map = (group_idx == torch.arange(
            self.n_groups).unsqueeze(1).long().cuda()).float()

        group_count = group_map.sum(1)
        group_denom = group_count + (group_count == 0).float()  # avoid nans
        group_loss = (group_map @ losses.view(-1)) / group_denom
        return group_loss, group_count

    def update_exp_avg_loss(self, group_loss, group_count):
        prev_weights = (1 - self.gamma * (group_count > 0).float()) * (
            self.exp_avg_initialized > 0).float()
        curr_weights = 1 - prev_weights
        self.exp_avg_loss = self.exp_avg_loss * prev_weights + group_loss * curr_weights
        self.exp_avg_initialized = (self.exp_avg_initialized >
                                    0) + (group_count > 0)

    def reset_stats(self):
        self.processed_data_counts = torch.zeros(self.n_groups).cuda()
        self.update_data_counts = torch.zeros(self.n_groups).cuda()
        self.update_batch_counts = torch.zeros(self.n_groups).cuda()
        self.avg_group_loss = torch.zeros(self.n_groups).cuda()
        self.avg_group_acc = torch.zeros(self.n_groups).cuda()
        self.avg_per_sample_loss = 0.0
        self.avg_actual_loss = 0.0
        self.avg_acc = 0.0
        self.batch_count = 0.0

    def update_stats(self,
                     actual_loss,
                     group_loss,
                     group_acc,
                     group_count,
                     weights=None):
        # avg group loss
        denom = self.processed_data_counts + group_count
        denom += (denom == 0).float()
        prev_weight = self.processed_data_counts / denom
        curr_weight = group_count / denom
        self.avg_group_loss = prev_weight * self.avg_group_loss + curr_weight * group_loss

        # avg group acc
        self.avg_group_acc = prev_weight * self.avg_group_acc + curr_weight * group_acc

        # batch-wise average actual loss
        denom = self.batch_count + 1
        self.avg_actual_loss = (self.batch_count /
                                denom) * self.avg_actual_loss + (
                                    1 / denom) * actual_loss

        # counts
        self.processed_data_counts += group_count
        if self.loss_type == "group_dro":
            self.update_data_counts += group_count * ((weights > 0).float())
            self.update_batch_counts += ((group_count * weights) > 0).float()
        else:
            self.update_data_counts += group_count
            self.update_batch_counts += (group_count > 0).float()
        self.batch_count += 1

        # avg per-sample quantities
        group_frac = self.processed_data_counts / (
            self.processed_data_counts.sum())
        self.avg_per_sample_loss = group_frac @ self.avg_group_loss
        self.avg_acc = group_frac @ self.avg_group_acc

    def get_model_stats(self, model, args, stats_dict):
        model_norm_sq = 0.0
        for param in model.parameters():
            model_norm_sq += torch.norm(param)**2
        stats_dict["model_norm_sq"] = model_norm_sq.item()
        stats_dict["reg_loss"] = args.weight_decay / 2 * model_norm_sq.item()
        return stats_dict

    def get_stats(self, model=None, args=None):
        stats_dict = {}
        for idx in range(self.n_groups):
            stats_dict[f"avg_loss_group:{idx}"] = self.avg_group_loss[
                idx].item()
            stats_dict[f"exp_avg_loss_group:{idx}"] = self.exp_avg_loss[
                idx].item()
            stats_dict[f"avg_acc_group:{idx}"] = self.avg_group_acc[idx].item()
            stats_dict[
                f"processed_data_count_group:{idx}"] = self.processed_data_counts[
                    idx].item()
            stats_dict[
                f"update_data_count_group:{idx}"] = self.update_data_counts[
                    idx].item()
            stats_dict[
                f"update_batch_count_group:{idx}"] = self.update_batch_counts[
                    idx].item()

        stats_dict["avg_actual_loss"] = self.avg_actual_loss.item()
        stats_dict["avg_per_sample_loss"] = self.avg_per_sample_loss.item()
        stats_dict["avg_acc"] = self.avg_acc.item()

        # Model stats
        if model is not None:
            assert args is not None
            stats_dict = self.get_model_stats(model, args, stats_dict)

        return stats_dict

    def log_stats(self, logger, is_training):
        if logger is None:
            return

        logger.write(
            f"Average incurred loss: {self.avg_per_sample_loss.item():.3f}  \n"
        )
        logger.write(
            f"Average sample loss: {self.avg_actual_loss.item():.3f}  \n")
        logger.write(f"Average acc: {self.avg_acc.item():.3f}  \n")
        for group_idx in range(self.n_groups):
            logger.write(
                f"  {self.group_str(group_idx)}  "
                f"[n = {int(self.processed_data_counts[group_idx])}]:\t"
                f"loss = {self.avg_group_loss[group_idx]:.3f}  "
                f"exp loss = {self.exp_avg_loss[group_idx]:.3f}  "
                f"adjusted loss = {self.exp_avg_loss[group_idx] + self.adj[group_idx]/torch.sqrt(self.group_counts)[group_idx]:.3f}  "
                f"adv prob = {self.adv_probs[group_idx]:3f}   "
                f"acc = {self.avg_group_acc[group_idx]:.3f}\n")
        logger.flush()
