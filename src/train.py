import os
import types

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from tqdm import tqdm

from utils import AverageMeter, accuracy
from loss import LossComputer

from pytorch_transformers import AdamW, WarmupLinearSchedule
from src.model_attributes import model_attributes
from data.utils import dataset_attributes
from data.dro_dataset import get_loader
from src.utils import get_adv_model

device = torch.device("cuda")
import os
import pandas as pd


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x


def fetch_and_remove_last_layer(args, model):
    last_layer = getattr(model, model_attributes[args.model]["last_layer_name"])
    setattr(model, model_attributes[args.model]["last_layer_name"], Identity())
    return last_layer, model


def get_features(args, model, loader):
    model.eval()
    last_layer, model = fetch_and_remove_last_layer(args, model)
    if args.show_progress:
        prog_bar_loader = tqdm(loader)
    else:
        prog_bar_loader = loader
    features = []
    indices = []
    with torch.set_grad_enabled(False):
        for batch_idx, batch in enumerate(prog_bar_loader):
            batch = tuple(t.to(device) for t in batch)
            x = batch[0]
            y = batch[1]
            g = batch[2]
            data_idx = batch[3]
            if args.model.startswith("bert"):
                raise NotImplementedError
            else:
                # outputs.shape: (batch_size, feature size)
                outputs = model(x)
            features.append(outputs.cpu())
            indices.append(batch[3].cpu())
    setattr(model, model_attributes[args.model]["last_layer_name"], last_layer)
    return torch.cat(features, 0), torch.cat(indices, 0)
                
            


def run_epoch(
    epoch,
    model,
    optimizer,
    loader,
    loss_computer,
    logger,
    csv_logger,
    args,
    is_training,
    show_progress=False,
    log_every=50,
    scheduler=None,
    csv_name=None,
    wandb_group=None,
    wandb=None,
):
    """
    scheduler is only used inside this function if model is bert.
    """

    
    
    if is_training:
        model.train()
        if (args.model.startswith("bert") and args.use_bert_params): # or (args.model == "bert"):
            model.zero_grad()
        
    else:
        model.eval()

    if show_progress:
        prog_bar_loader = tqdm(loader)
    else:
        prog_bar_loader = loader

    with torch.set_grad_enabled(is_training):

        for batch_idx, batch in enumerate(prog_bar_loader):
            batch = tuple(t.to(device) for t in batch)
            x = batch[0]
            y = batch[1]
            g = batch[2]
            data_idx = batch[3]
            
            if args.model.startswith("bert"):
                input_ids = x[:, :, 0]
                input_masks = x[:, :, 1]
                segment_ids = x[:, :, 2]
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=input_masks,
                    token_type_ids=segment_ids,
                    labels=y,
                )[1]  # [1] returns logits
            else:
                # outputs.shape: (batch_size, num_classes)
                outputs = model(x)

                
            output_df = pd.DataFrame()

            # Calculate stats
            if batch_idx == 0:
                acc_y_pred = np.argmax(outputs.detach().cpu().numpy(), axis=1)
                acc_y_true = y.cpu().numpy()
                indices = data_idx.cpu().numpy()
                
                probs = outputs.detach().cpu().numpy()
            else:
                acc_y_pred = np.concatenate([
                    acc_y_pred,
                    np.argmax(outputs.detach().cpu().numpy(), axis=1)
                ])
                acc_y_true = np.concatenate([acc_y_true, y.cpu().numpy()])
                indices = np.concatenate([indices, data_idx.cpu().numpy()])
                probs = np.concatenate([probs, outputs.detach().cpu().numpy()], axis = 0)
                
            assert probs.shape[0] == indices.shape[0]
            # TODO: make this cleaner.
            run_name = f"{csv_name}_epoch_{epoch}_val"
            output_df[f"y_pred_{run_name}"] = acc_y_pred
            output_df[f"y_true_{run_name}"] = acc_y_true
            output_df[f"indices_{run_name}"] = indices
            
            for class_ind in range(probs.shape[1]):
                output_df[f"pred_prob_{run_name}_{class_ind}"] = probs[:, class_ind]
            
            if loss_computer.loss_type == 'bitrate_dro':
                with torch.no_grad():
                    last_layer, model = fetch_and_remove_last_layer(args, model)
                    features = model(x) 
                    setattr(model, model_attributes[args.model]["last_layer_name"], last_layer)                
                loss_main, wt_updates = loss_computer.loss(outputs, y, g, is_training, bitrate_dro_adversary_inputs=features)
                if is_training:
                    loader.dataset.update_weights(data_idx, wt_updates.cpu())
            else:    
                loss_main = loss_computer.loss(outputs, y, g, is_training)

            if is_training:
                if (args.model.startswith("bert") and args.use_bert_params): 
                    loss_main.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                   args.max_grad_norm)
                    scheduler.step()
                    optimizer.step()
                    model.zero_grad()
                else:
                    optimizer.zero_grad()
                    loss_main.backward()
                    optimizer.step()

            if is_training and (batch_idx + 1) % log_every == 0:
                run_stats = loss_computer.get_stats(model, args)
                if csv_logger is not None:
                    csv_logger.log(epoch, batch_idx, run_stats)
                    csv_logger.flush()
                loss_computer.log_stats(logger, is_training)
                loss_computer.reset_stats()
                if wandb is not None:
                    wandb_stats = {
                        wandb_group + "/" + key: run_stats[key] for key in run_stats.keys()
                    }
                    wandb_stats["epoch"] = epoch
                    wandb_stats["batch_idx"] = batch_idx
                    wandb.log(wandb_stats)

        if csv_logger is not None:
            save_dir = "/".join(csv_logger.path.split("/")[:-1])
            output_df.to_csv(
                os.path.join(save_dir, 
                                f"output_{wandb_group}_epoch_{epoch}.csv"))
            print("Saved", os.path.join(save_dir, 
                                f"output_{wandb_group}_epoch_{epoch}.csv"))


        if (not is_training) or loss_computer.batch_count > 0:
            run_stats = loss_computer.get_stats(model, args)
            if wandb is not None:
                assert wandb_group is not None
                wandb_stats = {
                    wandb_group + "/" + key: run_stats[key] for key in run_stats.keys()
                }
                wandb_stats["epoch"] = epoch
                wandb_stats["batch_idx"] = batch_idx
                wandb.log(wandb_stats)
                print("logged to wandb")
            
            if csv_logger is not None:
                csv_logger.log(epoch, batch_idx, run_stats)
                csv_logger.flush()
            loss_computer.log_stats(logger, is_training)
            if is_training:
                loss_computer.reset_stats()


def train(
    model,
    criterion,
    dataset,
    logger,
    train_csv_logger,
    val_csv_logger,
    test_csv_logger,
    args,
    epoch_offset,
    csv_name=None,
    wandb=None,
):
    model = model.to(device)

    # process generalization adjustment stuff
    adjustments = [float(c) for c in args.generalization_adjustment.split(",")]
    assert len(adjustments) in (1, dataset["train_data"].n_groups)
    if len(adjustments) == 1:
        adjustments = np.array(adjustments * dataset["train_data"].n_groups)
    else:
        adjustments = np.array(adjustments)
        
    # birate DRO adversary
    if args.loss_type == 'bitrate_dro':
        bitrate_dro_adversary = get_adv_model(
            model='linear',
            input_dim=model.fc.in_features,
            n_classes=dataset['train_data'].n_classes).to(device)
        logger.write(f"Creating bitrate adversary ... \n\n {bitrate_dro_adversary}" + "\n\n")
    else:
        bitrate_dro_adversary = None

    train_loss_computer = LossComputer(
        criterion,
        loss_type=args.loss_type,
        dataset=dataset["train_data"],
        alpha=args.alpha,
        gamma=args.gamma,
        adj=adjustments,
        step_size=args.robust_step_size,
        normalize_loss=args.use_normalized_loss,
        btl=args.btl,
        min_var_weight=args.minimum_variational_weight,
        joint_dro_alpha=args.joint_dro_alpha,
        bitrate_dro_adversary=bitrate_dro_adversary
    )

    # BERT uses its own scheduler and optimizer
    if (args.model.startswith("bert") and args.use_bert_params): 
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay":
                args.weight_decay,
            },
            {
                "params": [
                    p for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay":
                0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=args.lr,
                          eps=args.adam_epsilon)
        t_total = len(dataset["train_loader"]) * args.n_epochs
        print(f"\nt_total is {t_total}\n")
        scheduler = WarmupLinearSchedule(optimizer,
                                         warmup_steps=args.warmup_steps,
                                         t_total=t_total)
    else:
        optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr,
            momentum=0.9,
            weight_decay=args.weight_decay,
        )
        if args.scheduler:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                "min",
                factor=0.1,
                patience=5,
                threshold=0.0001,
                min_lr=0,
                eps=1e-08,
            )
        else:
            scheduler = None
            
    # Initial Validation (Before training begins, helpful for linear head training)
    logger.write(f"\nInitial Validation:\n")
    val_loss_computer =  LossComputer(
        criterion,
        loss_type=args.loss_type,
        dataset=dataset["val_data"],
        alpha=args.alpha,
        gamma=args.gamma,
        adj=adjustments,
        step_size=args.robust_step_size,
        normalize_loss=args.use_normalized_loss,
        btl=args.btl,
        min_var_weight=args.minimum_variational_weight,
        joint_dro_alpha=args.joint_dro_alpha,
        bitrate_dro_adversary=bitrate_dro_adversary
    )
    # run_epoch(
    #     -1,
    #     model,
    #     optimizer,
    #     dataset["val_loader"],
    #     val_loss_computer,
    #     logger,
    #     None,
    #     args,
    #     is_training=False,
    #     csv_name=None,
    #     wandb_group=None,
    #     wandb=None,
    # )

    best_val_acc = 0
    for epoch in range(epoch_offset, epoch_offset + args.n_epochs):
        logger.write("\nEpoch [%d]:\n" % epoch)

        # for bit rate dro we need to update loader with new weights
        if args.loss_type == 'bitrate_dro':
            dataset['train_data'].normalize_weights()
            for group_idx in range(dataset['train_data'].n_groups):
                group_wt = dataset['train_data']._weight_array[dataset['train_data']._group_array == group_idx].sum() / dataset['train_data']._weight_array.sum() 
                logger.write(
                        f"  {dataset['train_data'].group_str(group_idx)}:\t  "
                        f"sampling weight = {group_wt:.3f}\n")
            loader_kwargs = {
                "batch_size": args.batch_size,
                "num_workers": 4,
                "pin_memory": True,
            }
            dataset['train_loader'] = get_loader(dataset['train_data'],
                train=True,
                reweight_groups=False,
                reweight_samples=True,
                **loader_kwargs
            )
            

        logger.write(f"Training:\n")
        run_epoch(
            epoch,
            model,
            optimizer,
            dataset["train_loader"],
            train_loss_computer,
            logger,
            train_csv_logger,
            args,
            is_training=True,
            csv_name=csv_name,
            show_progress=args.show_progress,
            log_every=args.log_every,
            scheduler=scheduler,
            wandb_group="train",
            wandb=wandb,
        )

        logger.write(f"\nValidation:\n")
        # run_epoch(
        #     epoch,
        #     model,
        #     optimizer,
        #     dataset["val_loader"],
        #     val_loss_computer,
        #     logger,
        #     val_csv_logger,
        #     args,
        #     is_training=False,
        #     csv_name=csv_name,
        #     wandb_group="val",
        #     wandb=wandb,
        # )

        # Test set; don't print to avoid peeking
        if dataset["test_data"] is not None:
            test_loss_computer = LossComputer(
                criterion,
                loss_type=args.loss_type,
                dataset=dataset["test_data"],
                step_size=args.robust_step_size,
                alpha=args.alpha,
                gamma=args.gamma,
                adj=adjustments,
                normalize_loss=args.use_normalized_loss,
                btl=args.btl,
                min_var_weight=args.minimum_variational_weight,
                joint_dro_alpha=args.joint_dro_alpha,
                bitrate_dro_adversary=bitrate_dro_adversary
            )
            # run_epoch(
            #     epoch,
            #     model,
            #     optimizer,
            #     dataset["test_loader"],
            #     test_loss_computer,
            #     None,
            #     test_csv_logger,
            #     args,
            #     is_training=False,
            #     csv_name=csv_name,
            #     wandb_group="test",
            #     wandb=wandb,
            # )

        # Inspect learning rates
        if (epoch + 1) % 1 == 0:
            for param_group in optimizer.param_groups:
                curr_lr = param_group["lr"]
                logger.write("Current lr: %f\n" % curr_lr)

        if args.scheduler and args.model != "bert":
            if args.loss_type == "group_dro":
                val_loss, _ = val_loss_computer.compute_robust_loss_greedy(
                    val_loss_computer.avg_group_loss,
                    val_loss_computer.avg_group_loss)
            else:
                val_loss = val_loss_computer.avg_actual_loss
            scheduler.step(
                val_loss)  # scheduler step to update lr at the end of epoch

        if epoch % args.save_step == 0:
            torch.save(model, os.path.join(args.log_dir,
                                           "%d_model.pth" % epoch))

        if args.save_last:
            torch.save(model, os.path.join(args.log_dir, "last_model.pth"))

        if args.save_best:
            if args.loss_type == "group_dro" or args.reweight_groups:
                curr_val_acc = min(val_loss_computer.avg_group_acc)
            else:
                curr_val_acc = val_loss_computer.avg_acc
            logger.write(f"Current validation accuracy: {curr_val_acc}\n")
            if curr_val_acc > best_val_acc:
                best_val_acc = curr_val_acc
                torch.save(model, os.path.join(args.log_dir, "best_model.pth"))
                logger.write(f"Best model saved at epoch {epoch}\n")    
                if args.save_features:
                    logger.write(f"Saving features for the best model saved at epoch {epoch}\n")    
                    train_features, train_indices = get_features(args, model, dataset["train_loader"])
                    val_features, val_indices = get_features(args, model, dataset["val_loader"])
                    test_features, test_indices = get_features(args, model, dataset["test_loader"])
                    logger.write(f"Found {len(train_indices)} train {len(val_indices)} val {len(test_indices)} test features, \n")    
                    total_num_feat = len(train_features) + len(val_features) + len(test_features)
                    assert total_num_feat == len(set(train_indices.tolist() + val_indices.tolist() + test_indices.tolist())), "indices dont match features"
                    feature_array = torch.zeros(total_num_feat, *train_features.shape[1:])
                    feature_array[train_indices, :] = train_features.cpu()
                    feature_array[val_indices, :] = val_features.cpu()
                    feature_array[test_indices, :] = test_features.cpu()
                    np.save(
                        os.path.join(
                            dataset_attributes[args.dataset]["root_dir"],
                            "features",
                            model_attributes[args.model]["feature_filename"]),
                        feature_array)

        if args.automatic_adjustment:
            gen_gap = val_loss_computer.avg_group_loss - train_loss_computer.exp_avg_loss
            adjustments = gen_gap * torch.sqrt(
                train_loss_computer.group_counts)
            train_loss_computer.adj = adjustments
            logger.write("Adjustments updated\n")
            for group_idx in range(train_loss_computer.n_groups):
                logger.write(
                    f"  {train_loss_computer.get_group_name(group_idx)}:\t"
                    f"adj = {train_loss_computer.adj[group_idx]:.3f}\n")
        logger.write("\n")
