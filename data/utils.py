import sys,os
sys.path.append(os.getcwd())

import os
import torch
import pandas as pd
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from data.waterbirds import WaterbirdsDataset
from data.celeba import CelebADataset
from absl import flags, app
from src.model_attributes import model_attributes
from data.dro_dataset import DRODataset, get_loader
from torch.utils.data import Subset


FLAGS = flags.FLAGS

flags.DEFINE_string('root_dir', '../datasets/waterbirds', 'root directory of data')
flags.DEFINE_string('target_name', 'waterbird_complete95', 'target name')
flags.DEFINE_string('dataset', 'Waterbirds', 'dataset name')
flags.DEFINE_list('confounder_names', ['forest2water2'], 'confounder names')
flags.DEFINE_string('model', 'resnet50', 'type of model')
flags.DEFINE_boolean('augment_data', False, 'to perform data augmentation')
flags.DEFINE_string('metadata_csv_name', 'metadata.csv', 'name for csv metadata')


########################
####### SETTINGS #######
########################
confounder_settings = {
    "Waterbirds": {
        "constructor": WaterbirdsDataset
    },
    "CelebA": {
        "constructor": CelebADataset
    }
}
root_dir = os.getcwd()
dataset_attributes = {
    "CelebA": {
        "root_dir": "datasets/CelebA"
    },
    "Waterbirds": {
        "root_dir": "datasets/Waterbirds"
    },
    "CIFAR10": {
        "root_dir": "datasets/CIFAR10/data"
    },
    "MultiNLI": {
        "root_dir": "datasets/multinli"
    },
    'jigsaw': {
        'root_dir': 'datasets/jigsaw'
    },
}
for dataset in dataset_attributes:
    dataset_attributes[dataset]["root_dir"] = os.path.join(
        root_dir, dataset_attributes[dataset]["root_dir"])

shift_types = ["confounder"]


def prepare_data(args, train, return_full_dataset=False):
    # Set root_dir to defaults if necessary
    if args.root_dir is None:
        args.root_dir = dataset_attributes[args.dataset]["root_dir"]
    if args.shift_type == "confounder":
        return prepare_confounder_data(
            args,
            train,
            return_full_dataset,
        )
    else:
        raise NotImplementedError


def log_data(data, logger):
    logger.write("Training Data...\n")
    for group_idx in range(data["train_data"].n_groups):
        logger.write(
            f'    {data["train_data"].group_str(group_idx)}: n = {data["train_data"].group_counts()[group_idx]:.0f}\n'
        )
    logger.write("Validation Data...\n")
    for group_idx in range(data["val_data"].n_groups):
        logger.write(
            f'    {data["val_data"].group_str(group_idx)}: n = {data["val_data"].group_counts()[group_idx]:.0f}\n'
        )
    if data["test_data"] is not None:
        logger.write("Test Data...\n")
        for group_idx in range(data["test_data"].n_groups):
            logger.write(
                f'    {data["test_data"].group_str(group_idx)}: n = {data["test_data"].group_counts()[group_idx]:.0f}\n'
            )



########################
### DATA PREPARATION ###
########################
def prepare_confounder_data(args, train, return_full_dataset=False):
    full_dataset = confounder_settings[args.dataset]["constructor"](
        root_dir=args.root_dir,
        target_name=args.target_name,
        confounder_names=args.confounder_names,
        model_type=args.model,
        augment_data=args.augment_data,
        metadata_csv_name=args.metadata_csv_name    
    )
    if return_full_dataset:
        return DRODataset(
            full_dataset,
            process_item_fn=None,
            n_groups=full_dataset.n_groups,
            n_classes=full_dataset.n_classes,
            group_str_fn=full_dataset.group_str,
        )
    if train:
        splits = ["train", "val", "test"]
    else:
        splits = ["test"]
    subsets = full_dataset.get_splits(splits, train_frac=args.fraction)
    dro_subsets = [
        DRODataset(
            subsets[split],
            process_item_fn=None,
            n_groups=full_dataset.n_groups,
            n_classes=full_dataset.n_classes,
            group_str_fn=full_dataset.group_str,
        ) for split in splits
    ]
    return dro_subsets


def main(_):
    dataset = prepare_confounder_data(
        FLAGS, train=True, return_full_dataset=True)
    data = iter(get_loader(dataset, train=True, reweight_groups=True,  batch_size=512)).next()
    from collections import Counter
    print(Counter(data[2].numpy()))
if __name__ == '__main__':
    app.run(main)