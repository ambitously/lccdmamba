import argparse
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from common.cdloader import CDReader, TestReader
from common.ready import Args
from work.train import train


def parse_args():
    parser = argparse.ArgumentParser(description="Train LCCDMamba on a change detection dataset")
    parser.add_argument("--data-root", type=str, required=True,
                        help="Dataset root. GVLM-CD256 should contain A, B, label and list folders.")
    parser.add_argument("--dataset-name", type=str, default="GVLM-CD256")
    parser.add_argument("--output-dir", type=str, default="output")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1.4e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", type=str, default="0",
                        help="Single GPU id before CUDA_VISIBLE_DEVICES remapping. Use 0 on AutoDL.")
    parser.add_argument("--seed", type=int, default=32765)
    parser.add_argument("--model", type=str, default="lccdmamba", choices=["lccdmamba"])
    parser.add_argument("--pred-idx", type=int, default=0)
    parser.add_argument("--no-test", action="store_true", help="Skip final test after training.")
    return parser.parse_args()


def seed_torch(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(name):
    if name == "lccdmamba":
        try:
            from lccdmamba.model import LCCDMamba
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "LCCDMamba needs the VMamba source files that are missing from this fork. "
                "Please copy lccdmamba/vmamba/vmamba.py and lccdmamba/configs/config.py "
                "plus the VSSM yaml configs from the original LCCDMamba/VMamba release."
            ) from exc
        return LCCDMamba()
    raise ValueError(f"Unsupported model: {name}")


if __name__ == "__main__":
    cli = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = cli.device
    seed_torch(cli.seed)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This training script is intended for a single NVIDIA GPU.")
    device = torch.device("cuda:0")
    torch.cuda.empty_cache()

    model = build_model(cli.model)
    model_name = model.__class__.__name__

    args = Args(os.path.join(cli.output_dir, cli.dataset_name.lower()), model_name)
    args.data_name = cli.dataset_name
    args.num_classes = 2
    args.batch_size = cli.batch_size
    args.iters = cli.epochs
    args.pred_idx = cli.pred_idx
    args.device = device
    args.lr = cli.lr
    args.weight_decay = cli.weight_decay
    args.results_dir = cli.results_dir
    args.skip_test = cli.no_test

    train_data = CDReader(path_root=cli.data_root, mode="train", en_edge=False)
    eval_data = CDReader(path_root=cli.data_root, mode="val", en_edge=False)
    test_data = TestReader(path_root=cli.data_root, mode="test", en_edge=False)

    dataloader_train = DataLoader(
        dataset=train_data,
        batch_size=args.batch_size,
        num_workers=cli.num_workers,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
    )
    dataloader_eval = DataLoader(
        dataset=eval_data,
        batch_size=args.batch_size,
        num_workers=cli.num_workers,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
    )
    dataloader_test = DataLoader(
        dataset=test_data,
        batch_size=args.batch_size,
        num_workers=cli.num_workers,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
    )

    model = model.to(device=device, dtype=torch.float)
    train(model, dataloader_train, dataloader_eval, dataloader_test, args)
