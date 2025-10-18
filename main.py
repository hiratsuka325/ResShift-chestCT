#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2023-10-26 20:20:36

import argparse
from omegaconf import OmegaConf

from utils.util_common import get_obj_from_str
from utils.util_opts import str2bool

import wandb
from dataclasses import is_dataclass, asdict

def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
            "--save_dir",
            type=str,
            default="./save_dir",
            help="Folder to save the checkpoints and training log",
            )
    parser.add_argument(
            "--resume",
            type=str,
            const=True,
            default="",
            nargs="?",
            help="resume from the save_dir or checkpoint",
            )
    parser.add_argument(
            "--cfg_path",
            type=str,
            default="./configs/training/ffhq256_bicubic8.yaml",
            help="Configs of yaml file",
            )
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = get_parser()

    configs = OmegaConf.load(args.cfg_path)

    # merge args to config
    for key in vars(args):
        if key in ['cfg_path', 'save_dir', 'resume', ]:
            configs[key] = getattr(args, key)
            
    # wandb初期化
    if hasattr(configs, 'logger') and hasattr(configs.logger, 'wandb') and configs.logger.wandb.project:
        if is_dataclass(configs):
            wandb_config = asdict(configs)
        else:
            wandb_config = OmegaConf.to_container(configs, resolve=True) 
            
        wandb_args = {
            'project': configs.logger.wandb.project,
            'name': configs.logger.wandb.get('name', None),
            'config': wandb_config,
        }

        run_id = configs.logger.wandb.get('id', None)
        if run_id is not None:
            wandb_args['id'] = run_id
            wandb_args['resume'] = 'allow'  # 既存runがあればresume、なければ新規作成

        wandb.init(**wandb_args)

    trainer = get_obj_from_str(configs.trainer.target)(configs)
    trainer.train()
