import argparse
import json
import os
import time
from datetime import datetime
from typing import Tuple
import torch
from tqdm import tqdm

from trainutils import Config, TrainerBase

os.environ['TZ'] = 'Asia/Shanghai'
time.tzset()


class Trainer(TrainerBase):
    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.penalty_flag = True

    def train_batch(self, batched_graph) -> float:
        batched_graph = batched_graph.to(self.device)
        targets = batched_graph.targets

        self.optimizer.zero_grad()
        preds = self.nn_model(batched_graph)
        loss = self.loss_module(preds, targets.squeeze(-1))
        loss.backward()
        self.optimizer.step()
        loss_val = loss.item()
        return loss_val

    def evaluate_epoch(self, loader, data_split) -> Tuple[float, float]:
        targets_list = []
        preds_list = []
        self.nn_model.eval()
        with torch.no_grad():
            with tqdm(loader, total=len(loader), dynamic_ncols=True,
                      disable=self.disable_tqdm) as batches_tqdm:
                for idx, batched_graph in enumerate(batches_tqdm):
                    desc = f'Evaluating on {data_split} set'
                    desc = f'{desc:30s} | Iteration #{idx+1}/{len(loader)}'
                    batches_tqdm.set_description(desc)
                    batched_graph = batched_graph.to(self.device)
                    targets = batched_graph.targets
                    targets_list.append(targets)
                    preds = self.nn_model(batched_graph)
                    preds_list.append(preds)

            targets = torch.concat(targets_list, 0).detach().cpu()
            preds = torch.concat(preds_list, 0).detach().cpu()
            loss = self.loss_module(preds, targets.squeeze(-1)).item()
            score = self.evaluator(preds, targets)

        return loss, score


def set_config(args, fold_idx):
    config_path = f'RFGNN/config/{args.config_name}.json'
    with open(config_path, 'r') as rfile:
        config_dict: dict = json.load(rfile)

    datamodule_args_dict = config_dict['datamodule_args_dict']
    if args.loss_module_name is not None:
        config_dict['loss_module_name'] = args.loss_module_name
    datamodule_args_dict['fold_idx'] = fold_idx
    datamodule_args_dict['transform_fn_kwargs']['height'] = args.tree_height
    config_dict['nn_model_args_dict']['readout'] = args.readout
    config_dict['nn_model_args_dict']['height'] = args.tree_height
    config_dict['disable_tqdm'] = args.disable_tqdm

    config = Config()
    config.config_key = f"T{args.tree_height}.{args.readout}"
    config.seed = config_dict['seed']
    config.fold_idx = fold_idx
    config.device = 0
    config.datamodule_name = f"datautils.{config_dict['datamodule_name']}"
    config.datamodule_args_dict = datamodule_args_dict
    config.nn_model_name = f"RFGNN.{config_dict['nn_model_name']}"
    config.nn_model_args_dict = config_dict['nn_model_args_dict']
    config.loss_module_name = config_dict['loss_module_name']
    config.loss_module_args_dict = config_dict['loss_module_args_dict']
    config.lr = config_dict['lr']
    config.min_lr = config_dict['min_lr']
    config.optimizer_name = config_dict['optimizer_name']
    config.optimizer_args = config_dict['optimizer_args']
    config.higher_better = config_dict['higher_better']
    config.schedule_step = config_dict['schedule_step']
    config.lr_decay_factor = config_dict['lr_decay_factor']
    config.min_num_epochs = config_dict['min_num_epochs']
    config.max_num_epochs = config_dict['max_num_epochs']
    config.eval_train_step = config_dict['eval_train_step']
    config.save_model_step = config_dict['save_model_step']
    config.early_stoping_patience = config_dict['early_stoping_patience']
    config.test_only_when_improved = config_dict['test_only_when_improved']
    config.disable_tqdm = config_dict['disable_tqdm']
    return config


def main():
    parser = argparse.ArgumentParser(
        description='Evaluating the performance of RFGNN.')
    parser.add_argument('--config_name', type=str, default='csl')
    parser.add_argument('--tree_height', type=int)
    parser.add_argument('--loss_module_name', type=str)
    parser.add_argument('--readout', type=str, default='sum')
    parser.add_argument('--num_runs', type=int, default=5)
    parser.add_argument('--disable_tqdm', action='store_true')
    args = parser.parse_args()
    for fold_idx in range(args.num_runs):
        config = set_config(args, fold_idx)
        try:
            trainer = Trainer(config)
            trainer.run()
        except Exception as e:
            with open('run_error.log', 'a') as afile:
                now = datetime.now()
                date_time = now.strftime("%m/%d/%Y, %H:%M:%S")
                print(date_time, file=afile)
                print(e, file=afile)
            raise e


if __name__ == "__main__":
    main()
