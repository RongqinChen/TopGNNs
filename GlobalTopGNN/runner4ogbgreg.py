import argparse
import json
from datetime import datetime
from trainutils import Config, TrainerBase
from torch import Tensor
import os
import torch
import time
from typing import Tuple
from tqdm import tqdm
os.environ['TZ'] = 'Asia/Shanghai'
time.tzset()


class Trainer(TrainerBase):
    def __init__(self, config: Config) -> None:
        super().__init__(config)

    def get_targets(self, graph_batch):
        targets = graph_batch.graph_attr
        return targets

    def train_batch(self, graph_batch) -> float:
        graph_batch = graph_batch.to(self.device)

        if graph_batch.batch_size < 8:
            # print("A batch skipped")
            # print("due to graphs.batch_size() < 8")
            return 0.
        num_nodes = graph_batch.num_nodes
        if num_nodes < 8:
            # print("A batch skipped")
            # print("due to min(num_nodes) < 8")
            return 0.

        self.optimizer.zero_grad()
        preds = self.nn_model(graph_batch)
        targets = self.get_targets(graph_batch)
        loss: Tensor = self.loss_module(preds, targets)
        loss.backward()
        self.optimizer.step()
        loss_val = loss.item()
        return loss_val

    def evaluate_epoch(self, loader, data_split) -> "Tuple[float, float]":
        targets_list = []
        preds_list = []
        self.nn_model.eval()
        with torch.no_grad():
            with tqdm(loader, total=len(loader), dynamic_ncols=True,
                      disable=self.disable_tqdm) as batches_tqdm:
                for idx, graph_batch in enumerate(batches_tqdm):
                    desc = f'Evaluating on {data_split} set'
                    desc = f'{desc:30s} | Iteration #{idx+1}/{len(loader)}'
                    batches_tqdm.set_description(desc)
                    graph_batch = graph_batch.to(self.device)
                    preds = self.nn_model(graph_batch)
                    preds_list.append(preds.detach().cpu())
                    targets = self.get_targets(graph_batch)
                    targets_list.append(targets.cpu())

        targets = torch.concat(targets_list, 0)
        preds = torch.concat(preds_list, 0)
        score = self.evaluator(preds, targets)
        loss = self.loss_module(preds, targets).item()
        return loss, score


def set_config(args):
    config_path = 'GlobalTopGNN/config/ogbgreg.json'
    with open(config_path, 'r') as rfile:
        config_dict: dict = json.load(rfile)

    datamodule_args_dict = config_dict['datamodule_args_dict']
    datamodule_args_dict['name'] = args.dataset_name
    datamodule_args_dict['transform_fn_kwargs']['height'] = args.tree_height
    config_dict['nn_model_args_dict']['hilayers'] = args.hilayers
    config_dict['nn_model_args_dict']['height'] = args.tree_height
    config_dict['nn_model_args_dict']['readout'] = args.readout
    config_dict['loss_module_name'] = args.loss_module_name
    config_dict['disable_tqdm'] = args.disable_tqdm

    config = Config()
    dname = datamodule_args_dict['name']
    config.config_key = \
        f"{dname}.Hi{args.hilayers}.T{args.tree_height}.{args.readout}" # noqa
    config.seed = config_dict['seed']
    config.fold_idx = None
    config.device = 0
    config.datamodule_name = f"datautils.{config_dict['datamodule_name']}"
    config.datamodule_args_dict = datamodule_args_dict
    config.nn_model_name = f"GlobalTopGNN.{config_dict['nn_model_name']}"
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
        description='Evaluating GlobalTopGNN\'s performance on OGBG datasets.')
    parser.add_argument('--dataset_name', type=str)
    parser.add_argument('--hilayers', type=int)
    parser.add_argument('--tree_height', type=int)
    parser.add_argument('--loss_module_name', type=str,
                        default='torch.nn.L1Loss')
    parser.add_argument('--readout', type=str, default='sum')
    parser.add_argument('--num_runs', type=int, default=10)
    parser.add_argument('--disable_tqdm', action='store_true')
    args = parser.parse_args()

    config = set_config(args)
    for fold_idx in range(1, args.num_runs+1):
        config.fold_idx = fold_idx
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
