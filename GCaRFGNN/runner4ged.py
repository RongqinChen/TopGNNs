import argparse
import json
import os
import time
from datetime import datetime
import torch as th
from trainutils import Config, TrainerBase
from tqdm import tqdm
os.environ['TZ'] = 'Asia/Shanghai'
time.tzset()


class Trainer(TrainerBase):
    def __init__(self, config: Config) -> None:
        super().__init__(config)

    def train_batch(self, batch) -> float:
        ged_ts, gbatch, gidxs = batch
        ged_ts = ged_ts.to(self.device)
        gbatch = gbatch.to(self.device)
        gidxs = gidxs.to(self.device)
        self.optimizer.zero_grad()
        graph_h = self.nn_model.encode(gbatch)
        left_h = th.index_select(graph_h, 0, gidxs[0])
        right_h = th.index_select(graph_h, 0, gidxs[1])
        preds = th.mean(th.abs(left_h - right_h), dim=1, keepdim=True)
        loss = self.loss_module(preds, ged_ts)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def evaluate_epoch(self, loader, data_split):
        targets_list = []
        preds_list = []
        self.nn_model.eval()
        with th.no_grad():
            with tqdm(loader, total=len(loader), dynamic_ncols=True,
                      disable=self.disable_tqdm) as batches_tqdm:
                for idx, batch in enumerate(batches_tqdm):
                    desc = f'Evaluating on {data_split} set'
                    desc = f'{desc:30s} | Iteration #{idx+1}/{len(loader)}'
                    batches_tqdm.set_description(desc)
                    ged_ts, gbatch, gidxs = batch
                    gbatch = gbatch.to(self.device)
                    gidxs = gidxs.to(self.device)
                    targets_list.append(ged_ts)
                    graph_h = self.nn_model.encode(gbatch)
                    left_h = th.index_select(graph_h, 0, gidxs[0])
                    right_h = th.index_select(graph_h, 0, gidxs[1])
                    preds = th.mean(th.abs(left_h-right_h),
                                    dim=1, keepdim=True)
                    preds_list.append(preds.cpu())

        targets = th.concat(targets_list, 0)
        preds = th.concat(preds_list, 0)
        score = self.evaluator(preds, targets)
        loss = self.loss_module(preds, targets).item()
        return loss, score


def set_config(args):
    config_path = 'GCaRFGNN/config/ged.json'
    with open(config_path, 'r') as rfile:
        config_dict: dict = json.load(rfile)

    datamodule_args_dict = config_dict['datamodule_args_dict']
    datamodule_args_dict['name'] = args.dataset_name
    datamodule_args_dict['transform_fn_kwargs']['height'] = args.tree_height
    config_dict['nn_model_args_dict']['height'] = args.tree_height
    config_dict['nn_model_args_dict']['readout'] = args.readout
    config_dict['disable_tqdm'] = args.disable_tqdm

    config = Config()
    dname = datamodule_args_dict['name']
    config.config_key = \
        f"{dname}.T{args.tree_height}.{args.readout}"
    config.seed = config_dict['seed']
    config.fold_idx = None
    config.device = 0
    config.datamodule_name = f"datautils.{config_dict['datamodule_name']}"
    config.datamodule_args_dict = datamodule_args_dict
    config.nn_model_name = f"GCaRFGNN.{config_dict['nn_model_name']}"
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
        description='Evaluating the performance of GCaRFGNN on GED datasets.')
    parser.add_argument('--dataset_name', type=str, default='AIDS700nef')
    parser.add_argument('--tree_height', type=int)
    parser.add_argument('--readout', type=str, default='mean')
    parser.add_argument('--num_runs', type=int, default=5)
    parser.add_argument('--disable_tqdm', action='store_true')
    args = parser.parse_args()

    config = set_config(args)
    for fold_idx in range(1, args.num_runs + 1):
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
