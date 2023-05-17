import importlib
import json
import logging
import os
import os.path as osp
import random
from datetime import datetime
from decimal import Decimal
from typing import Tuple

# import dgl
import numpy as np
import torch
from torch import Tensor, nn, optim
from tqdm import tqdm
import fcntl

from .config import Config
from datautils.datamodule import DataModuleBase


class TrainerBase(object):
    def __init__(self, config: Config) -> None:
        super().__init__()

        if torch.cuda.is_available() and config.device > -1:
            self._device = torch.device('cuda:%d' % config.device)
        else:
            self._device = torch.device('cpu')
        self._config = config
        # seed all
        seed = config.seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        # dgl.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
        if self._device.type[:4] == 'cuda':
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.benchmark = False

        self._log_root_dir = 'logs'
        early_stoping_patience = config.early_stoping_patience
        if early_stoping_patience is None or early_stoping_patience < 1:
            config.early_stoping_patience = config.max_num_epochs
        self._data_module = self._make_data_module()
        self._nn_model = self._make_nn_model()
        self._num_parameters = config.num_parameters = self._count_parameters()
        self._loss_module = self._make_loss_module()
        self._optimizer = self._make_optimizer()
        self._lr_scheduler = self._make_lr_scheduler()
        self._evaluator = self._data_module.evaluator
        self._set_logging()

        def init_weights(m):
            m_name = f'\t{m}'
            if hasattr(m, 'reset_parameters') and 'Embedding' not in m_name:
                self._logger.debug(m_name)
                m.reset_parameters()

        self._logger.debug('reset_parameters')
        self._nn_model.apply(init_weights)
        self.disable_tqdm = config.disable_tqdm
        self.epoch_idx = 1

        # results
        self._improved = None
        self._best_epoch = None
        self._train_loss = None
        self._valid_loss = None
        self._test_loss = None
        self._best_valid_loss = None
        self._best_test_loss = None

        self._train_score_dict = None
        self._valid_score_dict = None
        self._test_score_dict = None
        self._best_valid_score_dict = None
        self._best_test_score_dict = None

    @property
    def device(self):
        return self._device

    @property
    def nn_model(self):
        return self._nn_model

    @property
    def optimizer(self):
        return self._optimizer

    @property
    def data_module(self):
        return self._data_module

    @property
    def loss_module(self):
        return self._loss_module

    @property
    def lr_scheduler(self):
        return self._lr_scheduler

    @property
    def evaluator(self) -> dict:
        return self._evaluator

    def _make_data_module(self) -> DataModuleBase:
        config = self._config
        module_name, datamodule_name = config.datamodule_name.rsplit('.', 1)
        args_dict = config.datamodule_args_dict
        try:
            module = importlib.import_module(module_name)
            module_cls = getattr(module, datamodule_name)
        except Exception as e:
            raise KeyError(
                f'{e}\nInvalid data_module, {config.datamodule_name}')
        try:
            data_module = module_cls(**args_dict)
        except Exception as e:
            raise KeyError(f'{e}\nInvalid data_module_paras, {args_dict}')
        assert isinstance(data_module, DataModuleBase)
        return data_module

    def _make_nn_model(self) -> nn.Module:
        config = self._config
        module_name, nn_model_name = config.nn_model_name.rsplit('.', 1)
        try:
            module = importlib.import_module(module_name)
            model_module = getattr(module, nn_model_name)
        except Exception as e:
            raise KeyError(f'{e}\nInvalid nn_model, {config.nn_model_name}')
        nn_model_args_dict = config.nn_model_args_dict
        for key, val in nn_model_args_dict.items():
            if isinstance(val, str) and val.startswith('datamodule.'):
                prop = val[11:]
                prop_val = getattr(self._data_module, prop)
                nn_model_args_dict[key] = prop_val
        try:
            model = model_module(**nn_model_args_dict)
            assert isinstance(model, nn.Module)
        except Exception as e:
            raise KeyError(
                f'{e}\nInvalid nn_model_args_dict, {nn_model_args_dict}')
        model.to(self._device)
        return model

    def _make_optimizer(self) -> optim.Optimizer:
        try:
            optimizer_type = getattr(optim, self._config.optimizer_name)
            optimizer = optimizer_type(
                self._nn_model.parameters(),
                lr=self._config.lr,
                **self._config.optimizer_args)
        except Exception as e:
            raise KeyError(
                f'{e}\nInvalid optimizer {self._config.optimizer_name}')
        return optimizer

    def _make_lr_scheduler(self) -> optim.lr_scheduler._LRScheduler:
        schedule_step = self._config.schedule_step
        lr_decay_factor = self._config.lr_decay_factor
        if schedule_step > 0 and lr_decay_factor and 0. < lr_decay_factor < 1.:
            try:
                lr_scheduler = torch.optim.lr_scheduler.StepLR(
                    self._optimizer, schedule_step, gamma=lr_decay_factor, )
            except Exception as e:
                raise KeyError(
                    f'{e}\nInvalid scheduler {schedule_step, lr_decay_factor}')
        else:
            print('no lr scheduler')
            lr_scheduler = None
        return lr_scheduler

    def _make_loss_module(self) -> nn.Module:
        config = self._config
        module_name, fn_name = config.loss_module_name.rsplit('.', 1)
        try:
            module = importlib.import_module(module_name)
        except Exception as e:
            raise KeyError(
                f'{e}\nInvalid loss_module, {config.loss_module_name}')
        try:
            loss_module = getattr(module, fn_name)(
                **config.loss_module_args_dict)
            assert isinstance(loss_module, nn.Module)
        except Exception as e:
            args_dict = config.loss_module_args_dict
            raise KeyError(
                f'{e}\nInvalid loss_module_args_dict, {args_dict}')
        return loss_module

    def _whether_improve(self, epoch_idx):
        if self._data_module.whether_improve(
                self._valid_score_dict, self._best_valid_score_dict):
            self._best_epoch = epoch_idx
            self._best_valid_loss = self._valid_loss
            self._best_valid_score_dict = self._valid_score_dict
            self._improved = True
        else:
            self._improved = False

    def _set_logging(self) -> None:
        config = self._config
        timestamp = datetime.now().strftime('%m%d-%H%M%S')
        datamodule_name = str(self._data_module)
        nn_model_name = config.nn_model_name.rsplit('.', 1)[1]
        self._config_key_dir = osp.join(
            self._log_root_dir, datamodule_name,
            nn_model_name, config.config_key)

        exper_key = f"seed{config.seed}"
        if config.fold_idx is not None:
            exper_key = f"fold{config.fold_idx}_{exper_key}"
        exper_log_dir = osp.join(
            self._config_key_dir, exper_key, timestamp)
        self._exper_log_dir = exper_log_dir
        os.makedirs(exper_log_dir)
        config_dict = config.todict()
        with open(f'{exper_log_dir}/config.json', 'w') as wfile:
            json.dump(config_dict, wfile, indent=4)
        self._logger = logging.getLogger()
        self._logger.handlers.clear()
        self._logger.setLevel(logging.DEBUG)
        fh = logging.FileHandler(f'{exper_log_dir}/train.log')
        fh.setLevel(logging.DEBUG)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        self._logger.addHandler(fh)
        self._logger.addHandler(ch)
        self._logger.debug(self._nn_model)
        self._logger.debug(self._optimizer)

    def _count_parameters(self) -> int:
        model_parameters = filter(
            lambda p: p.requires_grad, self._nn_model.parameters())
        num_parameters = sum([np.prod(p.size()) for p in model_parameters])
        return int(num_parameters)

    def _update_statues(self, epoch_idx) -> None:
        lr_list = [
            f"{Decimal(param_group['lr']):.1E}"
            for param_group in self._optimizer.param_groups
        ]
        train_loss_str = f'{self._train_loss:.5f}' \
            if self._train_loss > 1e-4 else f'{Decimal(self._train_loss):.2E}'
        valid_loss_str = f'{self._valid_loss:.5f}' \
            if self._valid_loss > 1e-4 else f'{Decimal(self._valid_loss):.2E}'

        if self._test_loss is None:
            test_loss_str = ' ' * 7
        elif self._test_loss > 1e-4:
            test_loss_str = f'{self._test_loss:.5f}'
        else:
            test_loss_str = f'{Decimal(self._test_loss):.2E}'

        train_score_str = "_".join([
            f'{val:.5f}' for val in self._train_score_dict.values()])
        valid_score_str = "_".join([
            f'{val:.5f}' for val in self._valid_score_dict.values()])

        if self._test_score_dict is None:
            test_score_str = ' ' * 7
        else:
            test_score_str = "_".join([
                f'{val:.5f}' for val in self._test_score_dict.values()])

        if self._improved:
            statue_dict = {
                '_'.join(self._data_module.metric_name):
                    f'{train_score_str}/{valid_score_str}*,' +
                    ' ' * 7 + '/' + f'{test_score_str}*,' + ' ' * 7,
                'loss': f'{train_loss_str}/{valid_loss_str}*,' +
                        ' ' * 7 + f'/{test_loss_str}*,' + ' ' * 7,
            }
        else:
            if self._best_valid_loss > 1e-4:
                best_valid_loss_str = f'{self._best_valid_loss:.5f}'
            else:
                best_valid_loss_str = f'{Decimal(self._best_valid_loss):.2E}'

            if self._best_test_loss is None:
                best_test_loss_str = ' ' * 7
            elif self._best_test_loss > 1e-4:
                best_test_loss_str = f'{self._best_test_loss:.5f}'
            else:
                best_test_loss_str = f'{Decimal(self._best_test_loss):.2E}'

            best_valid_score_str = "_".join([
                f'{val:.5f}' for val in self._best_valid_score_dict.values()])

            if self._best_test_score_dict is None:
                best_test_score_str = ' ' * 7
            else:
                best_test_score_str = "_".join([
                    f'{val:.5f}'
                    for val in self._best_test_score_dict.values()])

            statue_dict = {
                '_'.join(self._data_module.metric_name):
                f'{train_score_str}/{best_valid_score_str}*,' +
                f'{valid_score_str}/{best_test_score_str}*,' +
                f'{test_score_str}',
                'loss': f'{train_loss_str}/{best_valid_loss_str}*,' +
                f'{valid_loss_str}/{best_test_loss_str}*,{test_loss_str}',
            }

        statue_list = [
            ('  LR: ' + ','.join(lr_list)),
            f" Epoch: {epoch_idx+1:03d}/{self._best_epoch+1:03d}"
        ]
        statue_list += [f'{key}: {val}' for key, val in statue_dict.items()]
        statue = ' | '.join(statue_list)
        print(statue)
        self._logger.debug(statue)

    def _save_result(self) -> None:
        result_dict = {
            'num_parameters': self._num_parameters,
            'best_epoch': self._best_epoch,
            'train_loss': self._train_loss,
            'valid_loss*': self._best_valid_loss,
        }
        result_dict.update({
            f'train_{key}': val
            for key, val in self._train_score_dict.items()
        })
        result_dict.update({
            f'valid_{key}': val
            for key, val in self._best_valid_score_dict.items()
        })
        if self._best_test_loss is not None:
            result_dict['test_loss'] = self._best_test_loss
            result_dict.update({
                f'test_{key}': val
                for key, val in self._best_test_score_dict.items()
            })

        fname = osp.join(self._config_key_dir, 'results.json')
        if not os.path.isfile(fname):
            with open(fname, 'w') as wfile:
                json.dump({}, wfile, indent=4)

        with open(fname, 'r+') as file:
            fcntl.flock(file, fcntl.LOCK_EX | fcntl.LOCK_NB)
            whole_dict = json.load(file)
            file.seek(0)
            file.truncate()
            whole_dict[self._exper_log_dir] = result_dict
            with open(fname, 'w') as wfile:
                json.dump(whole_dict, wfile, indent=4)
            fcntl.flock(file, fcntl.LOCK_UN)

    def _deconstrcut(self):
        del self._nn_model
        del self._optimizer
        del self._data_module
        del self._loss_module
        del self._lr_scheduler

    def run(self,) -> None:
        self._logger.info(f"Sizes: {self._data_module.dataset_sizes}")
        self._logger.info(f"config_key: {self._config.config_key}")
        self._logger.info(f"num_parameters: {self._num_parameters}")
        self._logger.info(f"seed: {self._config.seed}")
        save_model_step = self._config.save_model_step
        test_only_when_improved = self._config.test_only_when_improved
        early_stoping_patience = self._config.early_stoping_patience
        for epoch_idx in range(self._config.max_num_epochs):
            self.epoch_idx = epoch_idx
            print(self._exper_log_dir.split(os.path.sep, 1)[1])
            self.train_epoch(epoch_idx)
            self.valid_epoch(epoch_idx)
            self._whether_improve(epoch_idx)
            if self.data_module.test_loader is not None and \
                    (not test_only_when_improved or self._improved):
                self.test_epoch(epoch_idx)

            self._update_statues(epoch_idx)
            self._improved = False
            if epoch_idx > self._config.min_num_epochs and \
                    epoch_idx - self._best_epoch > early_stoping_patience:
                self._logger.info(
                    f'\nTraining stopped after {early_stoping_patience}'
                    ' epochs with no improvement!')
                break

            self.lr_scheduler_step()
            if save_model_step is not None \
                    and (epoch_idx+1) % save_model_step == 0:
                saving_path = os.path.join(
                    self._exper_log_dir, 'checkpoint{}.pth'.format(epoch_idx))
                torch.save({'nn_model': self._nn_model.state_dict(),
                            "optimizer": self._optimizer.state_dict()},
                           saving_path)

        self._save_result()
        self._deconstrcut()

    def lr_scheduler_step(self):
        if self._lr_scheduler is not None:
            lr_decay_factor = self._config.lr_decay_factor
            result_lr = min(self._lr_scheduler.get_last_lr()) * lr_decay_factor
            if result_lr > self._config.min_lr:
                self._lr_scheduler.step()

    def train_epoch(self, epoch_idx) -> None:
        loader = self.data_module.train_loader
        self.nn_model.train()
        loss_accsum = 0.
        with tqdm(loader, total=len(loader), dynamic_ncols=True,
                  disable=self.disable_tqdm) as batches_tqdm:
            for idx, batched_data in enumerate(batches_tqdm):
                desc = 'Training'
                desc = f'{desc:30s} | Iteration #{idx+1}/{len(loader)}'
                batches_tqdm.set_description(desc)
                loss = self.train_batch(batched_data)
                loss_accsum += loss
                batches_tqdm.set_postfix(loss=loss)

            loss_mean = loss_accsum / (idx + 1)
            batches_tqdm.set_postfix(loss=loss_mean)

        if epoch_idx % self._config.eval_train_step == 0:
            loss, score_dict = self.evaluate_epoch(loader, 'training')
            self._train_loss, self._train_score_dict = loss, score_dict

    def valid_epoch(self, epoch_idx) -> None:
        loader = self.data_module.valid_loader
        loss, score_dict = self.evaluate_epoch(loader, 'validation')
        self._valid_loss, self._valid_score_dict = loss, score_dict

    def test_epoch(self, epoch_idx) -> None:
        loader = self.data_module.test_loader
        loss, score_dict = 0., {}
        if loader is not None:
            loss, score_dict = self.evaluate_epoch(loader, 'test')
        if self._improved:
            self._best_test_loss, self._best_test_score_dict = loss, score_dict

        self._test_loss, self._test_score_dict = loss, score_dict

    def train_batch(self, batched_graph) -> float:
        batched_graph = batched_graph.to(self.device)
        targets = batched_graph.targets
        self.optimizer.zero_grad()
        preds = self.nn_model(batched_graph)
        loss: Tensor = self.loss_module(preds, targets)
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

        targets = torch.concat(targets_list, 0)
        preds = torch.concat(preds_list, 0)
        loss = self.loss_module(preds, targets).detach().cpu().item()
        score_dict = self.evaluator(preds, targets)
        return loss, score_dict

    def inference(self, loader):
        preds_list = []
        self.nn_model.eval()
        with torch.no_grad():
            with tqdm(loader, total=len(loader), dynamic_ncols=True,
                      disable=self.disable_tqdm) as batches_tqdm:
                for idx, batched_graph in enumerate(batches_tqdm):
                    batches_tqdm.set_description(
                        f'Inference Iteration #{idx+1}/{len(loader)}')
                    batched_graph.to(self.device)
                    preds = self.nn_model(batched_graph)
                    preds_list.append(preds)

        preds = torch.vstack(preds_list).detach().cpu()
        return preds
