# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import json
import logging
from pathlib import Path
import os
import time

import torch
import torch.nn as nn
from easydict import EasyDict as edict

import src.distrib as distrib
from src.utils import bold, copy_state, LogProgress, AverageMeter, accuracy


logger = logging.getLogger(__name__)


class Trainer(object):
    def __init__(self, data, model, criterion, optimizer, args):
        self.tr_loader = data['tr']
        self.tt_loader = data['tt']
        self.model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        self.dmodel = distrib.wrap(model)
        self.optimizer = optimizer
        self.criterion = criterion

        if args.lr_sched == 'step':
            from torch.optim.lr_scheduler import StepLR
            sched = StepLR(self.optimizer, step_size=args.step.step_size, gamma=args.step.gamma)
        elif args.lr_sched == 'multistep':
            from torch.optim.lr_scheduler import MultiStepLR
            sched = MultiStepLR(self.optimizer, milestones=[30, 60, 80], gamma=args.multistep.gamma)
        elif args.lr_sched == 'plateau':
            from torch.optim.lr_scheduler import ReduceLROnPlateau
            sched = ReduceLROnPlateau(
                self.optimizer, factor=args.plateau.factor, patience=args.plateau.patience)
        elif args.lr_sched == 'cosine':
            from torch.optim.lr_scheduler import CosineAnnealingLR
            sched = CosineAnnealingLR(
                self.optimizer, T_max=args.cosine.T_max, eta_min=args.cosine.min_lr)
        else:
            sched = None
        self.sched = sched

        # Training config
        self.device = args.device
        self.epochs = args.epochs
        self.max_norm = args.max_norm

        # Checkpoints
        self.continue_from = args.continue_from
        self.checkpoint = Path(
            args.checkpoint_file) if args.checkpoint else None
        if self.checkpoint:
            logger.debug("Checkpoint will be saved to %s", self.checkpoint.resolve())
        self.history_file = args.history_file

        self.best_state = None
        self.restart = args.restart
        # keep track of losses
        self.history = []

        # logging
        self.num_prints = args.num_prints

        if args.mixed:
            self.scaler = torch.cuda.amp.GradScaler()

        # for seperation tests
        self.args = args
        self._load_or_init_quant()

    def _init_quant_model(self):
        # initialize quantization
        for layers in self.model.modules():
            if hasattr(layers, 'init'):
                layers.init.data.fill_(1)

        inputs, labels = next(iter(self.tr_loader))
        inputs, labels = inputs.to(self.device), labels.to(self.device)
        with torch.no_grad():
            for bit in self.args.quant.bit_list:
                print("bit : ", bit)
                for name, layers in self.model.named_modules():
                    if hasattr(layers, 'act_bit'):
                        setattr(layers, "act_bit", int(bit))
                    if hasattr(layers, 'weight_bit'):
                        setattr(layers, "weight_bit", int(bit))    
                self.model(inputs)

        for layers in self.model.modules():
            if hasattr(layers, 'init'):
                layers.init.data.fill_(0)

    def _serialize(self, path):
        package = {}
        package['state'] = self.model.state_dict()
        package['optimizer'] = self.optimizer.state_dict()
        if self.args.mixed:
            package['scaler'] = self.scaler.state_dict()
        if self.sched is not None:
            package['sched'] = self.sched.state_dict()
        package['history'] = self.history
        package['best_state'] = self.best_state
        package['args'] = self.args
        tmp_path = str(self.checkpoint) + ".tmp"

        torch.save(package, tmp_path)
        os.rename(tmp_path, path)

    def _load_or_init_quant(self):
        if self.args.pre_load_pretrained:
            logger.info("Preload pretrained model")
            return 
        load_from = None
        # load checkpoint if exists and not restart
        if self.checkpoint and self.checkpoint.exists() and not self.restart:
            load_from = self.checkpoint
        elif self.continue_from:
            load_from = self.continue_from
        # initialize quantization
        else:
            if hasattr(self, '_init_quant_model'):
                logger.info('Initializing quantization model')
                self._init_quant_model()

        if load_from:
            logger.info(f'Loading checkpoint model: {load_from}')
            package = torch.load(load_from, 'cpu')
            strict = load_from == self.checkpoint
            if load_from == self.continue_from and self.args.continue_best:
                self.model.load_state_dict(package['best_state'], strict=strict)
            else:
                self.model.load_state_dict(package['state'], strict=strict)
            if load_from == self.checkpoint:
                self.optimizer.load_state_dict(package['optimizer'])
                if self.args.mixed:
                    self.scaler.load_state_dict(package['scaler'])
                if self.sched is not None:
                    self.sched.load_state_dict(package['sched'])
                self.history = package['history']
                self.best_state = package['best_state']

    def train(self):
        # Optimizing the model
        if self.history:
            logger.info("Replaying metrics from previous run")

        for epoch, metrics in enumerate(self.history):
            info = " ".join(f"{k}={v:.5f}" for k, v in metrics.items())
            logger.info(f"Epoch {epoch}: {info}")
            if self.sched is not None:
                self.sched.step()

        
        for epoch in range(len(self.history), self.epochs):
            # Train one epoch            
            self.model.train()  # Turn on BatchNorm & Dropout
                                                                        
            start = time.time()
            logger.info('-' * 70)
            logger.info("Training...")
            
            train_bit_info = self._run_one_epoch(epoch)
            
            logger.info(bold(f'Train Summary | End of Epoch {epoch + 1} | '
                              f'Time {time.time() - start:.2f}s | \n {train_bit_info}'))
            
            # # Cross validation
            logger.info('-' * 70)
            logger.info('Cross validation...')
            self.model.eval()  # Turn off Batchnorm & Dropout & Diffq

            with torch.no_grad():
                val_bit_info = self._run_one_epoch(epoch, validation=True)
            
            logger.info(bold(f'Valid Summary | End of Epoch {epoch + 1} | '
                             f'Time {time.time() - start:.2f}s |\n {val_bit_info} '))

            if self.sched:
                if self.args.lr_sched == 'plateau':
                    print("will support later")
                    #self.sched.step(bit_info[''])
                else:
                    self.sched.step()
                new_lr = self.optimizer.state_dict()["param_groups"][0]["lr"]
                logger.info(f'Learning rate adjusted: {new_lr:.5f}')

            best_loss = float('inf')
            best_acc1 = 0
            best_acc5 = 0
            for metrics in self.history:
                if metrics['avg_val_loss'] < best_loss:
                    best_acc1 = metrics['avg_val_acc1']
                    best_loss = metrics['avg_val_loss']
                    best_acc5 = metrics['avg_val_acc5']
            
            metrics = {'avg_train_loss': train_bit_info['avg_loss'], 'avg_train_acc1': train_bit_info['avg_Acc1'],
                        'avg_train_acc5': train_bit_info['avg_Acc5'],
                       'avg_val_loss': val_bit_info['avg_loss'], 'avg_val_acc1': val_bit_info['avg_Acc1'],
                        'avg_val_acc5': val_bit_info['avg_Acc5'],
                       'best_loss': best_loss, 'best_acc1': best_acc1,
                       'best_acc5': best_acc5}

            # Save the best model
            if metrics['avg_val_loss'] == best_loss:
                logger.info(bold('New best valid loss %.4f'), metrics['avg_val_loss'])
                self.best_state = copy_state(self.model.state_dict())

            self.history.append(metrics)
            info = " | ".join(f"{k.capitalize()} {v:.5f}" for k, v in metrics.items())
            logger.info('-' * 70)
            logger.info(bold(f"Overall Summary | Epoch {epoch + 1} | {info}"))

            if distrib.rank == 0:
                json.dump(self.history, open(self.history_file, "w"), indent=2)
                # Save model each epoch
                if self.checkpoint:
                    self._serialize(self.checkpoint)
                    logger.debug("Checkpoint saved to %s", self.checkpoint.resolve())

    def evaluate(self):
        self.model.eval()
        start = time.time()
        with torch.no_grad():
            bit_info = self._run_one_epoch(0, validation=True)
            logger.info(bold(f'Valid Summary | '
                             f'Time {time.time() - start:.2f}s \n'
                             f'{bit_info}'))



    def _run_one_epoch(self, epoch, validation=False, ori_model=False):

        bit_loss_dict = edict()
        bit_Acc1_dict = edict()
        bit_Acc5_dict = edict()
        for bit in self.args.quant.bit_list:
            bit = str(bit)
            bit_loss_dict[bit]=AverageMeter(f'{int(bit)}bit loss', ':.4f')
            bit_Acc1_dict[bit]=AverageMeter(f'{int(bit)}bit Top 1 Acc', ':.4f')
            bit_Acc5_dict[bit]=AverageMeter(f'{int(bit)}bit Top 5 Acc', ':.4f')
        
        data_loader = self.tr_loader if not validation else self.tt_loader
        data_loader.epoch = epoch

        label = ["Train", "Valid"][validation]
        name = label + f" | Epoch {epoch + 1}"
        logprog = LogProgress(logger, data_loader, updates=self.num_prints, name=name)
        for i, (inputs, targets) in enumerate(logprog):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            for bit in self.args.quant.bit_list:
                bit = str(bit)
                for name, layers in self.dmodel.named_modules():
                    if hasattr(layers, 'act_bit'):
                        setattr(layers, "act_bit", int(bit))
                    if hasattr(layers, 'weight_bit'):
                        setattr(layers, "weight_bit", int(bit))

                if not validation:
                    with torch.cuda.amp.autocast(bool(self.args.mixed)):
                        yhat = self.dmodel(inputs)
                        loss = self.criterion(yhat, targets)            
                else:
                    if hasattr(self.dmodel, 'module'):
                        yhat = self.dmodel.module.forward(inputs)
                        loss = self.criterion(yhat, targets)
                    else:
                        yhat = self.dmodel.forward(inputs)
                        loss = self.criterion(yhat, targets)
                
                if not validation:
                    # optimize model in training mode
                    self.optimizer.zero_grad()
                    
                    if self.args.mixed:
                        self.scaler.scale(loss).backward()
                        self.scaler.unscale_(self.optimizer)
                    else:
                        loss.backward()

                    torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                self.max_norm)
                    if self.args.mixed:
                        self.scaler.step(self.optimizer)
                    else:
                        self.optimizer.step()
                    
                    if self.args.mixed:
                        self.scaler.update()

                bit_loss_dict[bit].update(loss.item(), inputs.size(0))
                acc1, acc5 = accuracy(yhat, targets, topk=(1, 5))
                bit_Acc1_dict[bit].update(acc1[0], inputs.size(0))
                bit_Acc5_dict[bit].update(acc5[0], inputs.size(0))
                
            bit_info = edict()
            for bit in self.args.quant.bit_list:
                bit = str(bit)
                bit_info[f'{bit}_loss'] = bit_loss_dict[bit].avg
                bit_info[f'{bit}_Acc1'] = bit_Acc1_dict[bit].avg
                bit_info[f'{bit}_Acc5'] = bit_Acc5_dict[bit].avg
                
            logprog.update(**bit_info)
            
            del loss
        
        bit_info = edict()
        for bit in self.args.quant.bit_list:
            
            bit_info[f'{bit}_loss']= distrib.average(bit_loss_dict[bit].sum, bit_loss_dict[bit].count)[0]
            bit_info[f'{bit}_Acc1']= distrib.average(bit_Acc1_dict[bit].sum, bit_loss_dict[bit].count)[0]
            bit_info[f'{bit}_Acc5']= distrib.average(bit_Acc5_dict[bit].sum, bit_loss_dict[bit].count)[0]
        
        bit_info['avg_loss'] = sum([bit_info[f'{bit}_loss'] for bit in self.args.quant.bit_list]) / len(self.args.quant.bit_list)
        bit_info['avg_Acc1'] = sum([bit_info[f'{bit}_Acc1'] for bit in self.args.quant.bit_list]) / len(self.args.quant.bit_list)
        bit_info['avg_Acc5'] = sum([bit_info[f'{bit}_Acc5'] for bit in self.args.quant.bit_list]) / len(self.args.quant.bit_list)

        print(bit_info)
        return bit_info
