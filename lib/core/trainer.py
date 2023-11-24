import torch
import logging

from tqdm import tqdm

from lib.core.loss import *

from lib.utils.eval_metrics import *
from lib.utils.geometry_utils import *

import os
import shutil

logger = logging.getLogger(__name__)


class Trainer():  # merge

    def __init__(self,
                 cfg,
                 train_dataloader,
                 test_dataloader,
                 model,
                 loss,
                 optimizer,
                 start_epoch=0):
        super().__init__()
        self.cfg = cfg

        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.model = model
        self.loss = loss
        self.optimizer = optimizer

        self.logdir = cfg.LOGDIR

        self.start_epoch = start_epoch
        self.end_epoch = cfg.TRAIN.EPOCH
        self.epoch = 0

        self.train_global_step = 0
        self.valid_global_step = 0
        self.device = cfg.DEVICE
        self.resume = cfg.TRAIN.RESUME
        self.lr = cfg.TRAIN.LR

        if self.device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Resume from a pretrained model
        if self.resume != '':
            self.resume_pretrained(self.resume)

    def run(self):
        logger.info("\n")
        self.evaluate()
        for epoch_num in range(self.start_epoch, self.end_epoch):
            logger.info("epoch " + str(epoch_num))
            self.epoch = epoch_num
            self.train()

            if epoch_num > 10:
                performance = self.evaluate()

                if performance["output_mpjpe"] < self.cfg.TRAIN.REF_ACC:
                    self.save_model(performance, epoch_num)

            # Decay learning rate exponentially
            lr_decay = self.cfg.TRAIN.LRDECAY
            self.lr *= lr_decay
            for param_group in self.optimizer.param_groups:
                param_group['lr'] *= lr_decay
            logger.info("\n")
        if not self.cfg.TRAIN.VALIDATE:
            performance = self.evaluate()

    def train(self):

        self.model.train()
        summary_string = ''

        for i, data in enumerate(tqdm(self.train_dataloader)):

            data_pred = data["pred"].to(self.device)
            data_gt = data["gt"].to(self.device)

            self.optimizer.zero_grad()

            predicted_3d_pos = self.model(data_pred)

            loss_total = self.loss.forward(predicted_3d_pos, data_gt)

            loss_total.backward()
            self.optimizer.step()

            summary_string = f'loss: {loss_total:.4f}'

            self.train_global_step += 1

            if torch.isnan(loss_total):
                exit('Nan value in loss, exiting!...')

        summary_string += f' | learning rate: {self.lr}'
        logger.info(summary_string)

    def evaluate_3d(self):

        eval_dict = evaluate_msf_3D(self.model, self.test_dataloader,
                                          self.device, self.cfg)

        log_str = f'Epoch {self.epoch}, '
        log_str += ' '.join(
            [f'{k.upper()}: {v:.2f},' for k, v in eval_dict.items()])
        logger.info(log_str)

        return eval_dict

    def evaluate_smpl(self):
        eval_dict = evaluate_msf_smpl(self.model, self.test_dataloader,
                                      self.device, self.cfg)

        log_str = f'Epoch {self.epoch}, '
        log_str += ' '.join(
            [f'{k.upper()}: {v:.2f},' for k, v in eval_dict.items()])
        logger.info(log_str)

        return eval_dict

    def evaluate_2d(self):
        eval_dict = evaluate_msf_2D(self.model, self.test_dataloader,
                                          self.device, self.cfg)

        log_str = f'Epoch {self.epoch}, '
        log_str += ' '.join(
            [f'{k.upper()}: {v * 100:.2f}%,' for k, v in eval_dict.items()])
        logger.info(log_str)

        return eval_dict

    def evaluate(self):

        self.model.eval()
        if self.cfg.BODY_REPRESENTATION == "3D":
            return self.evaluate_3d()

        elif self.cfg.BODY_REPRESENTATION == "smpl":
            return self.evaluate_smpl()

        elif self.cfg.BODY_REPRESENTATION == "2D":
            return self.evaluate_2d()

    def resume_pretrained(self, model_path):
        if os.path.isfile(model_path):
            checkpoint = torch.load(model_path)
            self.start_epoch = checkpoint['epoch']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.performance = checkpoint['performance']

            logger.info(
                f"=> loaded checkpoint '{model_path}' "
                f"(epoch {self.start_epoch}, performance {self.performance})")
        else:
            logger.info(f"=> no checkpoint found at '{model_path}'")

    def save_model(self, performance, epoch):
        save_dict = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'performance': performance,
            'optimizer': self.optimizer.state_dict()
        }

        filename = os.path.join(self.logdir, 'checkpoint_e' + str(epoch) + '.pth.tar')
        torch.save(save_dict, filename)
