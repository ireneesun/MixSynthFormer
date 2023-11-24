import os

import torch

os.environ['PYOPENGL_PLATFORM'] = 'egl'

import pprint
import random
import numpy as np
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from lib.dataset import find_dataset_using_name
from lib.utils.utils import create_logger, prepare_output_dir, worker_init_fn
from lib.core.config import parse_args
from lib.core.loss import PoseLoss
from lib.core.trainer import Trainer
import torch.optim as optim
from thop import profile
import time

from lib.models.mixsynthformer import PoseRefinementModel


def main(cfg):
    if cfg.SEED_VALUE >= 0:
        print(f'Seed value for the experiment is {cfg.SEED_VALUE}')
        os.environ['PYTHONHASHSEED'] = str(cfg.SEED_VALUE)
        random.seed(cfg.SEED_VALUE)
        torch.manual_seed(cfg.SEED_VALUE)
        np.random.seed(cfg.SEED_VALUE)

    logger = create_logger(cfg.LOGDIR, phase='train')

    # logger.info(f'GPU name -> {torch.cuda.get_device_name()}')
    # logger.info(f'GPU feat -> {torch.cuda.get_device_properties("cuda")}')

    logger.info(pprint.pformat(cfg))

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    # ========= Dataloaders ========= #
    dataset_class = find_dataset_using_name(cfg.DATASET_NAME)
    train_dataset = dataset_class(cfg,
                                  estimator=cfg.ESTIMATOR,
                                  return_type=cfg.BODY_REPRESENTATION,
                                  phase='train')

    test_dataset = dataset_class(cfg,
                                 estimator=cfg.ESTIMATOR,
                                 return_type=cfg.BODY_REPRESENTATION,
                                 phase='test')

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=cfg.TRAIN.BATCH_SIZE,
                              shuffle=True,
                              num_workers=cfg.TRAIN.WORKERS_NUM,
                              pin_memory=True,
                              worker_init_fn=worker_init_fn)

    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=1,
                             shuffle=False,
                             num_workers=cfg.TRAIN.WORKERS_NUM,
                             pin_memory=True,
                             worker_init_fn=worker_init_fn)

    # # ========= Compile Loss ========= #
    loss = PoseLoss()

    # # ========= Initialize networks ========= #
    model = PoseRefinementModel(
        sample_interval=cfg.SAMPLE_INTERVAL,
        step=cfg.MODEL.SLIDE_WINDOW_Q,
        joint_dim=train_dataset.input_dimension,
        emb_dim=cfg.MODEL.EMBEDDING_DIMENSION,
        num_encoder_layers=cfg.MODEL.BLOCK_NUM,
        expansion_factor=2,
        dropout=cfg.MODEL.DROPOUT,
        device=cfg.DEVICE,
        reduce_token=4, # in temporal attention matrix gen
        reduce_channel=1, # in spatial attentino matrix gen
    ).to(cfg.DEVICE)

    calculate_parameter_number(model, logger)
    calculate_flops(cfg, model, train_dataset.input_dimension, logger, device=cfg.DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=cfg.TRAIN.LR, amsgrad=True)

    # ========= Start Training ========= #
    Trainer(train_dataloader=train_loader,
            test_dataloader=test_loader,
            model=model,
            loss=loss,
            optimizer=optimizer,
            cfg=cfg).run()


def calculate_parameter_number(model, logger):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters()
                        if p.requires_grad)
    log_str = f'Total Parameters: {total_num / (1000 ** 2)} M, Trainable Parameters: {trainable_num / (1000 ** 2)} M'
    logger.info(log_str)
    return {'Total': total_num, 'Trainable': trainable_num}


def calculate_flops(cfg, model, input_dim, logger, device="cuda"):
    input=torch.randn(1, cfg.MODEL.SLIDE_WINDOW_SIZE, input_dim).to(device)
    flops, _ = profile(model, inputs=input.unsqueeze(0))
    log_str = f'Flops Per Frame: {flops / cfg.MODEL.SLIDE_WINDOW_SIZE / (1000 ** 2)} M'
    logger.info(log_str)

    with torch.no_grad():
        start=time.time()
        predicted_pos = model(input)
        end = time.time()
    log_str = f'Inference time Per Frame: {(end - start) / cfg.MODEL.SLIDE_WINDOW_SIZE * 1000} ms'
    logger.info(log_str)

    return {'Flops': flops}


if __name__ == '__main__':
    cfg, cfg_file = parse_args()
    cfg = prepare_output_dir(cfg, cfg_file)

    main(cfg)
