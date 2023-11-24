import os
import torch
from lib.dataset import find_dataset_using_name
from lib.core.evaluate import Evaluator
from torch.utils.data import DataLoader
from lib.utils.utils import prepare_output_dir, worker_init_fn
from lib.core.config import parse_args
from lib.models.mixsynthformer import PoseRefinementModel


def main(cfg):
    dataset_class = find_dataset_using_name(cfg.DATASET_NAME)


    test_dataset = dataset_class(cfg,
                                estimator=cfg.ESTIMATOR,
                                return_type=cfg.BODY_REPRESENTATION,
                                phase='test')

    test_loader = DataLoader(dataset=test_dataset,
                            batch_size=1,
                            shuffle=False,
                            num_workers=cfg.TRAIN.WORKERS_NUM,
                            pin_memory=True,
                            worker_init_fn=worker_init_fn)

    rt = 1
    rc = 1

    model =PoseRefinementModel(sample_interval=cfg.SAMPLE_INTERVAL,
                                           step=cfg.MODEL.SLIDE_WINDOW_Q,
                                           joint_dim=test_dataset.input_dimension,
                                           emb_dim=cfg.MODEL.EMBEDDING_DIMENSION,
                                           num_encoder_layers=cfg.MODEL.BLOCK_NUM,
                                           expansion_factor=2,
                                           dropout=cfg.MODEL.DROPOUT,
                                           device=cfg.DEVICE,
                                           reduce_token=rt,  # in temporal attention matrix gen
                                           reduce_channel=rc,  # in spatial attentino matrix gen
                                           ).to(cfg.DEVICE)

    if cfg.EVALUATE.PRETRAINED != '' and os.path.isfile(
            cfg.EVALUATE.PRETRAINED):
        checkpoint = torch.load(cfg.EVALUATE.PRETRAINED)
        # checkpoint = torch.load(cfg.EVALUATE.PRETRAINED, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['state_dict'], False)
        performance = checkpoint['performance']
        print(f'==> Loaded pretrained model from {cfg.EVALUATE.PRETRAINED}...')
    else:
        print(f'{cfg.EVALUATE.PRETRAINED} is not a pretrained model!!!!')
        exit()

    evaluator = Evaluator(model=model, test_loader=test_loader, cfg=cfg, input_dim= test_dataset.input_dimension)
    # evaluator.calculate_flops()
    # evaluator.calculate_parameter_number()
    evaluator.run()


if __name__ == '__main__':
    cfg, cfg_file = parse_args()
    cfg = prepare_output_dir(cfg, cfg_file)

    main(cfg)