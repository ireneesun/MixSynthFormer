import os
import torch
from lib.dataset import find_dataset_using_name
from lib.models.mixsynthformer import PoseRefinementModel
from lib.core.config import parse_args
from lib.visualize.visualize import Visualize


def main(cfg):
    dataset_class = find_dataset_using_name(cfg.DATASET_NAME)
    test_dataset = dataset_class(cfg,
                                 estimator=cfg.ESTIMATOR,
                                 return_type=cfg.BODY_REPRESENTATION,
                                 phase='test')

    model = PoseRefinementModel(
        sample_interval=cfg.SAMPLE_INTERVAL,
        step=cfg.MODEL.SLIDE_WINDOW_Q,
        joint_dim=test_dataset.input_dimension,
        emb_dim=cfg.MODEL.EMBEDDING_DIMENSION,
        num_encoder_layers=cfg.MODEL.BLOCK_NUM,
        expansion_factor=2,
        dropout=cfg.MODEL.DROPOUT,
        device=cfg.DEVICE,
        reduce_token=1, # in temporal attention matrix gen
        reduce_channel=1, # in spatial attentino matrix gen
    ).to(cfg.DEVICE)

    if cfg.EVALUATE.PRETRAINED != '' and os.path.isfile(
            cfg.EVALUATE.PRETRAINED):
        # checkpoint = torch.load(cfg.EVALUATE.PRETRAINED, map_location=torch.device('cpu'))
        checkpoint = torch.load(cfg.EVALUATE.PRETRAINED)
        model.load_state_dict(checkpoint['state_dict'], False)
        print(f'==> Loaded pretrained model from {cfg.EVALUATE.PRETRAINED}...')
    else:
        print(f'{cfg.EVALUATE.PRETRAINED} is not a pretrained model!!!!')
        exit()

    visualizer = Visualize(test_dataset, cfg)
    visualizer.visualize(model)


if __name__ == '__main__':
    cfg, cfg_file = parse_args()
    main(cfg)
