## Wrapper for crystalgrw.cli.train.main ##

import time
import argparse
from omegaconf import DictConfig, OmegaConf

import torch
import torch.multiprocessing as mp

from crystalgrw.cli.train import run_train


def main(rank, world_size, cfg):
    run_train(rank, world_size, cfg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--predict_property", default=False)
    parser.add_argument("--predict_property_class", default=False)
    parser.add_argument("--early_stop", type=int, default=300)
    parser.add_argument("--ckpt_load", type=str, default="latest",
                        help="Two modes, latest or val, to continue training from previous checkpoint.")
    parser.add_argument("--ddp", type=bool, default=False)

    args = parser.parse_args()

    OmegaConf.clear_resolvers()
    OmegaConf.register_new_resolver("now", lambda x: time.strftime(x))

    cfg = OmegaConf.load(args.config_path)
    cfg.output_dir = args.output_path

    if args.predict_property is not None:
        cfg.model.predict_property = args.predict_property

    if args.predict_property_class is not None:
        cfg.model.predict_property_class = args.predict_property_class

    cfg.data = OmegaConf.load("./conf/data/" + cfg.data + ".yaml")
    cfg = OmegaConf.create(OmegaConf.to_container(OmegaConf.create(OmegaConf.to_yaml(cfg)), resolve=True))
    cfg.data.early_stopping_patience_epoch = args.early_stop
    cfg.ckpt_load = args.ckpt_load
    cfg.ddp = args.ddp

    if torch.cuda.is_available():
        if cfg.ddp:
            world_size = torch.cuda.device_count()
            mp.spawn(main, args=(world_size, cfg), nprocs=world_size)
        else:
            main("cuda", 0, cfg)
    else:
        main("cpu", 0, cfg)
