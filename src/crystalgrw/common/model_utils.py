# from omegaconf import DictConfig, OmegaConf
import os
from torch.distributed import init_process_group

from ..models import models, sde, encoders, decoders, decode_stats
from ..models.encoders import controller


def get_hparams(hparams):
    hparams = {k: v for k, v in hparams.items() if k != "_target_"}
    return hparams


def get_model(cfg):
    # Encoder
    if hasattr(cfg, "encoder"):
        if cfg.encoder._target_ is not None:
            encoder = getattr(
                encoders, cfg.encoder._target_)(**get_hparams(cfg.encoder))
        else:
            encoder = None
    else:
        encoder = None
  
    # Decode stats
    if hasattr(cfg, "param_decoder"):
        if cfg.param_decoder._target_ == "PretrainedModel":
            # from models.decoders import predict_formula
            import os
            import torch
            model_path = os.path.realpath(cfg.param_decoder.model_path)
            # param_decoder = predict_formula.MLPModel(**get_hparams(cfg.param_decoder))
            # param_decoder.load_state_dict(torch.load(model_path))
            param_decoder = torch.load(model_path)
            freeze_params(param_decoder, cfg.param_decoder.train)
        else:
            param_decoder = getattr(
                decode_stats, cfg.param_decoder._target_)(**get_hparams(cfg.param_decoder))
    else:
        param_decoder = None

    # SDE
    sde_fn = getattr(sde, cfg.sde._target_)(**get_hparams(cfg.sde))

    # Decoder
    score_fn = getattr(decoders, cfg.score_fn._target_)(**get_hparams(cfg.score_fn))

    # Controller
    if hasattr(cfg, "controller"):
        if hasattr(controller, cfg.controller._target_):
            control_fn = getattr(controller, cfg.controller._target_)(cfg.controller)
        else:
            control_fn = get_external_module(cfg.controller)
    else:
        control_fn = None

    # Model
    return getattr(models, cfg.model._target_)(
        encoder, sde_fn, score_fn, control_fn, cfg,
    )


def freeze_params(model, train="freeze"):
    if train == "freeze":
        model.eval()
        break_idx = len(list(model.parameters()))
    elif train == "transfer":
        break_idx = 0
    elif train == "fine_tune":
        break_idx = len(list(model.parameters())) - 2
    else:
        raise NotImplementedError(f"{train} not implemented for freezing parameters")

    i = 0
    for param in model.parameters():
        if i < break_idx:
            param.requires_grad_(False)
            i += 1
        else:
            break


def get_external_module(cfg):
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location(cfg.module_name, cfg.module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[cfg.module_name] = module
    spec.loader.exec_module(module)
    return getattr(module, cfg._target_)(cfg)


def ddp_setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)