## Wrapper for crystalgrw.cli.evaluate.main ##

import argparse
import torch
import torch.multiprocessing as mp
from crystalgrw.cli.evaluate import run_eval


def main(rank, world_size, cfg):
    run_eval(rank, world_size, cfg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--tasks", nargs="+", default=["gen"])

    parser.add_argument("--adaptive_timestep", default=1.0, type=float)
    parser.add_argument("--step_lr", default=1e-4, type=float)
    parser.add_argument("--min_sigma", default=0, type=float)
    parser.add_argument("--save_traj", default=False, type=bool)
    parser.add_argument("--disable_bar", default=False, type=bool)
    parser.add_argument("--num_evals", default=1, type=int)
    parser.add_argument("--num_batches_to_samples", default=20, type=int)
    parser.add_argument("--start_from", default="data", type=str)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--suffix", default="", type=str)

    parser.add_argument("--force_num_atoms", action="store_true")
    parser.add_argument("--force_atom_types", action="store_true")
    parser.add_argument("--down_sample_traj_step", default=10, type=int)
    parser.add_argument("--sample_num_atoms", default="random")
    parser.add_argument("--labels", nargs="+", default=None, type=float)
    parser.add_argument("--label_string", default=None)
    parser.add_argument("--guidance_strength", default=1, type=float)
    parser.add_argument("--sample_type_method", default="multinomial", type=str)

    parser.add_argument('--load_data', default=False, type=bool)
    parser.add_argument('--dataset_path', default=None, type=str)
    parser.add_argument("--save_xyz", default=False, type=bool)
    parser.add_argument("--ckpt_load", default="val", type=str)
    parser.add_argument("--ddp", default=False, type=bool)

    args = parser.parse_args()

    if torch.cuda.is_available():
        if args.ddp:
            world_size = torch.cuda.device_count()
            mp.spawn(main, args=(world_size, args), nprocs=world_size)
        else:
            main("cuda", 0, args)
    else:
        main("cpu", 0, args)
