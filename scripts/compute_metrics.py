## Wrapper for crystalgrw.cli.compute_metrics.main ##

import argparse
from crystalgrw.cli.compute_metrics import run_compute_metrics

def main(cfg):
    run_compute_metrics(cfg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", required=True, type=str)
    parser.add_argument("--dataset_path", required=True, type=str)
    parser.add_argument("--gen_file_name", required=True, type=str)
    parser.add_argument("--recon_file_name", default=None, type=str)
    parser.add_argument("--opt_file_name", default=None, type=str)
    parser.add_argument("--suffix", default="")
    parser.add_argument("--tasks", nargs="+", default=["gen"])
    parser.add_argument("--n_samples", default=1000, type=int)
    parser.add_argument("--unique_algo", default=1, type=int)
    parser.add_argument("--unique_sym", default=False, type=bool)
    parser.add_argument("--compute_unn_pg", default=False, type=bool)
    parser.add_argument("--save_unn_indices", default=False, type=bool)
    args = parser.parse_args()
    main(args)