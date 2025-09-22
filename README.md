# CrystalGRW: Generative Modeling of Crystal Structures with Targeted Properties via Geodesic Random Walks
[![arXiv](https://img.shields.io/badge/arXiv-2501.08998-blue)](https://arxiv.org/abs/2501.08998)

### Version

Current version: **1.1.1**

See the [CHANGELOG.md](./CHANGELOG.md) for details on updates.


### Training scheme for CrystalGRW
<p align="center">
<img align="middle" src="./assets/crystalgrw_training.png" alt="Training scheme for CrystalGRW." width="800" />
</p>

Manifolds depicted in the figure are $`\mathbb{T}^2`$, $`\Delta^2`$, and $`\mathbb{R}^3`$.

### Sampling scheme for CrystalGRW
<p align="center">
<img align="middle" src="./assets/crystalgrw_sampling.png" alt="Sampling scheme for CrystalGRW." width="800" />
</p>

## Installation
```bash
git clone https://github.com/trachote/crystalgrw.git
cd crystalgrw
conda env create -f environment.yaml
conda activate crystalgrw
```

## Usage
### Training a Model
To train a model, run the following command:
```bash
python scripts/train.py --config_path conf/mp20_condition.yaml \
                        --output_path output_dir \
                        --ddp True
```
The Distributed Data Parallel (DDP) option is set to `True` for multi-GPU training. This option is available for EquiformerV2 only. GemNet-dT does not support DDP training in the current version.

### Generating Crystal Structures
To generate structures, run the following command:
```bash
python scripts/evaluate.py --model_path output_dir \
                           --batch_size 8 \
                           --num_batches_to_samples 10 \
                           --adaptive_timestep 1 \
                           --save_xyz True 
```

### Generating Crystal Structures with Guided Conditions
To generate structures guided by an input point group, run the following command:
```bash
python scripts/evaluate.py --model_path output_dir \
                           --batch_size 8 \
                           --num_batches_to_samples 10 \
                           --adaptive_timestep 1 \
                           --label_string cpg_m-3m \
                           --guidance_strength 0.5 \
                           --save_xyz True
```

### Compute Metrics 
To evaluate compositional and structural validity, uniqueness, 
and novelty of generated structures, run the following command:
```bash
python compute_metrics.py --root_path path/to/folder \
                          --dataset_path data/mp_20/train.csv \
                          --gen_file_name gen_file_name \
                          --n_samples 1000 \
                          --task gen
```

### Example Runs with MP-20 Dataset
1. Unzip `data/mp_20.zip`.
2. Update paths in `conf/*.yaml` files by replacing `path-to-folder/crystalgrw` with `pwd`.
   - `conf/mp20_example.yaml`: Configuration for training the model.
   - `conf/mp20_condition.yaml`: Configuration for training with controlled conditions.
3. Train the model using the **Training a Model** command.

[//]: # (### Generate structures from pretrained models)

[//]: # (1. Unzip the pretrained models into your project folder.)

[//]: # (2. Use the **Generating Crystal Structures** command to generate structures.)

[//]: # (3. Use the **Generating Structures with Guided Conditions** command to generate structures based on specific point groups.)

## Configurations
CrystalGRW can choose to corrupt either of three crystal properties:
- *fractional coordinates*
- *atomic types*
- *lattice matrices*

Depending on the specific task, you may want to alter only some of these properties. 
Adjust the settings in the configuration file by modifying the `corrupt_{property}` tags in file `conf/*.yaml` accordingly.

## Pretrained Models
To use pretrained models, download them using the following command
```
git lfs pull -I checkpoints/<model-name>
```

### Generating structures from pretrained models
1. Once downloaded, edit the `PROJECT_ROOT` path in `hparams.yaml` file to point to your local project directory.
2. `pretrained_alexmp20_2025-04-15` a model trained on ALEX-MP-20 dataset, 
use the **Generating Crystal Structures** command to generate structures. 
3. `pretrained_alexmp20_pointgroups_2025-04-18` a model trained on ALEX-MP-20 dataset and conditioned on point groups, 
use the **Generating Structures with Guided Conditions** command to generate structures.

Models trained on the MP-20 dataset can be downloaded from the following link: <https://zenodo.org/records/14948252>.

## Citations
If you use CrystalGRW in your research, please cite:
```
@misc{tangsongcharoen2025crystalgrw,
      title={CrystalGRW: Generative Modeling of Crystal Structures with Targeted Properties via Geodesic Random Walks}, 
      author={Krit Tangsongcharoen and Teerachote Pakornchote and Chayanon Atthapak and Natthaphon Choomphon-anomakhun and Annop Ektarawong and Björn Alling and Christopher Sutton and Thiti Bovornratanaraks and Thiparat Chotibut},
      year={2025},
      eprint={2501.08998},
      archivePrefix={arXiv},
      primaryClass={cond-mat.mtrl-sci},
      url={https://arxiv.org/abs/2501.08998}, 
}
```

### GNN submodules
Two options for the denoiser (decoder) <br>
1) [EquiformerV2](https://github.com/atomicarchitects/equiformer_v2)
```
@inproceedings{
liao2024equiformerv,
title={EquiformerV2: Improved Equivariant Transformer for Scaling to Higher-Degree Representations},
author={Yi-Lun Liao and Brandon M Wood and Abhishek Das and Tess Smidt},
booktitle={The Twelfth International Conference on Learning Representations},
year={2024},
url={https://openreview.net/forum?id=mCOBKZmrzD}
}
```
2) [GemNet-dT](https://github.com/txie-93/cdvae/tree/main/cdvae/pl_modules/gemnet)
```
@inproceedings{
klicpera2021gemnet,
title={GemNet: Universal Directional Graph Neural Networks for Molecules},
author={Johannes Klicpera and Florian Becker and Stephan G{\"u}nnemann},
booktitle={Advances in Neural Information Processing Systems},
editor={A. Beygelzimer and Y. Dauphin and P. Liang and J. Wortman Vaughan},
year={2021},
url={https://openreview.net/forum?id=HS_sOaxS9K-}
}
```
One option for the encoder (if used) <br>
3) [DimeNet++](https://github.com/txie-93/cdvae/blob/main/cdvae/pl_modules/gnn.py)
```
@misc{gasteiger2022fast,
      title={Fast and Uncertainty-Aware Directional Message Passing for Non-Equilibrium Molecules}, 
      author={Johannes Gasteiger and Shankari Giri and Johannes T. Margraf and Stephan Günnemann},
      year={2022},
      eprint={2011.14115},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
