# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.1.0] - 2025-06-01

### Added
- Added support for Distributed Data Parallel (DDP) training for EquiformerV2 model. This can be activated by setting the `--ddp` argument to `True` in the `train.py` script.

### Changed
- Refactored sample to be a method in `samplers` instead of a method under the `CrystalGRW` class.
- Refactored `conf/data/mp.yaml` to `conf/data/mp_20.yaml`.

### Fixed
- Used `multinomial` instead of `argmax` for sampling atom types. This improves the diversity of generated structures; therefore, the uniqueness increases. However, the sampling method can be controlled by setting the `sample_method` argument in the `evaluate.py` script to `argmax` if desired.

---

## [0.1.0] - 2025-03-11
### Added
- Updated `compute_metrics.py` to evaluate the uniqueness, novelty and unique & novel metrics.
- Added label_string argument to evaluate.py for generating structures with guided conditions. This allows to control point groups through string rather than class indices.

### Fixed
- Corrected the corruption of lattice matrices when not corrupting fractional coordinates. Previously, the lattice matrices were not being corrupted when fractional coordinates were not corrupted.
- Fixed a bug when saving xyz files after sampling.
- Fixed a bug when adding noisy atom types to the node attributes.
- Loading primitive cell from CIF files when loading Dataset. Previously, Structure object did not receive primitive parameter whose default was False.

---

## [0.0.2] - 2025-01-24

### Fixed
- Get the last iteration of the atom types when sampling. It was previously returning the initial atom types which were random.

---

## [0.0.1] - 2025-01-15
### Added
- Initial public release of CrystalGRW.
- Full support for training and sampling on MP-20 dataset.
- Configurable corruption of atomic types, fractional coordinates, and lattice matrices.
- Conditioned sampling with symmetry guidance via label input.
- Example configs and pretrained model loading support.
