from collections import Counter
import argparse
import os
import json

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from p_tqdm import p_map
from scipy.stats import wasserstein_distance
from scipy.sparse import csr_matrix, lil_matrix, eye

from pymatgen.core.structure import Structure
from pymatgen.core.composition import Composition
from pymatgen.core.lattice import Lattice
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from matminer.featurizers.site.fingerprint import CrystalNNFingerprint
from matminer.featurizers.composition.composite import ElementProperty

from ..common.eval_utils import (
    smact_validity, structure_validity, CompScaler, get_fp_pdist,
    load_config, load_data, get_crystals_list, prop_model_eval, compute_cov, class_model_eval)

CrystalNNFP = CrystalNNFingerprint.from_preset("ops")
CompFP = ElementProperty.from_preset("magpie")

Percentiles = {
    "mp20": np.array([-3.17562208, -2.82196882, -2.52814761]),
    "mp20_class": np.array([-3.17562208, -2.82196882, -2.52814761]),
    "carbon": np.array([-154.527093, -154.45865733, -154.44206825]),
    "perovskite": np.array([0.43924842, 0.61202443, 0.7364607]),
}

COV_Cutoffs = {
    "mp20": {"struc": 0.4, "comp": 10.},
    "mp20_class": {"struc": 0.4, "comp": 10.},
    "carbon": {"struc": 0.2, "comp": 4.},
    "perovskite": {"struc": 0.2, "comp": 4},
}


class Crystal(object):

    def __init__(self, crystal, analyze=True):
        if isinstance(crystal, dict):
            self.frac_coords = crystal["frac_coords"]
            self.atom_types = crystal["atom_types"]
            self.lengths = crystal["lengths"]
            self.angles = crystal["angles"]
            self.dict = crystal
            self.get_structure()
        elif isinstance(crystal, str):
            self.structure = Structure.from_str(crystal, "cif", primitive=True)
            self.constructed = True
        else:
            raise TypeError("crystal must be either a dict or a cif string")

        if analyze:
            self.get_composition()
            self.get_validity()
            self.get_fingerprints()

    def get_structure(self):
        if min(self.lengths.tolist()) < 0:
            self.constructed = False
            self.invalid_reason = "non_positive_lattice"
        else:
            try:
                self.structure = Structure(
                    lattice=Lattice.from_parameters(
                        *(self.lengths.tolist() + self.angles.tolist())),
                    species=self.atom_types, coords=self.frac_coords, coords_are_cartesian=False)
                self.constructed = True
            except Exception:
                self.constructed = False
                self.invalid_reason = "construction_raises_exception"
            if self.structure.volume < 0.1:
                self.constructed = False
                self.invalid_reason = "unrealistically_small_lattice"

    def get_composition(self):
        elem_counter = Counter(self.atom_types)
        composition = [(elem, elem_counter[elem])
                       for elem in sorted(elem_counter.keys())]
        elems, counts = list(zip(*composition))
        counts = np.array(counts)
        counts = counts / np.gcd.reduce(counts)
        self.elems = elems
        self.comps = tuple(counts.astype("int").tolist())

    def get_validity(self):
        self.comp_valid = smact_validity(self.elems, self.comps)
        if self.constructed:
            self.struct_valid = structure_validity(self.structure)
        else:
            self.struct_valid = False
        self.valid = self.comp_valid and self.struct_valid

    def get_fingerprints(self):
        elem_counter = Counter(self.atom_types)
        comp = Composition(elem_counter)
        self.comp_fp = CompFP.featurize(comp)
        try:
            site_fps = [CrystalNNFP.featurize(
                self.structure, i) for i in range(len(self.structure))]
        except Exception:
            # counts crystal as invalid if fingerprint cannot be constructed.
            self.valid = False
            self.comp_fp = None
            self.struct_fp = None
            return
        self.struct_fp = np.array(site_fps).mean(axis=0)


class RecEval(object):

    def __init__(self, pred_crys, gt_crys, stol=0.5, angle_tol=10, ltol=0.3):
        assert len(pred_crys) == len(gt_crys)
        self.matcher = StructureMatcher(
            stol=stol, angle_tol=angle_tol, ltol=ltol)
        self.preds = pred_crys
        self.gts = gt_crys

    def get_match_rate_and_rms(self):
        def process_one(pred, gt, is_valid):
            if not is_valid:
                return None, np.array([None] * 3), np.array([None] * 3)
            try:
                rms_dist = self.matcher.get_rms_dist(pred.structure, gt.structure)
                rms_dist = None if rms_dist is None else rms_dist[0]
                pred_lat = pred.structure.lattice
                gt_lat = gt.structure.lattice
                rms_length = np.array(pred_lat.lengths) - np.array(gt_lat.lengths)
                rms_angle = np.array(pred_lat.angles) - np.array(gt_lat.angles)
                return rms_dist, rms_length, rms_angle
            except Exception:
                return None, np.array([None] * 3), np.array([None] * 3)

        validity = [c.valid for c in self.preds]

        rms_dists, rms_lengths, rms_angles = [], [], []
        for i in tqdm(range(len(self.preds))):
            d, l, a = process_one(self.preds[i], self.gts[i], validity[i])
            rms_dists.append(d), rms_lengths.append(l), rms_angles.append(a)
        rms_dists = np.array(rms_dists)
        rms_lengths = np.concatenate(rms_lengths)
        rms_angles = np.concatenate(rms_angles)
        match_rate = sum(rms_dists != None) / len(self.preds)
        mean_rms_dist = rms_dists[rms_dists != None].mean()
        mean_rms_lengths = np.sqrt((rms_lengths[rms_lengths != None] ** 2).mean())
        mean_rms_angles = np.sqrt((rms_angles[rms_angles != None] ** 2).mean())
        return {"match_rate": match_rate,
                "rms_dist": mean_rms_dist,
                "rms_lengths": mean_rms_lengths,
                "rms_angles": mean_rms_angles}

    #     def get_match_rate_and_rms(self):
    #         def process_one(pred, gt, is_valid):
    #             if not is_valid:
    #                 return None
    #             try:
    #                 rms_dist = self.matcher.get_rms_dist(
    #                     pred.structure, gt.structure)
    #                 rms_dist = None if rms_dist is None else rms_dist[0]
    #                 return rms_dist
    #             except Exception:
    #                 return None
    #         validity = [c.valid for c in self.preds]

    #         rms_dists = []
    #         for i in tqdm(range(len(self.preds))):
    #             rms_dists.append(process_one(
    #                 self.preds[i], self.gts[i], validity[i]))
    #         rms_dists = np.array(rms_dists)
    #         match_rate = sum(rms_dists != None) / len(self.preds)
    #         mean_rms_dist = rms_dists[rms_dists != None].mean()
    #         return {"match_rate": match_rate,
    #                 "rms_dist": mean_rms_dist}

    def get_metrics(self):
        return self.get_match_rate_and_rms()


class GenEval(object):

    def __init__(self, gen_crys, db_crys, n_samples=1000, eval_model_name=None,
                 ltol=0.2, stol=0.3, angle_tol=5, primitive_cell=True, scale=True,
                 attempt_supercell=True, allow_subset=False, unique_algo=1, unique_sym=True,
                 compute_unn_pg=True, save_unn_indices=False):
        self.gen_crys = gen_crys
        self.db_crys = db_crys
        self.n_samples = n_samples
        self.eval_model_name = eval_model_name
        self.unique_algo = unique_algo
        self.unique_sym = unique_sym
        self.compute_unn_pg = compute_unn_pg
        self.save_unn_indices = save_unn_indices
        self.unique_idx = None
        self.novel_idx = None

        self.matcher = StructureMatcher(
            ltol=ltol,
            stol=stol,
            angle_tol=angle_tol,
            primitive_cell=primitive_cell,
            scale=scale,
            attempt_supercell=attempt_supercell,
            allow_subset=allow_subset,
        )

        valid_crys = [c for c in gen_crys if c.valid]
        if len(valid_crys) >= n_samples:
            sampled_indices = np.random.choice(
                len(valid_crys), n_samples, replace=False)
            self.valid_samples = [valid_crys[i] for i in sampled_indices]
        else:
            raise Exception(
                f"not enough valid crystals in the predicted set: {len(valid_crys)}/{n_samples}")

    def get_validity(self):
        comp_valid = np.array([c.comp_valid for c in self.gen_crys]).mean()
        struct_valid = np.array([c.struct_valid for c in self.gen_crys]).mean()
        valid = np.array([c.valid for c in self.gen_crys]).mean()
        return {"comp_valid": comp_valid,
                "struct_valid": struct_valid,
                "valid": valid}

    def get_comp_diversity(self):
        comp_fps = [c.comp_fp for c in self.valid_samples]
        comp_fps = CompScaler.transform(comp_fps)
        comp_div = get_fp_pdist(comp_fps)
        return {"comp_div": comp_div}

    def get_struct_diversity(self):
        return {"struct_div": get_fp_pdist([c.struct_fp for c in self.valid_samples])}

    def get_density_wdist(self):
        pred_densities = [c.structure.density for c in self.valid_samples]
        gt_densities = [c.structure.density for c in self.db_crys]
        wdist_density = wasserstein_distance(pred_densities, gt_densities)
        return {"wdist_density": wdist_density}

    def get_num_elem_wdist(self):
        pred_nelems = [len(set(c.structure.species))
                       for c in self.valid_samples]
        gt_nelems = [len(set(c.structure.species)) for c in self.db_crys]
        wdist_num_elems = wasserstein_distance(pred_nelems, gt_nelems)
        return {"wdist_num_elems": wdist_num_elems}

    def get_prop_wdist(self):
        if self.eval_model_name is not None:
            pred_props = prop_model_eval(self.eval_model_name, [
                c.dict for c in self.valid_samples])
            gt_props = prop_model_eval(self.eval_model_name, [
                c.dict for c in self.db_crys])
            wdist_prop = wasserstein_distance(pred_props, gt_props)
            return {"wdist_prop": wdist_prop}
        else:
            return {"wdist_prop": None}

    def get_coverage(self):
        cutoff_dict = COV_Cutoffs[self.eval_model_name]
        (cov_metrics_dict, combined_dist_dict) = compute_cov(
            self.gen_crys, self.db_crys,
            struc_cutoff=cutoff_dict["struc"],
            comp_cutoff=cutoff_dict["comp"])
        return cov_metrics_dict

    def get_uniqueness(self):
        desc = f"Evaluating uniqueness [algo {self.unique_algo}]"
        N = len(self.gen_crys)

        if self.unique_algo == 1:
            data, noncrys, row, col = [], [], [], []
            unique = lil_matrix((N, N))

            for i, crys1 in enumerate(tqdm(self.gen_crys[:-1], desc=desc)):
                if crys1.constructed:
                    for j, crys2 in enumerate(self.gen_crys[i + 1:]):
                        j += i + 1
                        if self.matcher.fit(crys1.structure, crys2.structure, symmetric=self.unique_sym):
                            data.append(1)
                            row.append(i)
                            col.append(j)
                else:
                    noncrys.append(i)

            for r, c, d in zip(row, col, data):
                unique[r, c] = d
            diag = eye(N, format="csr")
            diag[noncrys] = 0
            unique += diag

            r_nz, c_nz = unique.tocsr().nonzero()
            for i in range(N):
                non_diag_indices = c_nz[(r_nz == i) & (c_nz != i)]
                for j in non_diag_indices:
                    unique[:, j] = 0
            diag = unique.diagonal()
            unique_idx = np.where(diag != 0)[0]
            uniqueness = {"uniqueness": diag.sum() / N}

        elif self.unique_algo == 2:
            unique_struct = []
            unique_idx = []
            for i, crys1 in enumerate(tqdm(self.gen_crys, desc=desc)):
                if crys1.constructed:
                    unique = True
                    for crys2 in unique_struct:
                        if self.matcher.fit(crys1.structure, crys2.structure, symmetric=self.unique_sym):
                            unique = False
                            break
                    if unique:
                        unique_struct.append(crys1)
                        unique_idx.append(i)
            uniqueness = {"uniqueness": len(unique_struct) / N}

        self.unique_idx = unique_idx
        return uniqueness

    def get_novelty(self):
        matched = 0
        non_novel_idx = []
        for i, crys1 in enumerate(tqdm(self.gen_crys, desc="Evaluating novelty")):
            if crys1.constructed:
                for j, crys2 in enumerate(self.db_crys):
                    if self.matcher.fit(crys1.structure, crys2.structure, symmetric=True):
                        matched += 1
                        non_novel_idx.append(i)
                        break
        self.novel_idx = [k for k in range(len(self.gen_crys)) if k not in non_novel_idx]
        return {"novelty": 1 - matched / len(self.gen_crys)}

    def get_unique_and_novel(self):
        assert self.unique_idx is not None
        assert self.novel_idx is not None
        self.unn_idx = [i for i in self.unique_idx if i in self.novel_idx]
        return {"unique_and_novel": len(self.unn_idx) / len(self.gen_crys)}

    def get_unique_and_novel_by_point_group(self):
        assert self.unique_idx is not None
        assert self.novel_idx is not None
        assert self.unn_idx is not None

        pgs = []
        for i, crys in enumerate(tqdm(self.gen_crys, desc="Evaluating unqiue&novel by point group")):
            if crys.constructed:
                pgs.append(SpacegroupAnalyzer(crys.structure).get_point_group_symbol())
            else:
                pgs.append('not-constructed')

        all_pgs = Counter(pgs)

        def get_ratio(indices):
            counter = Counter([pg for i, pg in enumerate(pgs) if i in indices])
            return {pg: count / all_pgs[pg] for pg, count in counter.items()}

        unqiue_pgs = get_ratio(self.unique_idx)
        novel_pgs = get_ratio(self.novel_idx)
        unn_pgs = get_ratio(self.unn_idx)

        return {
            "unique_pgs": unqiue_pgs,
            "novel_pgs": novel_pgs,
            "unique_and_novel_pgs": unn_pgs,
        }

    def get_metrics(self):
        metrics = {}
        metrics.update(self.get_validity())
        metrics.update(self.get_comp_diversity())
        metrics.update(self.get_struct_diversity())
        metrics.update(self.get_uniqueness())
        metrics.update(self.get_novelty())
        metrics.update(self.get_unique_and_novel())
        if self.compute_unn_pg:
            metrics.update(self.get_unique_and_novel_by_point_group())
        if self.save_unn_indices:
            metrics.update({
                "unique_idx": self.unique_idx,
                "novel_idx": self.novel_idx,
                "unn_idx": self.unn_idx,
            })
        # metrics.update(self.get_density_wdist())
        # metrics.update(self.get_num_elem_wdist())
        # metrics.update(self.get_prop_wdist())
        # metrics.update(self.get_coverage())
        print(metrics)
        return metrics


class OptEval(object):

    def __init__(self, crys, num_opt=100, eval_model_name=None):
        """
        crys is a list of length (<step_opt> * <num_opt>),
        where <num_opt> is the number of different initialization for optimizing crystals,
        and <step_opt> is the number of saved crystals for each intialzation.
        default to minimize the property.
        """
        step_opt = int(len(crys) / num_opt)
        self.crys = crys
        self.step_opt = step_opt
        self.num_opt = num_opt
        self.eval_model_name = eval_model_name

    def get_success_rate(self):
        valid_indices = np.array([c.valid for c in self.crys])
        valid_indices = valid_indices.reshape(self.step_opt, self.num_opt)
        valid_x, valid_y = valid_indices.nonzero()
        props = np.ones([self.step_opt, self.num_opt]) * np.inf
        valid_crys = [c for c in self.crys if c.valid]
        if len(valid_crys) == 0:
            sr_5, sr_10, sr_15 = 0, 0, 0
        else:
            pred_props = prop_model_eval(self.eval_model_name, [
                c.dict for c in valid_crys])
            percentiles = Percentiles[self.eval_model_name]
            props[valid_x, valid_y] = pred_props
            best_props = props.min(axis=0)
            sr_5 = (best_props <= percentiles[0]).mean()
            sr_10 = (best_props <= percentiles[1]).mean()
            sr_15 = (best_props <= percentiles[2]).mean()
        return {"SR5": sr_5, "SR10": sr_10, "SR15": sr_15}

    def get_metrics(self):
        return self.get_success_rate()


def get_crystal_array_list(file_path, batch_idx=0):
    data = load_data(file_path)
    crys_array_list = get_crystals_list(
        data["frac_coords"][batch_idx],
        data["atom_types"][batch_idx],
        data["lengths"][batch_idx],
        data["angles"][batch_idx],
        data["num_atoms"][batch_idx])

    if "input_data_batch" in data:
        batch = data["input_data_batch"]
        if isinstance(batch, dict):
            true_crystal_array_list = get_crystals_list(
                batch["frac_coords"], batch["atom_types"], batch["lengths"],
                batch["angles"], batch["num_atoms"])
        else:
            true_crystal_array_list = get_crystals_list(
                batch.frac_coords, batch.atom_types, batch.lengths,
                batch.angles, batch.num_atoms)
    else:
        true_crystal_array_list = None

    return crys_array_list, true_crystal_array_list


def run_compute_metrics(args):
    all_metrics = {}

    # cfg = load_config(Path(args.root_path))
    # eval_model_name = cfg.data.eval_model_name
    eval_model_name = None

    if "recon" in args.tasks:
        assert args.recon_file_name is not None
        recon_file_path = os.path.join(args.root_path, args.recon_file_name)
        crys_array_list, true_crystal_array_list = get_crystal_array_list(
            recon_file_path)

        print("Get predicted structures")
        pred_crys = p_map(lambda x: Crystal(x, True), crys_array_list)
        print("\nGet ground-truth structures")
        gt_crys = p_map(lambda x: Crystal(x, False), true_crystal_array_list)

        rec_evaluator = RecEval(pred_crys, gt_crys)
        recon_metrics = rec_evaluator.get_metrics()
        all_metrics.update(recon_metrics)

    if "gen" in args.tasks:
        # from ..common.datamodule import CrystDataModule

        assert args.gen_file_name is not None
        gen_file_path = os.path.join(args.root_path, args.gen_file_name)
        crys_array_list, _ = get_crystal_array_list(gen_file_path)
        gen_crys = p_map(lambda x: Crystal(x), crys_array_list, desc="Map gen crystals")

        # datamodule = CrystDataModule(**{k: v for k, v in cfg.data.datamodule.items()
        #                                 if k != "_target_"},
        #                              dataset=cfg.data.datamodule._target_)
        # datamodule.setup(training=True)
        # db_loader = datamodule.train_dataloader()
        # db_crystal_array_list = []
        # for batch in db_loader:
        #     db_crystal_array_list += get_crystals_list(
        #         batch.frac_coords, batch.atom_types, batch.lengths,
        #         batch.angles, batch.num_atoms)

        db_crystal_array_list = pd.read_csv(os.path.abspath(args.dataset_path))["cif"]
        db_crys = p_map(lambda x: Crystal(x, analyze=False), db_crystal_array_list, desc="Map db crystals")

        gen_evaluator = GenEval(
            gen_crys, db_crys, eval_model_name=eval_model_name, n_samples=args.n_samples,
            unique_algo=args.unique_algo, unique_sym=args.unique_sym, compute_unn_pg=args.compute_unn_pg,
            save_unn_indices=args.save_unn_indices,
        )
        gen_metrics = gen_evaluator.get_metrics()
        all_metrics.update(gen_metrics)

    if "opt" in args.tasks:
        assert args.opt_file_name is not None
        opt_file_path = os.path.join(args.root_path, args.opt_file_name)
        crys_array_list, _ = get_crystal_array_list(opt_file_path)
        opt_crys = p_map(lambda x: Crystal(x), crys_array_list)

        opt_evaluator = OptEval(opt_crys, eval_model_name=eval_model_name)
        opt_metrics = opt_evaluator.get_metrics()
        all_metrics.update(opt_metrics)

    print(all_metrics)

    if args.suffix == "":
        metrics_out_file = f"eval_metrics.json"
    else:
        metrics_out_file = f"eval_metrics_{args.suffix}.json"
    metrics_out_file = os.path.join(args.root_path, metrics_out_file)

    # only overwrite metrics computed in the new run.
    if Path(metrics_out_file).exists():
        with open(metrics_out_file, "r") as f:
            written_metrics = json.load(f)
            if isinstance(written_metrics, dict):
                written_metrics.update(all_metrics)
            else:
                with open(metrics_out_file, "w") as f:
                    json.dump(all_metrics, f)
        if isinstance(written_metrics, dict):
            with open(metrics_out_file, "w") as f:
                json.dump(written_metrics, f)
    else:
        with open(metrics_out_file, "w") as f:
            json.dump(all_metrics, f)
