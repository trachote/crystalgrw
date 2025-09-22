import pandas as pd
from omegaconf import ValueNode, OmegaConf
from tqdm import tqdm
import numpy as np

import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

from .data_utils import add_scaled_lattice_prop, preprocess_tensors
from . import DTYPE


def load_cif_data(df, primitive=True):
    from pymatgen.core import Structure
    tqdm.pandas()

    def get_data_dict(struct_data):
        if isinstance(struct_data, str):
            struct = Structure.from_str(struct_data, "cif", primitive=primitive)
        elif isinstance(struct_data, dict):
            struct = Structure.from_dict(struct_data)
            if primitive:
                struct = struct.get_primitive_structure()
        else:
            raise TypeError("Structure data must be a cif str or `Structure` dict")

        data = {}
        data["lengths"] = np.array(struct.lattice.lengths)
        data["angles"] = np.array(struct.lattice.angles)
        data["atom_types"] = np.array([x.specie.number for x in struct])
        data["frac_coords"] = np.stack([x.frac_coords for x in struct])
        data["num_atoms"] = len(struct)
        return data

    def get_id(material_id):
        return {"material_id": material_id}

    data_type = "cif" if "cif" in df.columns else "structure_dict"
    return df.progress_apply(
        lambda row: {**get_data_dict(row[data_type]), **get_id(row["material_id"])}, axis=1)


def load_xyz_data(path, data_type="ase_traj"):
    from ase.io import read
    if data_type == "ase_traj":
        atoms = read(path.split(".")[0]+".traj", ":")
    elif data_type == "xyz":
        atoms = read(path.split(".")[0]+".xyz", ":")
    cached_data = []
    
    for i, atom in enumerate(tqdm(atoms)):
        data = {}
        data["lengths"] = atom.cell.lengths()
        data["angles"] = atom.cell.angles()
        data["atom_types"] = [z.number for z in atom]
        data["frac_coords"] = atom.get_scaled_positions().tolist()
        data["num_atoms"] = len(atom)
        data["material_id"] = atom.info["material_id"]

        if atom.constraints:
            data["fixed_atoms"] = atom.constraints[0].index

        try:
            data["forces"] = atom.get_forces()
        except:
            data["forces"] = np.zeros_like(data["frac_coords"]) + np.nan
        cached_data.append(data)

    return cached_data


def load_data(path, data_type=None, prop_list=None, primitive=True):
    if path.endswith(".csv") or path.endswith(".json"):
        if path.endswith(".csv"):
            df = pd.read_csv(path)
        elif path.endswith(".json"):
            df = pd.read_json(path)
        else:
            raise NotImplementedError
        cached_data = load_cif_data(df, primitive).tolist()

    elif path.endswith(".xyz") or path.endswith(".traj"):
        cached_data = load_xyz_data(path, data_type=data_type)

    else:
        try:
            import pickle
            with open(path.replace(".csv", "_processed.pkl"), 'rb') as f:
                cached_data = pickle.load(f)
        except:
            import h5py
            with h5py.File(path.replace(".csv", "_processed.h5"), 'r') as h5file:
                cached_data = []
                for i, group_name in enumerate(tqdm(h5file)):
                    group = h5file[group_name]
                    data = {key: group[key][()] for key in group.keys()}
                    cached_data.append(data)

    if prop_list is not None:
        try:
            material_ids = df["material_id"].tolist()
            props = {prop: df[prop].values if prop in df.columns else np.array([np.nan]*len(df)) for prop in prop_list}
        except Exception as e:
            raise Exception(e)

        for prop in prop_list:
            for data_dict, p, material_id in zip(cached_data, props[prop], material_ids):
                assert material_id == data_dict["material_id"], f"{material_id} does not match {data_dict['material_id']}."
                data_dict.update({prop: p})

    # add_scaled_lattice_prop(cached_data, lattice_scale_method)
    return cached_data


class CrystDataset(Dataset):
    def __init__(self, name: ValueNode, path: ValueNode,
                 prop: ValueNode, niggli: ValueNode, primitive: ValueNode,
                 graph_method: ValueNode, preprocess_workers: ValueNode,
                 lattice_scale_method: ValueNode, data_type="cif", **kwargs):

        super().__init__()
        self.path = path
        self.name = name
        self.prop = prop
        self.niggli = niggli
        self.primitive = primitive
        self.graph_method = graph_method
        self.lattice_scale_method = lattice_scale_method

        self.cached_data = load_data(path, data_type, prop, primitive)

        self.lattice_scaler = None
        self.scaler = None

    def __len__(self) -> int:
        return len(self.cached_data)

    def __getitem__(self, index):
        data_dict = self.cached_data[index]

        data = {}
        data["frac_coords"] = torch.tensor(data_dict["frac_coords"]).to(DTYPE)
        data["atom_types"] = torch.tensor(data_dict["atom_types"]).long()
        data["lengths"] = torch.tensor(data_dict["lengths"]).view(1,-1).to(DTYPE)
        data["angles"] = torch.tensor(data_dict["angles"]).view(1,-1).to(DTYPE)
        data["num_atoms"] = torch.tensor(data_dict["num_atoms"]).long()
        #data["num_nodes"] = torch.tensor(data_dict["num_atoms"]).long()

        if "edge_indices" in data_dict.keys():
            data["edge_index"] = torch.tensor(data_dict["edge_indices"].T).long().contiguous()
            data["to_jimages"] = torch.tensor(data_dict["to_jimages"]).long()
            data["num_bonds"] = data_dict["edge_indices"].shape[0]

        if self.prop is not None:
            prop = torch.tensor([data_dict[prop].item() for prop in self.prop])
            data["y"] = prop.view(1, -1).to(DTYPE)

        data = Data(**data)
        # data.num_nodes = data.num_atoms
        return data

    def __repr__(self) -> str:
        return f"CrystDataset({self.name=}, {self.path=})"


class TensorCrystDataset(Dataset):
    def __init__(self, crystal_array_list, niggli, primitive,
                 graph_method, preprocess_workers,
                 lattice_scale_method, **kwargs):
        super().__init__()
        self.niggli = niggli
        self.primitive = primitive
        self.graph_method = graph_method
        self.lattice_scale_method = lattice_scale_method

        self.cached_data = preprocess_tensors(
            crystal_array_list,
            niggli=self.niggli,
            primitive=self.primitive,
            graph_method=self.graph_method)

        # add_scaled_lattice_prop(self.cached_data, lattice_scale_method, 1)
        self.lattice_scaler = None
        self.scaler = None

    def __len__(self) -> int:
        return len(self.cached_data)

    def __getitem__(self, index):
        data_dict = self.cached_data[index]

        (frac_coords, cart_coords, atom_types, lengths, angles, edge_indices,
         to_jimages, num_atoms) = data_dict['graph_arrays']

        # (frac_coords, atom_types, lengths, angles, edge_indices,
        #  to_jimages, num_atoms) = data_dict['graph_arrays']

        # atom_coords are fractional coordinates
        # edge_index is incremented during batching
        # https://pytorch-geometric.readthedocs.io/en/latest/notes/batching.html
        data = Data(
            frac_coords=torch.Tensor(frac_coords),
            cart_coords=torch.Tensor(cart_coords),
            atom_types=torch.LongTensor(atom_types),
            lengths=torch.Tensor(lengths).view(1, -1),
            angles=torch.Tensor(angles).view(1, -1),
            edge_index=torch.LongTensor(
                edge_indices.T).contiguous(),  # shape (2, num_edges)
            to_jimages=torch.LongTensor(to_jimages),
            num_atoms=num_atoms,
            num_bonds=edge_indices.shape[0],
            num_nodes=num_atoms,  # special attribute used for batching in pytorch geometric
        )
        return data

    def __repr__(self) -> str:
        return f"TensorCrystDataset(len: {len(self.cached_data)})"
