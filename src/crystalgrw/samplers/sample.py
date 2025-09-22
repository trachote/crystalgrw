import torch

from tqdm import tqdm
from functools import partial

from ..common.data_utils import lattice_params_from_matrix


def sample(model, frac_coords, lattices, atom_types, natoms, ld_kwargs,
           z=None, labels=None, guidance_strength=1, input_encoder=None,
           grad_context=None):
    if (model.encoder is not None) and (z is None):
        assert input_encoder is not None
        z = model.encoder(input_encoder)
        if model.vae:
            _, _, z = model.kld_reparam(z)

    if (natoms is None) and (z is not None):
        natoms = model.fc_natoms(z).argmax(-1) + 1

    x_T = {"frac_coords": frac_coords, "lattices": lattices, "atom_types": atom_types}
    data = {"natoms": natoms, "z": z}

    if labels is None:
        score_fn = partial(model.score_fn, **data)
        desc = "Sampling"
    else:
        score_fn = partial(model.control_score,
                           labels=labels,
                           guidance_strength=guidance_strength,
                           **data,
                           )
        desc = f"Condition-guided sampling [{labels}]"

    T = torch.ones((natoms.size(0),)).to(natoms.device)
    progress_bar = tqdm(total=model.T, desc=desc)

    assert grad_context is not None

    with grad_context:
        x_all, _ = model.sde_fn(
            x_T, T,
            natoms,
            N=model.T,
            score_fn=score_fn,
            stack_data=ld_kwargs.save_traj,
            adaptive_timestep=ld_kwargs.adaptive_timestep,
            sample_type_method=ld_kwargs.sample_type_method,
            embed_noisy_types=model.score_fn.embed_noisy_types,
            progress_bar=progress_bar
        )

    x = x_all[-1] if ld_kwargs.save_traj else x_all
    lengths, angles = lattice_params_from_matrix(
        x["lattices"].view(-1, 3, 3)
    )

    output_dict = {"num_atoms": natoms,
                   "lengths": lengths,
                   "angles": angles,
                   "frac_coords": x["frac_coords"],
                   "atom_types": x["atom_types"],
                   "is_traj": False}

    if ld_kwargs.save_traj:
        coords, atoms, lats = [], [], []
        for x in x_all:
            coords.append(x["frac_coords"])
            atoms.append(x["atom_types"])
            lats.append(torch.cat(lattice_params_from_matrix(
                x["lattices"].view(-1, 3, 3)), dim=-1))
        output_dict.update(dict(
            traj_frac_coords=torch.stack(coords, dim=1),
            traj_atom_types=torch.stack(atoms, dim=1),
            traj_lattices=torch.stack(lats, dim=1),
            is_traj=True))

    return output_dict
