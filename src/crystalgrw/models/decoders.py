import torch
import torch.nn as nn

from ..gnn.embeddings import MAX_ATOMIC_NUM

from ..gnn.mlp import FourierFeatures
from ..common.data_utils import lattice_params_from_matrix
from .utils import get_timestep_embedding, default_init


class BaseDecoder(nn.Module):
    def __init__(
            self,
            hidden_dim=128,
            latent_dim=256,
            max_neighbors=20,
            radius=6.,
            scale_file=None,
            condition_time=None,
            num_targets=1,
            regress_logvars=False,
            time_dim=128,
            noisy_atom_types=False,
            embed_lattices=True,
            embed_coord=False,
            condition_dim=0,
            regress_energy=False,
            regress_atoms=True,
            regress_forces=True,
            regress_lattices=True,
            is_decode=True,
    ):
        super(BaseDecoder, self).__init__()
        self.cutoff = radius
        self.max_num_neighbors = max_neighbors
        self.regress_logvars = regress_logvars
        self.condition_time = condition_time
        self.noisy_atom_types = noisy_atom_types
        self.regress_energy = regress_energy
        self.regress_forces = regress_forces
        self.regress_atoms = regress_atoms
        self.regress_lattices = regress_lattices
        self.embed_lattices = embed_lattices
        self.embed_coord = embed_coord
        self.is_decode = is_decode
        self.keys = {"forces": "frac_coords", "atoms": "atom_types",
                     "lattices": "lattices"}

        if is_decode:
            assert latent_dim != 0
        else:
            assert latent_dim == 0

        if condition_time == 'None':
            self.time_dim = 0
        elif condition_time == 'constant':
            self.time_dim = 1
        elif condition_time == 'embed':
            self.time_dim = time_dim
            # Condition on noise levels.
            # self.fc_time = nn.Embedding(self.timesteps, self.time_dim)
            self.fc_time = nn.Sequential(nn.Linear(self.time_dim, self.time_dim * 4),
                                         nn.ReLU(),
                                         nn.Linear(self.time_dim * 4, self.time_dim)
                                         )
            for i in [0, 2]:
                self.fc_time[i].weight.data = default_init()(self.fc_time[i].weight.data.shape)
                nn.init.zeros_(self.fc_time[i].bias)

        if self.noisy_atom_types:
            noisy_atom_dim = hidden_dim
            self.noisy_atom_emb = nn.Sequential(nn.Linear(MAX_ATOMIC_NUM - 1, noisy_atom_dim * 4),
                                                nn.ReLU(),
                                                nn.Linear(noisy_atom_dim * 4, noisy_atom_dim)
                                                )
            for i in [0, 2]:
                nn.init.xavier_uniform_(self.noisy_atom_emb[i].weight.data)
                nn.init.zeros_(self.noisy_atom_emb[i].bias)
        else:
            noisy_atom_dim = 0

        if self.embed_lattices:
            lattice_dim = hidden_dim
            self.lattice_emb = nn.Sequential(nn.Linear(9, lattice_dim),
                                             nn.ReLU(),
                                             nn.Linear(lattice_dim, lattice_dim))
        else:
            lattice_dim = 0

        if self.embed_coord:
            coord_dim = hidden_dim
            self.coord_emb = nn.Sequential(nn.Linear(3, coord_dim),
                                           nn.ReLU(),
                                           nn.Linear(coord_dim, coord_dim))
        else:
            coord_dim = 0

        self.extra_dim = noisy_atom_dim + lattice_dim + coord_dim + condition_dim

        self.gnn = nn.Module()
        self.gnn.forward = lambda *args, **kwargs: None

        # if regress_atoms:
        #     atom_hidden_dim = hidden_dim + latent_dim + self.time_dim + self.extra_dim
        #     self.fc_atom = nn.Linear(atom_hidden_dim, MAX_ATOMIC_NUM)

    def bundle_feats(self, z, t, noisy_atom_types,
                     noisy_lattices, cond_feat, natoms):
        node_feats = []

        if z is not None:
            node_feats.append(z.repeat_interleave(natoms, dim=0))

        if t is not None:
            if self.condition_time == "embed":
                assert len(t.shape) == 1
                time_emb = get_timestep_embedding(t, self.time_dim)
                time_emb = self.fc_time(time_emb)
            elif self.condition_time == "constant":
                time_emb = t
            elif self.condition_time == "neglect":
                time_emb = None
            else:
                raise NotImplementedError
            time_emb = time_emb.repeat_interleave(natoms, dim=0)
            node_feats.append(time_emb)

        if self.noisy_atom_types:
            node_feats.append(self.noisy_atom_emb(noisy_atom_types))

        if self.embed_lattices:
            lattice_feats = noisy_lattices.view(-1, 9)
            lattice_feats = self.lattice_emb(lattice_feats)
            node_feats.append(lattice_feats.repeat_interleave(natoms, dim=0))

        if cond_feat is not None:
            node_feats.append(cond_feat)

        return node_feats

    def key_map(self, outs):
        for k in list(outs.keys()):
            if k in self.keys:
                outs[self.keys[k]] = outs.pop(k)
            else:
                outs.pop(k)
        return outs

    def forward(self, t, frac_coords, atom_types, natoms, lattices=None,
                noisy_atom_types=None, lengths=None, angles=None,
                z=None, cond_feat=None, batch=None):

        if batch is None:
            batch = torch.arange(
                natoms.size(0), device=natoms.device
            ).repeat_interleave(natoms, dim=0)

        node_feats = self.bundle_feats(z, t, noisy_atom_types,
                                       lattices, cond_feat, natoms)

        if lattices is not None:
            assert lattices.shape[-1] == 3
            lengths, angles = lattice_params_from_matrix(lattices)

        outs = self.gnn(
            node_feats=node_feats,
            pos=frac_coords,
            atomic_numbers=atom_types - 1,  # set an atom index to start from zero.
            natoms=natoms,
            lengths=lengths,
            angles=angles,
            edge_index=None,
            to_jimages=None,
            nbonds=None,
            batch=batch,
        )

        outs = self.key_map(outs)

        if self.regress_atoms:
            outs["atom_types"] = torch.softmax(outs["atom_types"], dim=1)
        return outs


class GemNetTDecoder(BaseDecoder):
    def __init__(
            self,
            hidden_dim=128,
            latent_dim=0,
            max_neighbors=20,
            radius=6.,
            scale_file=None,
            condition_time=None,
            num_targets=1,
            regress_logvars=False,
            time_dim=128,
            noisy_atom_types=False,
            regress_energy=False,
            regress_forces=True,
            regress_atoms=True,
            regress_lattices=True,
            embed_lattices=True,
            embed_coord=False,
            condition_dim=0,
            is_decode=True,
    ):
        from ..gnn.gemnet.gemnet import GemNetT
        super(GemNetTDecoder, self).__init__(
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            max_neighbors=max_neighbors,
            radius=radius,
            condition_time=condition_time,
            regress_logvars=regress_logvars,
            time_dim=time_dim,
            noisy_atom_types=noisy_atom_types,
            regress_atoms=regress_atoms,
            regress_lattices=regress_lattices,
            embed_lattices=embed_lattices,
            embed_coord=embed_coord,
            condition_dim=condition_dim,
            is_decode=is_decode,
        )

        self.gnn = GemNetT(
            num_targets=num_targets,
            latent_dim=latent_dim,
            emb_size_atom=hidden_dim,
            emb_size_edge=hidden_dim,
            regress_forces=True,
            regress_logvars=self.regress_logvars,
            cutoff=self.cutoff,
            max_neighbors=self.max_num_neighbors,
            otf_graph=True,
            scale_file=scale_file,
            condition_time=self.condition_time,
            time_dim=self.time_dim,
            noisy_atom_types=False,  # self.noisy_atom_types,
            extra_dim=self.extra_dim,
            regress_atoms=self.regress_atoms,
            regress_lattices=self.regress_lattices,
        )


class EquiformerV2Decoder(BaseDecoder):

    def __init__(
            self,
            hidden_dim=128,
            latent_dim=0,
            max_neighbors=20,
            radius=12.,
            scale_file=None,
            condition_time=None,
            num_targets=1,
            regress_logvars=False,
            time_dim=128,
            noisy_atom_types=False,
            regress_energy=False,
            regress_forces=True,
            regress_atoms=True,
            regress_lattices=True,
            embed_lattices=True,
            embed_coord=False,
            condition_dim=0,
            is_decode=True,
            atom_readout="so2",

            use_pbc=True,
            otf_graph=True,
            max_num_elements=MAX_ATOMIC_NUM,

            num_layers=8,
            sphere_channels=128,
            attn_hidden_channels=64,
            num_heads=8,
            attn_alpha_channels=64,
            attn_value_channels=16,
            ffn_hidden_channels=128,

            norm_type='rms_norm_sh',

            lmax_list=[4],
            mmax_list=[2],
            grid_resolution=None,

            num_sphere_samples=128,

            edge_channels=128,
            use_atom_edge_embedding=True,
            share_atom_edge_embedding=False,
            use_m_share_rad=False,
            distance_function="gaussian",
            num_distance_basis=600,

            attn_activation='scaled_silu',
            use_s2_act_attn=False,
            use_attn_renorm=True,
            ffn_activation='scaled_silu',
            use_gate_act=False,
            use_grid_mlp=False,
            use_sep_s2_act=True,

            alpha_drop=0.1,
            drop_path_rate=0.05,
            proj_drop=0.0,

            weight_init='normal',
    ):
        from ..gnn.equiformer_v2.equiformer_v2 import EquiformerV2
        super(EquiformerV2Decoder, self).__init__(
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            max_neighbors=max_neighbors,
            radius=radius,
            scale_file=scale_file,
            condition_time=condition_time,
            num_targets=num_targets,
            regress_logvars=regress_logvars,
            time_dim=time_dim,
            noisy_atom_types=noisy_atom_types,
            regress_energy=regress_energy,
            regress_forces=regress_forces,
            regress_atoms=regress_atoms,
            regress_lattices=regress_lattices,
            embed_lattices=embed_lattices,
            embed_coord=embed_coord,
            condition_dim=condition_dim,
            is_decode=is_decode,
        )

        self.gnn = EquiformerV2(
            # num_targets=num_targets,
            # emb_size_atom=hidden_dim,
            # emb_size_edge=hidden_dim,
            regress_energy=self.regress_energy,
            regress_atoms=self.regress_atoms,
            regress_forces=self.regress_forces,
            regress_lattices=self.regress_lattices,
            latent_dim=latent_dim,
            time_dim=self.time_dim,
            extra_dim=self.extra_dim,
            atom_readout=atom_readout,

            use_pbc=use_pbc,
            otf_graph=otf_graph,
            max_neighbors=self.max_num_neighbors,
            max_radius=self.cutoff,
            max_num_elements=max_num_elements,

            num_layers=num_layers,
            sphere_channels=sphere_channels,
            attn_hidden_channels=attn_hidden_channels,
            num_heads=num_heads,
            attn_alpha_channels=attn_alpha_channels,
            attn_value_channels=attn_value_channels,
            ffn_hidden_channels=ffn_hidden_channels,

            norm_type=norm_type,

            lmax_list=lmax_list,
            mmax_list=mmax_list,
            grid_resolution=grid_resolution,

            num_sphere_samples=num_sphere_samples,

            edge_channels=edge_channels,
            use_atom_edge_embedding=use_atom_edge_embedding,
            share_atom_edge_embedding=share_atom_edge_embedding,
            use_m_share_rad=use_m_share_rad,
            distance_function=distance_function,
            num_distance_basis=num_distance_basis,

            attn_activation=attn_activation,
            use_s2_act_attn=use_s2_act_attn,
            use_attn_renorm=use_attn_renorm,
            ffn_activation=ffn_activation,
            use_gate_act=use_gate_act,
            use_grid_mlp=use_grid_mlp,
            use_sep_s2_act=use_sep_s2_act,

            alpha_drop=alpha_drop,
            drop_path_rate=drop_path_rate,
            proj_drop=proj_drop,

            weight_init=weight_init,
        )
