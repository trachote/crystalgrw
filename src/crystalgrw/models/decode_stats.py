import torch
import torch.nn as nn

from ..gnn.embeddings import MAX_ATOMIC_NUM

def build_mlp(in_dim, hidden_dim, fc_num_layers, out_dim, drop_rate=0):
    mods = [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
    if drop_rate > 0.:
        mods += [nn.Dropout(drop_rate)]
    for i in range(fc_num_layers-1):
        mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        if drop_rate > 0.:
            mods += [nn.Dropout(drop_rate)]
    mods += [nn.Linear(hidden_dim, out_dim)]
    return nn.Sequential(*mods)


class MLPDecodeStats(nn.Module):
    def __init__(self, hidden_dim, latent_dim, fc_num_layers, max_atoms,
                 lattice_scale_method=None, teacher_forcing_lattice=False,
                 drop_rate=0):
        super().__init__()
        self.lattice_scale_method = lattice_scale_method
        self.teacher_forcing_lattice = teacher_forcing_lattice

        self.fc_num_atoms = build_mlp(latent_dim, hidden_dim,
                                      fc_num_layers, max_atoms + 1,
                                      drop_rate)
        self.fc_lattice = build_mlp(latent_dim, hidden_dim,
                                    fc_num_layers, 6, drop_rate)
        self.fc_composition = build_mlp(latent_dim, hidden_dim,
                                        fc_num_layers, MAX_ATOMIC_NUM,
                                        drop_rate)

        self.lattice_scaler = None

    def predict_num_atoms(self, z):
        return self.fc_num_atoms(z)

    def predict_composition(self, z, num_atoms, l_and_a=None):
        # z = torch.cat([z, l_and_a], dim=-1)
        z_per_atom = z.repeat_interleave(num_atoms, dim=0)
        pred_composition_per_atom = self.fc_composition(z_per_atom)
        return pred_composition_per_atom

    def predict_lattice(self, z, num_atoms, emb_num_atoms=None):
        self.lattice_scaler.match_device(z)
        if emb_num_atoms is not None:
            z = torch.cat([z, emb_num_atoms], dim=-1)
        pred_lengths_and_angles = self.fc_lattice(z)  # (N, 6)
        scaled_preds = self.lattice_scaler.inverse_transform(
            pred_lengths_and_angles)
        pred_lengths = scaled_preds[:, :3]
        pred_angles = scaled_preds[:, 3:]
        if self.lattice_scale_method == 'scale_length':
            pred_lengths = pred_lengths * num_atoms.view(-1, 1).float() ** (1 / 3)
        # <pred_lengths_and_angles> is scaled.
        return pred_lengths_and_angles, pred_lengths, pred_angles

    def forward(self, z, gt_num_atoms=None, gt_lengths=None, gt_angles=None,
                teacher_forcing=False):
        """
        decode key stats from latent embeddings.
        batch is input during training for teach-forcing.
        """
        if gt_num_atoms is not None:
            num_atoms = self.predict_num_atoms(z)
            lengths_and_angles, lengths, angles = (
                self.predict_lattice(z, gt_num_atoms))

            composition_per_atom = self.predict_composition(z, gt_num_atoms)
            if self.teacher_forcing_lattice and teacher_forcing:
                lengths = gt_lengths
                angles = gt_angles
        else:
            num_atoms = self.predict_num_atoms(z).argmax(dim=-1)

            lengths_and_angles, lengths, angles = (
                self.predict_lattice(z, num_atoms))

            composition_per_atom = self.predict_composition(z, num_atoms)
        return num_atoms, lengths_and_angles, lengths, angles, composition_per_atom


class MLPDecodeAtoms(nn.Module):
    def __init__(self, hidden_dim, latent_dim, fc_num_layers, max_atoms,
                 decode_composition=True, drop_rate=0):
        super().__init__()
        self.decode_composition = decode_composition
        self.fc_num_atoms = build_mlp(latent_dim, hidden_dim,
                                      fc_num_layers, max_atoms+1,
                                      drop_rate)

        if self.decode_composition:
            self.fc_composition = build_mlp(latent_dim, hidden_dim,
                                            fc_num_layers, MAX_ATOMIC_NUM,
                                            drop_rate)

    def predict_num_atoms(self, z):
        return self.fc_num_atoms(z)
    
    def predict_composition(self, z, num_atoms):
        z_per_atom = z.repeat_interleave(num_atoms, dim=0)
        pred_composition_per_atom = self.fc_composition(z_per_atom)
        return pred_composition_per_atom
    
    def forward(self, z, gt_num_atoms=None):
        num_atoms = self.predict_num_atoms(z)
        num_atoms = num_atoms.argmax(dim=-1) if gt_num_atoms is None else num_atoms

        if self.decode_composition:
            if gt_num_atoms is not None:
                composition_per_atom = self.predict_composition(z, gt_num_atoms)
            else:
                composition_per_atom = self.predict_composition(z, num_atoms)
            return num_atoms, composition_per_atom
        else:
            return num_atoms
