from collections import namedtuple
import torch
import torchani
from torchani.utils import _get_derivatives_not_none, hessian
import numpy as np
import os
from pysisyphus.calculators.Calculator import Calculator
from pysisyphus.constants import BOHR2ANG, AU2EV
from pysisyphus.xyzloader import make_xyz_str

OptResult = namedtuple("OptResult", "opt_geom opt_log")

def compute_hessian(coords, energy, forces=None):
    # compute force if not given (first-order derivative)
    if forces is None:
        forces =  -_get_derivatives_not_none(coords, energy, create_graph=True)
    # get number of element (n_atoms * 3)
    n_comp = forces.view(-1).shape[0]
    # Initialize hessian
    hess = []
    for f in forces.view(-1):
        # compute second-order derivative for each element
        hess_row = _get_derivatives_not_none(coords, -f, retain_graph=True)
        hess.append(hess_row)
    # stack hessian
    hessian = torch.stack(hess)
    return hessian.reshape(n_comp,-1)

class MLFF(Calculator):

    conf_key = "mlff"

    def __init__(
        self,
        method='ani',
        **kwargs,
    ):
        """MLFF calculator.

        Wrapper for running energy, gradient and Hessian calculations by
        different MLFF.

        Parameters
        ----------
        method: str
            select a MLFF from dpa-2, ani-1xnr, mace-off23, leftnet
        
        mem : int
            Mememory per core in MB.
        quiet : bool, optional
            Suppress creation of log files.
        """
        super().__init__(**kwargs)

        self.method = method
        valid_method = ('ani', 'mace', 'dpa2', 'orb', 'left', 'chg', 'alpha', 'left-d', 'orb-d', 'matter', 'equiformerv2')
        assert (
            self.method in valid_method
        ), f"Invalid method argument. Allowed arguments are: {', '.join(valid_method)}!"
        
        # load model
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = "cpu"
        if self.method == 'equiformerv2':
            from equiformerv2.calculator import EquiformerV2Calculator
            self.model = EquiformerV2Calculator(weight='/root/.local/mlff/equiformerv2/ts1x-tuned_epoch2199.ckpt',device="cpu")
        elif self.method == 'alpha':
            from alphanet.calculator import AlphaNetCalculator
            from alphahess1.calculator import AlphaNetHessCalculator
            self.model = AlphaNetCalculator(weight='/root/.local/mlff/alphanet/ts1x-tuned.ckpt',device="cpu")
            self.hessmodel = AlphaNetHessCalculator(weight='/root/alphanet_test/ff-epoch=259-val-totloss=0.1261-val-MAE_E=0.1261-val-MAE_F=0.0000.ckpt',device="cpu")
        elif self.method == 'ani':
            # use a fine-tuned model
            #from torchani.calculator import ANICalculator
            #self.model = ANICalculator('/root/.local/mlff/ani/ts1x-tuned_epoch439.ckpt').model            
            # use a pretrained model
            self.model = torchani.models.ANI1x(periodic_table_index=True).to(self.device).double()
        elif self.method == 'chg':
            from chgnet.model.dynamics import CHGNetCalculator, chgnet_finetuned
            # use a fine-tuned model
            model = chgnet_finetuned(device='cpu')
            self.model = CHGNetCalculator(model, device=self.device)
            # use a pretrained model
            #self.model = CHGNetCalculator(device=self.device)
        elif self.method == 'dpa2':
            from deepmd.infer import DeepEval as DeepPot
            #deepeval = DeepPot("/root/.local/mlff/dpa2/dpa2-26head.pt", head='Domains_Drug')
            deepeval = DeepPot("/root/.local/mlff/dpa2/ts1x-tuned_epoch1000.pt", head='Domains_Drug')
            self.model = deepeval.deep_eval.dp.to(self.device)
        elif self.method[:4] == 'left':
            from oa_reactdiff.trainer.calculator import LeftNetCalculator
            if '-d' in self.method:
                self.model = LeftNetCalculator('/root/.local/mlff/leftnet/ts1x-tuned_df_epoch799.ckpt', device=self.device, use_autograd=False)
            else:
                self.model = LeftNetCalculator('/root/.local/mlff/leftnet/ts1x-tuned_epoch999.ckpt', device=self.device, use_autograd=True)
        elif self.method == 'mace':
            from mace.calculators import mace_off, mace_off_finetuned
            # choose to use fine-tuned/pretrained model
            if torch.cuda.is_available():
                #calc = mace_off(model="medium", default_dtype="float64", device='cuda')
                calc = mace_off_finetuned(device="cuda")
            else:
                #calc = mace_off(model="medium", default_dtype="float64", device='cpu')
                calc = mace_off_finetuned(device="cpu")
            self.model = calc
        elif self.method[:3] == 'orb':
            self.device = 'cpu'
            from orb_models.forcefield import pretrained
            # use pretrained model
            #orbff = pretrained.orb_v2()
            # use fine-tuned model
            orbff = pretrained.orb_v2_finetuned(device=self.device)
            self.model = orbff
        elif self.method == 'matter':
            from mattersim.forcefield import MatterSimCalculator
            calc = MatterSimCalculator(load_path="MatterSim-v1.0.0-5M.pth",device=self.device)
            self.model = calc

    def prepare_mol(self, atoms, coords):
        from ase.io import read
        coords = coords * BOHR2ANG
        string = make_xyz_str(atoms, coords.reshape((-1, 3)))
        with open('mlff.xyz','w') as f: f.write(string)
        mol = read('mlff.xyz')
        os.remove('mlff.xyz')
        return mol

    def store_and_track(self, results, func, atoms, coords):
        if self.track:
            self.store_overlap_data(atoms, coords)
            if self.track_root():
                # Redo the calculation with the updated root
                results = func(atoms, coords, **prepare_kwargs)
        return results

    def get_energy(self, atoms, coords):
        molecule = self.prepare_mol(atoms, coords)
        if self.method == 'ani':
            species = torch.tensor(molecule.get_atomic_numbers(), dtype=torch.long).to(self.device).unsqueeze(0)
            coordinates = torch.from_numpy(molecule.get_positions()).unsqueeze(0).to(self.device).requires_grad_(True)
            # compute energy
            energy = self.model((species, coordinates)).energies
            energy = energy.item()
        elif self.method == 'dpa2':
            species = torch.tensor([i-1 for i in molecule.get_atomic_numbers()], dtype=torch.long).to(self.device).unsqueeze(0)
            coordinates = torch.from_numpy(molecule.get_positions()).unsqueeze(0).to(self.device).requires_grad_(True)
            # compute energy
            output = self.model(coordinates, species)
            energy = output[0]['energy'].item() / AU2EV
        elif self.method[:3] == 'orb':
            from orb_models.forcefield.atomic_system import SystemConfig,ase_atoms_to_atom_graphs
            batch = ase_atoms_to_atom_graphs(molecule,system_config=\
                                             SystemConfig(radius=10.0, max_num_neighbors=20),brute_force_knn=None).to(self.device)
            out = self.model.predict(batch)
            energy = out['graph_pred'].item() / AU2EV
        else: # work for MACE, LEFTNET, CHGNET, ALPHANET
            # set box for chgnet 
            if self.method == 'chg': molecule.cell = [100,100,100]
            molecule.calc = self.model
            # compute energy
            energy = molecule.get_potential_energy() / AU2EV 
            
        results = {
            "energy": energy,
        }
        return results

    def get_forces(self, atoms, coords):
        molecule = self.prepare_mol(atoms, coords)
        if self.method == 'ani':
            species = torch.tensor(molecule.get_atomic_numbers(), dtype=torch.long).to(self.device).unsqueeze(0)
            coordinates = torch.from_numpy(molecule.get_positions()).unsqueeze(0).to(self.device).requires_grad_(True)
            # compute force
            energy = self.model((species, coordinates)).energies
            forces = -_get_derivatives_not_none(coordinates,energy).detach().cpu().numpy() * BOHR2ANG
            energy = energy.item()
        elif self.method == 'dpa2':
            species = torch.tensor([i-1 for i in molecule.get_atomic_numbers()], dtype=torch.long).to(self.device).unsqueeze(0)
            coordinates = torch.from_numpy(molecule.get_positions()).unsqueeze(0).to(self.device).requires_grad_(True)
            # compute energy
            output = self.model(coordinates, species)
            energy = output[0]['energy'].item() / AU2EV
            forces = output[0]['force'].detach().numpy() / AU2EV * BOHR2ANG
        elif self.method[:3] == 'orb':
            from orb_models.forcefield.atomic_system import SystemConfig,ase_atoms_to_atom_graphs
            batch = ase_atoms_to_atom_graphs(molecule,system_config=\
                                             SystemConfig(radius=10.0, max_num_neighbors=20),brute_force_knn=None).to(self.device)
            out = self.model.predict(batch)
            energy = out['graph_pred'] / AU2EV
            if '-d' in self.method:
                forces = out['node_pred'].detach().numpy() / AU2EV * BOHR2ANG
            else:
                forces = -_get_derivatives_not_none(batch.positions,energy).detach().cpu().numpy() * BOHR2ANG
            energy = energy.item()
        else: # work for MACE, LEFTNET, CHGNET, and ALPHANET
            # set box for chgnet
            if self.method == 'chg': molecule.cell = [100,100,100]
            molecule.calc = self.model
            # compute energy
            energy = molecule.get_potential_energy() / AU2EV 
            forces = molecule.get_forces() / AU2EV * BOHR2ANG 

        results = {
            "energy": energy,
            "forces": forces.flatten(),
        }

        return results

    def get_hessian(self, atoms, coords):
        molecule = self.prepare_mol(atoms, coords)
        if self.method == 'alpha':
            from alphanet.calculator import mols_to_batch
            data = mols_to_batch([molecule]).to(self.device)
            energy, forces = self.model.model.forward(data)
            hessian = compute_hessian(data.pos, energy, forces).detach().cpu().numpy() / AU2EV * BOHR2ANG * BOHR2ANG
            energy = energy.item() / AU2EV
        elif self.method == 'ani':
            species = torch.tensor(molecule.get_atomic_numbers(), dtype=torch.long).to(self.device).unsqueeze(0)
            coordinates = torch.from_numpy(molecule.get_positions()).unsqueeze(0).to(self.device).requires_grad_(True)
            # compute hessian
            energy = self.model((species, coordinates)).energies
            hessian = compute_hessian(coordinates,energy).detach().cpu().numpy() * BOHR2ANG * BOHR2ANG
            energy = energy.item()
        elif self.method == 'chg':
            from pymatgen.io.ase import AseAtomsAdaptor
            molecule.cell = [100,100,100]
            structure = AseAtomsAdaptor.get_structure(molecule)
            model = self.model.model
            model.is_intensive = False
            model.composition_model.is_intensive=False
            graph = model.graph_converter(structure)
            model_prediction = model.forward([graph.to(self.device)],task="ef")
            energy = model_prediction["e"][0]
            forces = model_prediction["f"][0]
            pos    = model_prediction["bg"].atom_positions[0]
            hessian= compute_hessian(pos, energy, forces).detach().cpu().numpy() / AU2EV * BOHR2ANG * BOHR2ANG
            energy = energy.item() / AU2EV
        elif self.method == 'dpa2':
            species = torch.tensor([i-1 for i in molecule.get_atomic_numbers()], device=self.device, dtype=torch.long).unsqueeze(0)
            coordinates = torch.from_numpy(molecule.get_positions()).unsqueeze(0).to(self.device).requires_grad_(True)
            # compute energy
            output  = self.model(coordinates, species)
            energy  = output[0]['energy'].item() / AU2EV
            forces  = output[0]['force'] / AU2EV
            # use autograd to compute hessian
            hessian = compute_hessian(coordinates, energy, forces).detach().cpu().numpy() * BOHR2ANG * BOHR2ANG 
        elif self.method[:4] == 'left':
            from oa_reactdiff.trainer.calculator import mols_to_batch
            data = mols_to_batch([molecule]).to(self.device)
            # compute energy and force
            if '-d' in self.method:
                energy, forces = self.model.model.forward(data)
            else: 
                energy, forces = self.model.model.forward_autograd(data)
            # use autograd of force to compute hessian
            hessian = compute_hessian(data.pos, energy, forces).detach().cpu().numpy() / AU2EV * BOHR2ANG * BOHR2ANG
            energy = energy.item() / AU2EV            
        elif self.method == 'mace':
            molecule.calc = self.model
            # compute hessian
            hessian = self.model.get_hessian(atoms=molecule).reshape(molecule.get_number_of_atoms()*3,\
                                                                     molecule.get_number_of_atoms()*3) / AU2EV * BOHR2ANG * BOHR2ANG
            energy = molecule.get_potential_energy() / AU2EV 
        elif self.method[:3] == 'orb':
            from orb_models.forcefield.atomic_system import SystemConfig,ase_atoms_to_atom_graphs
            batch = ase_atoms_to_atom_graphs(molecule,system_config=\
                                             SystemConfig(radius=10.0, max_num_neighbors=20),brute_force_knn=None).to(self.device)
            # compute energy
            out    = self.model.predict(batch)
            energy = out['graph_pred'] / AU2EV
            # compute force
            if '-d' in self.method:
                forces = out['node_pred'] / AU2EV
                hessian= compute_hessian(batch.positions, energy, forces).detach().cpu().numpy() * BOHR2ANG * BOHR2ANG
            else:
                hessian= compute_hessian(batch.positions, energy).detach().cpu().numpy() * BOHR2ANG * BOHR2ANG
            # compute hessian
            energy = energy.item()
        elif self.method == 'matter':
            from mattersim.datasets.utils.build import build_dataloader
            from mattersim.forcefield.potential import batch_to_dict
            # prepare input dict
            model = self.model.potential
            args_dict = {"batch_size": 1, "only_inference": 1}
            dataloader = build_dataloader([molecule], model_type=model.model_name, **args_dict)
            graph = [graph for graph in dataloader][0]
            inp = batch_to_dict(graph)
            out = model.forward(inp, include_forces=True, include_stresses=False)
            energy = out['total_energy'][0] / AU2EV
            forces = out['forces'] / AU2EV
            hessian = compute_hessian(inp['atom_pos'], energy, forces).detach().cpu().numpy() * BOHR2ANG * BOHR2ANG
            energy = energy.item()
        elif self.method == 'equiformerv2':
            from equiformerv2.calculator import mols_to_batch
            data = mols_to_batch([molecule])
            energy, forces = self.model.model.forward(data)
            hessian = compute_hessian(data.pos, energy, forces).detach().cpu().numpy() / AU2EV * BOHR2ANG * BOHR2ANG
            energy = energy.item() / AU2EV

        results = {
            "energy": energy,
            "hessian": hessian,
        }

        return results

    def run_calculation(self, atoms, coords):
        return self.get_energy(atoms, coords)

    def __str__(self):
        return f"MLFF({self.method})"
