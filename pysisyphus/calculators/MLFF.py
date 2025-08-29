from collections import namedtuple
import os
import sys
import datetime
import random
from pysisyphus.calculators.Calculator import Calculator
from pysisyphus.constants import BOHR2ANG, AU2EV
from pysisyphus.xyzloader import make_xyz_str

from qcservice.calculators import get_calculator


class MLFF(Calculator):

    conf_key = "mlff"

    def __init__(
        self,
        method='leftnet',
        device='cpu',
        **kwargs,
    ):
        """MLFF calculator.

        Wrapper for running energy, gradient and Hessian calculations by
        different MLFF.

        Parameters
        ----------
        method: str
            select a MLFF from calculators in ReactBench
        
        mem : int
            Mememory per core in MB.
        quiet : bool, optional
            Suppress creation of log files.
        """
        super().__init__(**kwargs)

        self.method = method
        self.device = device
        self.model = get_calculator(self.method, device=self.device)

    def prepare_mol(self, atoms, coords):
        from ase.io import read
        coords = coords * BOHR2ANG
        string = make_xyz_str(atoms, coords.reshape((-1, 3)))

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        random_num = random.randint(1000,9999)
        filename = f'mlff_{timestamp}_{random_num}.xyz'
        with open(filename,'w') as f: f.write(string)
        mol = read(filename)
        os.remove(filename)

        return mol


    def get_energy(self, atoms, coords):
        molecule = self.prepare_mol(atoms, coords)
        result = self.model.get_energy(molecule, return_dict=True)
        if 'energy' in result:
            result['energy'] = result['energy'] / AU2EV
        return result

    def get_forces(self, atoms, coords):
        molecule = self.prepare_mol(atoms, coords)
        result = self.model.get_forces(molecule, return_dict=True)
        if 'forces' in result:
            result['forces'] = result['forces'] / AU2EV / BOHR2ANG
        if 'energy' in result:
            result['energy'] = result['energy'] / AU2EV
        return result
    
    def get_hessian(self, atoms, coords):
        molecule = self.prepare_mol(atoms, coords)
        result = self.model.get_hessian(molecule, return_dict=True)
        if 'hessian' in result:
            result['hessian'] = result['hessian'] / AU2EV / BOHR2ANG**2
        if 'energy' in result:
            result['energy'] = result['energy'] / AU2EV
        return result

    def run_calculation(self, atoms, coords):
        return self.get_energy(atoms, coords)

    def __str__(self):
        return f"MLFF({self.method})"