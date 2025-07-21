from collections import namedtuple
import os
import sys
from pysisyphus.calculators.Calculator import Calculator
from pysisyphus.constants import BOHR2ANG
from pysisyphus.xyzloader import make_xyz_str

REACTBENCH_PATH = os.environ.get('REACTBENCH_PATH', "/root/ReactBench")

if REACTBENCH_PATH not in sys.path:
    sys.path.insert(0, REACTBENCH_PATH)

from ReactBench.Calculators import get_mlff, AVAILABLE_CALCULATORS

OptResult = namedtuple("OptResult", "opt_geom opt_log")


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
        valid_method = AVAILABLE_CALCULATORS
        assert (
            self.method in valid_method
        ), f"Invalid method argument. Allowed arguments are: {', '.join(valid_method)}!"
        
        self.model = get_mlff(self.method, device=self.device)

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
        results = self.model.get_energy(molecule)
        return results

    def get_forces(self, atoms, coords):
        molecule = self.prepare_mol(atoms, coords)
        results = self.model.get_forces(molecule)
        return results

    def get_hessian(self, atoms, coords):
        molecule = self.prepare_mol(atoms, coords)
        results = self.model.get_hessian(molecule)
        return results

    def run_calculation(self, atoms, coords):
        return self.get_energy(atoms, coords)

    def __str__(self):
        return f"MLFF({self.method})"
