from pysisyphus.calculators.Calculator import Calculator
from pysisyphus.constants import BOHR2ANG, AU2EV
from pysisyphus.xyzloader import make_xyz_str

from yamo.calculators import get_calculator


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
            select a MLFF from calculators

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

        # Preserve XYZ parsing/precision without shared temporary files.
        from io import StringIO
        mol = read(StringIO(string), format="xyz")

        return mol


    def get_energy(self, atoms, coords):
        molecule = self.prepare_mol(atoms, coords)
        result = dict(self.model.get_energy(molecule, return_dict=True))
        if 'energy' in result:
            result['energy'] = result['energy'] / AU2EV
        return result

    def get_forces(self, atoms, coords):
        molecule = self.prepare_mol(atoms, coords)
        result = dict(self.model.get_forces(molecule, return_dict=True))
        if 'forces' in result:
            result['forces'] = result['forces'] * BOHR2ANG / AU2EV
        if 'energy' in result:
            result['energy'] = result['energy'] / AU2EV
        return result

    def get_hessian(self, atoms, coords):
        molecule = self.prepare_mol(atoms, coords)
        result = dict(self.model.get_hessian(molecule, return_dict=True))
        if 'hessian' in result:
            result['hessian'] = result['hessian'] * BOHR2ANG**2 / AU2EV
        if 'energy' in result:
            result['energy'] = result['energy'] / AU2EV
        return result

    def run_calculation(self, atoms, coords):
        return self.get_energy(atoms, coords)

    def __str__(self):
        return f"MLFF({self.method})"
