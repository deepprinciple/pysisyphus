import os
import shutil
import warnings

import numpy as np
import pyscf
from pyscf import grad, gto, lib, hessian, qmmm, tddft, solvent

try:
    from gpu4pyscf.drivers.dft_3c_driver import parse_3c, MethodType, gen_disp_fun, gen_disp_grad_fun, gen_disp_hess_fun
except Exception as e:
    print()
    print("Either you don't have a GPU, so cupy failed with \"CUDA driver version is insufficient for CUDA runtime version\"")
    print("Or GPU4PySCF version is lower than 1.3.1, so import failed with \"No module named 'gpu4pyscf.drivers.dft_3c_driver'\"")
    print("Or some other problem occurs when trying to load parse_3c() function from gpu4pyscf.")
    print("Please contact gpu4pyscf developers for more info.")
    print()
    #raise e
    pass

from pysisyphus.calculators.OverlapCalculator import OverlapCalculator
from pysisyphus.helpers import geom_loader


class PySCF(OverlapCalculator):
    conf_key = "pyscf"
    drivers = {
        # key: (method, unrestricted?)
        ("dft", False): pyscf.dft.RKS,
        ("dft", True): pyscf.dft.UKS,
        ("scf", False): pyscf.scf.RHF,
        ("scf", True): pyscf.scf.UHF,
        ("mp2", False): pyscf.mp.MP2,
        ("mp2", True): pyscf.mp.UMP2,
    }
    multisteps = {
        "scf": ("scf",),
        "dft": ("dft",),
        "mp2": ("scf", "mp2"),
        "tddft": ("dft", "tddft"),
        "tda": ("dft", "tda"),
    }
    pruning_method = {
        "nwchem": pyscf.dft.gen_grid.nwchem_prune,
        "sg1": pyscf.dft.gen_grid.sg1_prune,
        "treutler": pyscf.dft.gen_grid.treutler_prune,
        "none": None,
    }

    def __init__(
        self,
        basis,
        xc=None,
        method="scf",
        solvation_model=False,
        solvent_epi=78.3553,
        ecp=None,
        pseudo=None,
        root=None,
        nstates=None,
        auxbasis=None,
        keep_chk=True,
        verbose=0,
        unrestricted=None,
        grid_level=3,
        pruning="nwchem",
        use_gpu=False,
        atom_grid=None,
        derivative_grid_response=False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.basis = basis
        self.xc = xc
        self.method = method.lower()
        self.solvation_model = solvation_model
        self.solvent_epi = solvent_epi
        if self.method in ("tda", "tddft") and self.xc is None:
            self.multisteps[self.method] = ("scf", self.method)
        if self.xc and self.method != "tddft":
            self.method = "dft"

        self.ecp = ecp
        self.pseudo = pseudo
        if isinstance(xc, str) and len(self.xc) > 13 and self.xc[-13:] == "3c_customized":
            pyscf_xc, nlc, basis, ecp, (xc_disp, disp), xc_gcp = parse_3c(xc[:-11])
            self.parameters_3c = pyscf_xc, nlc, basis, ecp, (xc_disp, disp), xc_gcp
        else:
            self.parameters_3c = None

        self.root = root
        self.nstates = nstates
        if self.method == "tddft":
            assert self.nstates, "nstates must be set with method='tddft'!"
        if self.track:
            assert self.root <= self.nstates, (
                "'root' must be smaller " "than 'nstates'!"
            )
        self.auxbasis = auxbasis
        self.keep_chk = keep_chk
        self.verbose = int(verbose)
        if unrestricted is None:
            self.unrestricted = self.mult > 1
        else:
            self.unrestricted = unrestricted
        self.grid_level = int(grid_level)
        self.atom_grid = atom_grid
        self.derivative_grid_response = derivative_grid_response
        self.pruning = pruning.lower()

        self.chkfile = None
        self.out_fn = "pyscf.out"

        self.use_gpu = use_gpu

        lib.num_threads(self.pal)

    @staticmethod
    def geom_from_fn(fn, **kwargs):
        geom = geom_loader(fn)
        geom.set_calculator(PySCF(**kwargs))
        return geom

    def set_scf_params(self, mf):
        mf.conv_tol = 1e-8
        mf.max_cycle = 150

    def build_grid(self, mf):
        mf.grids.level = self.grid_level
        if self.atom_grid is not None:
            mf.grids.atom_grid = atom_grid
        mf.grids.prune = self.pruning_method[self.pruning]
        mf.grids.build()

    def prepare_mf(self, mf):
        # Method can be overriden in a subclass to modify the mf object.
        if self.use_gpu:
            return mf.to_gpu()
        else:
            return mf

    def get_driver(self, step, mol=None, mf=None):
        def _get_driver():
            return self.drivers[(step, self.unrestricted)]

        if mol and (step == "dft"):
            driver = _get_driver()
            mf = driver(mol)
            mf.xc = self.xc
            self.set_scf_params(mf)
            self.build_grid(mf)
            mf = self.prepare_mf(mf)
        elif mol and (step == "scf"):
            driver = _get_driver()
            mf = driver(mol)
            self.set_scf_params(mf)
            mf = self.prepare_mf(mf)
        elif mf and (step == "mp2"):
            mp2_mf = _get_driver()
            mf = mp2_mf(mf)
        elif mf and (step == "tddft"):
            mf = pyscf.tddft.TDDFT(mf)
            mf.nstates = self.nstates
        elif mf and (step == "tda"):
            mf = pyscf.tddft.TDA(mf)
            mf.nstates = self.nstates
        else:
            raise Exception("Unknown method '{step}'!")

        # set up solvation model
        if self.solvation_model:
            if self.solvation_model in ['IEF-PCM', 'C-PCM', 'SS(V)PE', 'COSMO']:
                mf = mf.PCM()
                mf.with_solvent.method = self.solvation_model
                mf.with_solvent.eps = self.solvent_epi
            elif self.solvation_model == 'DDCOSMO':
                mf = mf.DDCOSMO()
                mf.with_solvent.eps = self.solvent_epi
            elif self.solvation_model == 'SMD':
                mf = mf.SMD()
                mf.with_solvent.eps = self.solvent_epi
            else:
                print(f"Solvation model {self.solvation_model} is not supported in GPU4PySCF, treat as Null")

        return mf

    def prepare_mol(self, atoms, coords, build=True):
        mol = gto.Mole()
        mol.atom = [(atom, c) for atom, c in zip(atoms, coords.reshape(-1, 3))]
        mol.basis = self.basis
        if self.ecp is not None:
            mol.ecp = self.ecp
        if self.pseudo is not None:
            mol.pseudo = self.pseudo
        if self.parameters_3c is not None:
            pyscf_xc, nlc, basis, ecp, (xc_disp, disp), xc_gcp = self.parameters_3c
            if self.basis != basis:
                warnings.warn(f"The basis provided in the input ({self.basis}) is not the required basis ({basis}) for the 3c method. "
                              f"The 3c basis ({basis}) will be used.")
            mol.basis = basis
            mol.ecp = ecp
        mol.unit = "Bohr"
        mol.charge = self.charge
        mol.spin = self.mult - 1
        mol.symmetry = False
        mol.verbose = self.verbose
        # Personally, I patched mole.py so it doesn't print
        # messages regarding the output-file for verbose > QUIET.
        # Just uncomment the lines after
        #   if self.verbose > logger.QUIET:
        #       ...
        # in 'mole.Mole.build'. Around line 2046 for pyscf 1.6.5.
        # Search for "output file" in gto/mole.py
        # Search for "Large deviations found" in scf/{uhf,dhf,ghf}.py
        mol.output = self.make_fn(self.out_fn)
        mol.max_memory = self.mem * self.pal
        if build:
            mol.build(parse_arg=False)
        return mol

    def prepare_input(self, atoms, coords, build=True):
        mol = self.prepare_mol(atoms, coords, build=build)
        assert mol._built, "Please call mol.build(parse_arg=False)!"
        return mol

    def store_and_track(self, results, func, atoms, coords, **prepare_kwargs):
        if self.track:
            self.store_overlap_data(atoms, coords)
            if self.track_root():
                # Redo the calculation with the updated root
                results = func(atoms, coords, **prepare_kwargs)
        results["all_energies"] = self.parse_all_energies()
        return results

    def get_energy(self, atoms, coords, **prepare_kwargs):
        point_charges = prepare_kwargs.get("point_charges", None)

        mol = self.prepare_input(atoms, coords)
        mf = self.run(mol, point_charges=point_charges)
        results = {
            "energy": mf.e_tot,
        }
        results = self.store_and_track(
            results, self.get_energy, atoms, coords, **prepare_kwargs
        )
        return results

    def get_forces(self, atoms, coords, **prepare_kwargs):
        point_charges = prepare_kwargs.get("point_charges", None)

        mol = self.prepare_input(atoms, coords)
        mf = self.run(mol, point_charges=point_charges)
        grad_driver = mf.Gradients()
        if self.root:
            grad_driver.state = self.root
        if self.parameters_3c is not None:
            pyscf_xc, nlc, basis, ecp, (xc_disp, disp), xc_gcp = self.parameters_3c
            grad_driver.get_dispersion = MethodType(gen_disp_grad_fun(xc_disp, xc_gcp), grad_driver)
        with_df = getattr(mf, 'with_df', None)
        if with_df:
            grad_driver.auxbasis_response = 1
        if self.derivative_grid_response:
            grad_driver.grid_response = True
        gradient = grad_driver.kernel()
        self.log("Completed gradient step")

        try:
            e_tot = mf._scf.e_tot
        except AttributeError:
            e_tot = mf.e_tot

        results = {
            "energy": e_tot,
            "forces": -gradient.flatten(),
        }
        results = self.store_and_track(
            results, self.get_forces, atoms, coords, **prepare_kwargs
        )
        return results

    def get_hessian(self, atoms, coords, **prepare_kwargs):
        point_charges = prepare_kwargs.get("point_charges", None)

        mol = self.prepare_input(atoms, coords)
        mf = self.run(mol, point_charges=point_charges)
        hessian_driver = mf.Hessian()
        if self.parameters_3c is not None:
            pyscf_xc, nlc, basis, ecp, (xc_disp, disp), xc_gcp = self.parameters_3c
            hessian_driver.get_dispersion = MethodType(gen_disp_hess_fun(xc_disp, xc_gcp), hessian_driver)
        with_df = getattr(mf, 'with_df', None)
        if with_df:
            hessian_driver.auxbasis_response = 2
        if self.derivative_grid_response:
            hessian_driver.grid_response = True
        H = hessian_driver.kernel()

        # The returned hessian is 4d ... ok. This probably serves a purpose
        # that I don't understand. We transform H to a nice, simple 2d array.
        H = np.hstack(np.concatenate(H, axis=1))
        results = {
            "energy": mf.e_tot,
            "hessian": H,
        }
        # results = self.store_and_track(
        # results, self.get_hessian, atoms, coords, **prepare_kwargs
        # )
        return results

    def run_calculation(self, atoms, coords, **prepare_kwargs):
        return self.get_energy(atoms, coords, **prepare_kwargs)

    def run(self, mol, point_charges=None):
        steps = self.multisteps[self.method]
        self.log(f"Running steps '{steps}' for method {self.method}")
        for i, step in enumerate(steps):
            if i == 0:
                mf = self.get_driver(step, mol=mol)
                assert step in ("scf", "dft")
                if self.chkfile:
                    # Copy old chkfile to new chkfile
                    new_chkfile = self.make_fn("chkfile", return_str=True)
                    shutil.copy(self.chkfile, new_chkfile)
                    self.chkfile = new_chkfile
                    mf.chkfile = self.chkfile
                    mf.init_guess = "chkfile"
                    self.log(
                        f"Using '{self.chkfile}' as initial guess for {step} calculation."
                    )
                if self.auxbasis:
                    mf = mf.density_fit(auxbasis=self.auxbasis)
                    self.log(f"Using density fitting with auxbasis {self.auxbasis}.")

                if self.parameters_3c is not None:
                    # Caution: make sure the dispersion is set after to_gpu() function
                    pyscf_xc, nlc, basis, ecp, (xc_disp, disp), xc_gcp = self.parameters_3c
                    mf.xc = pyscf_xc
                    mf.nlc = nlc
                    mf.get_dispersion = MethodType(gen_disp_fun(xc_disp, xc_gcp), mf)
                    mf.do_disp = lambda: True

                if point_charges is not None:
                    mf = qmmm.mm_charge(mf, point_charges[:, :3], point_charges[:, 3])
                    self.log(
                        f"Added {len(point_charges)} point charges with "
                        f"sum(q)={sum(point_charges[:,3]):.4f}."
                    )
            else:
                mf = self.get_driver(step, mf=prev_mf)  # noqa: F821

            if self.keep_chk and (self.chkfile is None) and (step in ("dft", "scf")):
                self.chkfile = self.make_fn("chkfile", return_str=True)
                try:
                    os.remove(self.chkfile)
                except FileNotFoundError:
                    self.log(f"Tried to remove '{self.chkfile}'. It doesn't exist.")
                self.log(f"Created chkfile '{self.chkfile}'")
                mf.chkfile = self.chkfile
            mf.kernel()
            self.log(f"Completed {step} step")
            prev_mf = mf

        # Keep mf and dump mol
        # save_mol(mol, self.make_fn("mol.chk"))
        self.mf = mf.reset()  # release integrals and other temporary intermediates.
        if self.use_gpu:
            # DF methods are eager to use more memory. Recycle as much memory as
            # possible for the DF tensor.
            import cupy
            cupy.get_default_memory_pool().free_all_blocks()
        self.calc_counter += 1

        return mf

    def parse_all_energies(self, exc_mf=None):
        if exc_mf is None:
            exc_mf = self.mf

        try:
            gs_energy = exc_mf._scf.e_tot
            exc_energies = exc_mf.e_tot
            all_energies = np.zeros(exc_energies.size + 1)
            all_energies[0] = gs_energy
            all_energies[1:] = exc_energies
        except AttributeError:
            gs_energy = exc_mf.e_tot
            all_energies = np.array((gs_energy,))

        return all_energies

    def prepare_overlap_data(self, path):
        gs_mf = self.mf._scf
        exc_mf = self.mf

        C = gs_mf.mo_coeff

        first_Y = exc_mf.xy[0][1]
        # In TDA calculations Y is just the integer 0.
        if isinstance(first_Y, int) and (first_Y == 0):
            X = np.array([state[0] for state in exc_mf.xy])
            Y = np.zeros_like(X)
        # In TD-DFT calculations the Y vectors is also present
        else:
            # Shape = (nstates, 2 (X,Y), occ, virt)
            ci_coeffs = np.array(exc_mf.xy)
            X = ci_coeffs[:, 0]
            Y = ci_coeffs[:, 1]

        all_energies = self.parse_all_energies(exc_mf)
        return C, X, Y, all_energies

    def parse_charges(self):
        results = self.mf.analyze(with_meta_lowdin=False)
        # Mulliken charges
        charges = results[0][1]
        return charges

    def get_chkfiles(self):
        return {
            "chkfile": self.chkfile,
        }

    def set_chkfiles(self, chkfiles):
        try:
            chkfile = chkfiles["chkfile"]
            self.chkfile = chkfile
            self.log(f"Set chkfile '{chkfile}' as chkfile.")
        except KeyError:
            self.log("Found no chkfile information in chkfiles!")

    def __str__(self):
        return f"PySCF({self.name})"
