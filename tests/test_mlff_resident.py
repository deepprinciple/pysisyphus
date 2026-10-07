"""CPU contract tests; no trained model or GPU required."""
import importlib
import os
import sys
import types
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from pysisyphus.calculators.mlff_resident import ResidentMLFFModel
from pysisyphus.constants import AU2EV, BOHR2ANG


class Registry:
    def __init__(self):
        self._calculators = {}

    def register(self, name, factory):
        self._calculators[name] = factory

    def get(self, name, **kwargs):
        return self._calculators[name](**kwargs)


class HarmonicModel:
    """YAMO-shaped eV/Angstrom potential with persistent native result dicts."""
    def __init__(self):
        self.initializations = 0
        self.resets = 0
        self.last_results = None

    def initialize(self):
        self.initializations += 1
        return True

    def reset(self):
        self.resets += 1
        self.last_results = None

    def get_energy(self, atoms, return_dict=True):
        self.last_results = {'energy': float((atoms.positions**2).sum()/2)}
        return self.last_results

    def get_forces(self, atoms, return_dict=True):
        self.last_results = {'energy': float((atoms.positions**2).sum()/2),
                             'forces': -atoms.positions.reshape(-1).copy()}
        return self.last_results

    def get_hessian(self, atoms, return_dict=True):
        self.last_results = {'hessian': np.eye(atoms.positions.size)}
        return self.last_results


@pytest.fixture
def mlff(monkeypatch):
    model = HarmonicModel()
    # MLFF's optional YAMO dependency is replaced by its contract fake; these
    # CPU unit/optimizer tests also run in CI without trained models or YAMO.
    yamo = types.ModuleType('yamo')
    calculators = types.ModuleType('yamo.calculators')
    calculators.get_calculator = lambda *a, **k: model
    monkeypatch.setitem(sys.modules, 'yamo', yamo)
    monkeypatch.setitem(sys.modules, 'yamo.calculators', calculators)
    module = importlib.import_module('pysisyphus.calculators.MLFF')
    monkeypatch.setattr(module, 'get_calculator', lambda *a, **k: model)
    return module, model


def test_residency_restores_registry_and_resets_properties():
    registry = Registry()
    original = lambda **kw: HarmonicModel()
    registry.register('eqv3', original)
    cache = ResidentMLFFModel()
    models = []
    for _ in range(2):
        with cache.registration({'checkpoint_sha256': 'a', 'charge': 0}, registry=registry):
            registry.register('eqv3', lambda **kw: HarmonicModel())
            models.append(registry.get('eqv3', device='cpu'))
        assert registry._calculators['eqv3'] is original
    assert models[0] is models[1]
    assert models[0].initializations == 1
    assert models[0].resets == 2
    assert cache.metrics()['model_load_count'] == 1
    assert cache.metrics()['model_cache_hits'] == 1


@pytest.mark.parametrize('change', ['charge', 'checkpoint_sha256', 'use_ema',
                                    'force_mode', 'device', 'method', 'mask'])
def test_incompatible_scope_is_not_reused(change, monkeypatch):
    registry = Registry()
    cache = ResidentMLFFModel()
    scope = dict(checkpoint_sha256='a', charge=0, use_ema=True, force_mode='native')
    with cache.registration(scope, registry=registry):
        registry.register('eqv3', lambda **kw: HarmonicModel())
        first = registry.get('eqv3', device='cpu')
    method, device = 'eqv3', 'cpu'
    if change == 'device':
        device = 'cuda'
    elif change == 'method':
        method = 'other'
    elif change == 'mask':
        monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '7')
    else:
        scope[change] = {'charge': 1, 'checkpoint_sha256': 'b',
                         'use_ema': False, 'force_mode': 'autograd'}[change]
    with cache.registration(scope, method=method, registry=registry):
        registry.register(method, lambda **kw: HarmonicModel())
        second = registry.get(method, device=device)
    assert first is not second
    assert cache.loads == 2


def test_failure_evicts_and_restores_registry():
    registry = Registry()
    cache = ResidentMLFFModel()
    with pytest.raises(ValueError):
        with cache.registration({'model': 'a'}, registry=registry):
            registry.register('eqv3', lambda **kw: HarmonicModel())
            registry.get('eqv3')
            raise ValueError('optimization failed')
    assert cache.calculator is None and cache.key is None
    assert not cache._active and not registry._calculators
    with cache.registration({'model': 'a'}, registry=registry):
        registry.register('eqv3', lambda **kw: HarmonicModel())
        registry.get('eqv3')
    assert cache.loads == 2


def test_failed_initialization_is_not_cached():
    registry = Registry()
    cache = ResidentMLFFModel()
    model = HarmonicModel()
    model.initialize = lambda: False
    with pytest.raises(RuntimeError, match='initialization failed'):
        with cache.registration({'model': 'a'}, registry=registry):
            registry.register('eqv3', lambda **kw: model)
            registry.get('eqv3')
    assert cache.calculator is None and cache.loads == 0
    assert not registry._calculators


def test_thread_and_process_ownership_and_nesting(monkeypatch):
    cache = ResidentMLFFModel()
    with ThreadPoolExecutor(max_workers=1) as pool:
        with pytest.raises(RuntimeError, match='another process/thread'):
            pool.submit(cache._check_owner).result()
    monkeypatch.setattr(cache, 'pid', os.getpid()+1)
    with pytest.raises(RuntimeError, match='another process/thread'):
        cache._check_owner()
    monkeypatch.setattr(cache, 'pid', os.getpid())
    with cache.registration({'model': 'a'}, registry=Registry()):
        with pytest.raises(RuntimeError, match='nested'):
            with cache.registration({'model': 'a'}, registry=Registry()):
                pass
    assert not cache._active


def test_energy_force_hessian_units_and_native_results(mlff, tmp_path):
    module, model = mlff
    calc = module.MLFF(out_dir=tmp_path)
    coords = np.array([0.8, -0.4, 0.2])
    force = calc.get_forces(('H',), coords)['forces'].reshape(-1)
    expected_hessian = np.eye(3)*BOHR2ANG**2/AU2EV
    hessian = calc.get_hessian(('H',), coords)['hessian']
    assert np.allclose(hessian, expected_hessian)
    assert np.allclose(model.last_results['hessian'], np.eye(3))
    delta = 1e-4
    fd_force, fd_hessian = [], []
    for i in range(3):
        offset = np.eye(3)[i]*delta
        ep = calc.get_energy(('H',), coords+offset)['energy']
        em = calc.get_energy(('H',), coords-offset)['energy']
        fd_force.append(-(ep-em)/(2*delta))
        fp = calc.get_forces(('H',), coords+offset)['forces'].reshape(-1)
        fm = calc.get_forces(('H',), coords-offset)['forces'].reshape(-1)
        fd_hessian.append(-(fp-fm)/(2*delta))
    assert np.allclose(force, fd_force, atol=1e-6)
    assert np.allclose(hessian, np.array(fd_hessian).T, atol=1e-6)
    calc.get_energy(('H',), coords)
    assert model.last_results['energy'] == pytest.approx(float((coords*BOHR2ANG)**2 @ np.ones(3)/2), abs=1e-8)
    assert not list(tmp_path.glob('mlff_*.xyz'))


def test_rfo_cold_and_resident_keep_identical_optimization(mlff, monkeypatch, tmp_path):
    from pysisyphus.Geometry import Geometry
    from pysisyphus.optimizers.RFOptimizer import RFOptimizer
    module, _ = mlff
    registry = Registry()
    monkeypatch.setattr(module, 'get_calculator', registry.get)
    cache = ResidentMLFFModel()
    results = []
    for policy in ('cold', 'resident', 'resident'):
        active = cache if policy == 'resident' else ResidentMLFFModel()
        with active.registration({'model': 'harmonic'}, registry=registry):
            registry.register('eqv3', lambda **kw: HarmonicModel())
            geom = Geometry(('H',), np.array([0.8, -0.4, 0.2]))
            calc = module.MLFF(method='eqv3', out_dir=tmp_path)
            geom.set_calculator(calc)
            opt = RFOptimizer(geom, thresh='gau', max_cycles=30,
                              hessian_init='unit', hessian_update='bfgs',
                              hessian_recalc=0, dump=False)
            opt.run()
            results.append((opt.is_converged, opt.cur_cycle, geom.energy, geom.coords.copy()))
    assert all(x[0] for x in results)
    assert len({x[1] for x in results}) == 1
    for result in results[1:]:
        assert result[2] == pytest.approx(results[0][2], abs=1e-12)
        assert np.allclose(result[3], results[0][3], atol=1e-12)
    assert cache.loads == 1 and cache.hits == 1
