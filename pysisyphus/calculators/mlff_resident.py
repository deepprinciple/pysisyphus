"""Serial, process-local YAMO model reuse across PYSIS optimizations.

Intercept registration of a selected native YAMO factory (EqV3 by default). Each PYSIS run still
creates its own MLFF wrapper and optimizer; model weights alone persist.
Use inside a pinned worker, around model registration and PYSIS execution.
"""
from contextlib import contextmanager
import json
import os
import threading
import time


class ResidentMLFFModel:
    def __init__(self):
        self.pid = os.getpid()
        self.owner_thread = threading.get_ident()
        self.calculator = None
        self.key = None
        self.loads = 0
        self.hits = 0
        self.load_seconds = 0.0
        self._active = False

    def _check_owner(self):
        if self.pid != os.getpid() or self.owner_thread != threading.get_ident():
            raise RuntimeError('Resident model belongs to another process/thread')

    @contextmanager
    def registration(self, scope, *, method="eqv3", registry=None):
        """Scope includes checkpoint SHA, settings/EMA, q/m and physical GPU.

        Only one resident model is retained, bounding memory. Incompatible
        scope replaces it; a failed optimization invalidates it. No histories
        or ASE energy/force caches are reused between endpoints.
        """
        self._check_owner()
        if self._active:
            raise RuntimeError('Concurrent/nested PYSIS registration is unsupported')
        if registry is None:
            from yamo.calculators.base import _registry
            registry = _registry
        original_register = registry.register
        prior = registry._calculators.get(method)
        scope_key = json.dumps(scope, sort_keys=True, default=str)
        self._active = True

        def register(name, factory):
            self._check_owner()
            if name != method:
                return original_register(name, factory)

            def cached_factory(**kwargs):
                self._check_owner()
                key = (method, scope_key, os.environ.get("CUDA_VISIBLE_DEVICES"),
                       json.dumps(kwargs, sort_keys=True, default=str))
                if key != self.key:
                    self.calculator = None
                    self.key = None
                if self.calculator is None:
                    start = time.perf_counter()
                    calc = factory(**kwargs)
                    if not calc.initialize():
                        raise RuntimeError('Native YAMO model initialization failed')
                    self.calculator, self.key = calc, key
                    self.loads += 1
                    self.load_seconds += time.perf_counter() - start
                else:
                    self.hits += 1
                self.calculator.reset()
                return self.calculator

            return original_register(name, cached_factory)

        registry.register = register
        try:
            yield self
        except BaseException:
            self.calculator = None
            self.key = None
            raise
        finally:
            registry.register = original_register
            if prior is None:
                registry._calculators.pop(method, None)
            else:
                registry._calculators[method] = prior
            self._active = False

    def metrics(self):
        return dict(pid=self.pid, model_load_count=self.loads,
                    model_cache_hits=self.hits, model_load_seconds=self.load_seconds)
