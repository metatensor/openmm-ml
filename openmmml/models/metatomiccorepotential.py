"""
metatomiccorepotential.py: Implements OpenMM-ML potentials from metatomic-core models.

This is part of the OpenMM molecular simulation toolkit originating from
Simbios, the NIH National Center for Physics-Based Simulation of
Biological Structures at Stanford, funded under the NIH Roadmap for
Medical Research, grant U54 GM072970. See https://simtk.org.

Portions copyright (c) 2026 Stanford University and the Authors.
Authors: Eric D. Boittier

Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
THE AUTHORS, CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import math
import os
import warnings
from collections.abc import Iterable

import numpy as np
import openmm

from openmmml.mlpotential import MLPotentialImpl, MLPotentialImplFactory

from .metatomicpotential import _resolve_atom_types

# Data origins of arrays returned by native and TorchScript plugins. Older
# metatensor releases need an explicit wrapper to read them from Python.
_EXTERNAL_ORIGINS = ("metatensor::SimpleDataArray", "metatensor_torch::TorchDataArray")
_VALID_NC = (True, False, "forces")
_INPUT_DEFAULTS = {"charge": (0.0, "e"), "spin_multiplicity": (1.0, "")}


class MetatomicCorePotentialImplFactory(MLPotentialImplFactory):
    """This is the factory that creates MetatomicCorePotentialImpl objects."""

    def createImpl(self, name, model, **args) -> MLPotentialImpl:
        return MetatomicCorePotentialImpl(model, **args)


class MetatomicCorePotentialImpl(MLPotentialImpl):
    """This MLPotentialImpl evaluates any model behind the metatomic-core C API.

    ``model`` is either an in-process :class:`metatomic.BaseModel` (for example
    a ``JaxModel`` or ``NumpyModel``), or something :func:`metatomic.load_model`
    understands: a TorchScript ``.pt`` file once the metatomic-torch plugin is
    loaded (``LegacyModel``), or a model name exposed by a native
    ``mta_plugin_t`` shared library.

    >>> potential = MLPotential(
    ...     "metatomic-core",
    ...     model="metatomic-lj-model",
    ...     plugins=["metatomic-lj-plugin.so"],
    ...     modelOptions={"sigma": "3.4", ...},
    ... )

    Every evaluation goes through :func:`metatomic.execute_model` with the
    system in nm and the energy requested in kJ/mol, so metatomic-core does all
    unit conversions, including the positions gradient (kJ/mol/nm). Pair lists
    are computed with vesin behind a Verlet skin. The force is an
    :class:`openmm.PythonForce`: energy and forces only (no virial), evaluated
    on the host with one Python call per step.

    Options: ``plugins`` (shared libraries passed to
    :func:`metatomic.load_plugin` first), ``pluginName`` and ``modelOptions``
    (passed to :func:`metatomic.load_model`), ``checkConsistency``,
    ``nonConservative`` (``False``, ``True`` or ``"forces"``), ``variants``
    (``{"energy": "<variant>"}``), and ``neighborSkin`` (nm, default 0.2).
    ``createSystem()`` takes ``atomTypes``, ``charge`` and ``multiplicity``, as
    with ``MLPotential("metatomic")``.

    A subset of atoms (mechanical embedding) is evaluated on its own, as an
    isolated non-periodic molecule.
    """

    def __init__(
        self,
        model,
        plugins=(),
        pluginName=None,
        modelOptions=None,
        checkConsistency=False,
        nonConservative=False,
        variants=None,
        neighborSkin=0.2,
        device=None,
    ):
        if nonConservative not in _VALID_NC:
            raise ValueError(
                f"nonConservative must be one of {list(_VALID_NC)}, got {nonConservative!r}"
            )
        if device not in (None, "cpu"):
            raise ValueError("MLPotential('metatomic-core') only evaluates on the CPU so far")
        self.model = model
        self.plugins = [plugins] if isinstance(plugins, (str, os.PathLike)) else list(plugins)
        self.pluginName = pluginName
        self.modelOptions = modelOptions
        self.checkConsistency = checkConsistency
        self.nonConservative = nonConservative
        self.variants = dict(variants or {})
        self.neighborSkin = neighborSkin

    def getMLLongRange(self) -> bool:
        return False

    def _load(self):
        try:
            import metatensor
            import metatomic
        except ImportError as e:
            raise ImportError(
                "Failed to import metatomic. Install it with 'pip install metatomic'."
            ) from e
        for origin in _EXTERNAL_ORIGINS:
            try:
                metatensor.register_external_data_wrapper(origin, metatensor.ExternalCpuArray)
            except Exception:
                pass  # already registered, or not needed on this metatensor
        for plugin in self.plugins:
            metatomic.load_plugin(os.fspath(plugin))
        if isinstance(self.model, metatomic.BaseModel):
            return self.model
        options = {k: str(v) for k, v in (self.modelOptions or {}).items()}
        return metatomic.load_model(
            os.fspath(self.model), options=options or None, plugin_name=self.pluginName
        )

    def _pick(self, capabilities, name, sample_kind):
        variant = self.variants.get(name, self.variants.get("energy"))
        target = name if variant is None else f"{name}/{variant}"
        candidates = [
            q
            for q in capabilities.outputs
            if q.sample_kind == sample_kind and q.name.split("/")[0] == name
        ]
        exact = [q for q in candidates if q.name == target]
        if exact:
            return exact[0]
        if variant is None and len(candidates) == 1:
            return candidates[0]
        raise ValueError(
            f"this model does not unambiguously provide {target!r} with "
            f"{sample_kind!r} samples"
        )

    def addForces(
        self,
        topology: openmm.app.Topology,
        system: openmm.System,
        atoms: Iterable[int] | None,
        forceGroup: int,
        **args,
    ):
        from metatomic import Quantity

        if "mlLongRange" in args:
            raise ValueError("This ML model does not support the mlLongRange option.")
        model = self._load()
        capabilities = model.capabilities()
        types = np.array(
            _resolve_atom_types(list(topology.atoms()), args.get("atomTypes")), dtype=np.int32
        )
        if atoms is not None and math.isinf(capabilities.interaction_range):
            raise ValueError(
                "this model has an infinite interaction range, which is not "
                "supported for mixed ML/MM systems"
            )
        unsupported = set(types.tolist()) - set(capabilities.atomic_types)
        if unsupported:
            raise ValueError(f"this model does not support atomic types {sorted(unsupported)}")

        nc_forces = self.nonConservative in (True, "forces")
        energy = self._pick(capabilities, "energy", "system")
        if not nc_forces and "positions" not in energy.gradients:
            raise ValueError("this model does not provide positions gradients of the energy")
        requests = [
            Quantity(
                name=energy.name,
                unit="kJ/mol",
                sample_kind="system",
                gradients=[] if nc_forces else ["positions"],
            )
        ]
        if nc_forces:
            forces = self._pick(capabilities, "non_conservative_force", "atom")
            requests.append(Quantity(name=forces.name, unit="kJ/mol/nm", sample_kind="atom"))

        periodic = (
            topology.getPeriodicBoxVectors() is not None
            or system.usesPeriodicBoundaryConditions()
        )
        pbc = np.array(args.get("pbc", [periodic] * 3), dtype=bool)
        if pbc.shape != (3,):
            raise ValueError("pbc must be a length-3 sequence of booleans")
        region = None
        if atoms is not None:
            region = np.array(list(atoms), dtype=np.int64)
            pbc = np.zeros(3, dtype=bool)

        compute = _ComputeMetatomicCore(
            model, types, pbc, region, requests, args, self.checkConsistency, self.neighborSkin
        )
        force = openmm.PythonForce(compute)
        force.setForceGroup(forceGroup)
        force.setUsesPeriodicBoundaryConditions(bool(pbc.any()))
        system.addForce(force)


def _system_input(name, args, dtype):
    from metatensor import Labels, TensorBlock, TensorMap

    if name not in _INPUT_DEFAULTS:
        raise ValueError(
            f"this model requests extra input {name!r}, which is not implemented "
            "by MLPotential('metatomic-core')"
        )
    value, unit = _INPUT_DEFAULTS[name]
    aliases = ("multiplicity", "spinMultiplicity", "spin_multiplicity") if name != "charge" else (name,)
    for alias in aliases:
        value = args.get(alias, value)
    block = TensorBlock(
        np.array([[float(value)]], dtype=dtype),
        Labels("system", np.zeros((1, 1), dtype=np.int32)),
        [],
        Labels.range(name, 1),
    )
    tensor = TensorMap(Labels.range("_", 1), [block])
    tensor.set_info("unit", unit)
    return tensor


class _ComputeMetatomicCore:
    def __init__(self, model, types, pbc, region, requests, args, check_consistency, skin):
        import metatomic

        self.model, self.pbc, self.region = model, pbc, region
        self.requests, self.check_consistency = requests, check_consistency
        self.n_atoms = len(types)
        self.types = types if region is None else types[region]
        capabilities = model.capabilities()
        self.dtype = np.dtype(capabilities.dtype)
        to_nm = metatomic.unit_conversion_factor(capabilities.length_unit, "nm")
        self.pair_lists = []
        for options in model.requested_pair_lists(requests):
            import vesin

            neighbors = vesin.NeighborList(
                cutoff=options.cutoff * to_nm, full_list=options.full_list, skin=skin
            )
            self.pair_lists.append((options, neighbors))
        self.inputs = [
            (q.name, _system_input(q.name, args, self.dtype))
            for q in model.requested_inputs(requests)
        ]

    def _system(self, positions, cell):
        import metatomic
        from metatensor import Labels, TensorBlock

        system = metatomic.System(
            "nm", self.types, positions.astype(self.dtype), cell.astype(self.dtype), self.pbc
        )
        for options, neighbors in self.pair_lists:
            i, j, shifts, vectors = neighbors.compute(
                points=positions, box=cell, periodic=self.pbc, quantities="ijSD", copy=True
            )
            pairs = TensorBlock(
                vectors.astype(self.dtype).reshape(-1, 3, 1),
                Labels(
                    ["first_atom", "second_atom", "cell_shift_a", "cell_shift_b", "cell_shift_c"],
                    np.column_stack([i, j, shifts]).astype(np.int32).reshape(-1, 5),
                ),
                [Labels.range("xyz", 3)],
                Labels.range("distance", 1),
            )
            system.add_pairs(options, pairs)
        for name, tensor in self.inputs:
            system.add_custom_data(name, tensor)
        return system

    def __call__(self, state):
        import metatomic

        positions = np.asarray(state.getPositions(asNumpy=True), dtype=np.float64)
        if self.region is not None:
            positions = positions[self.region]
        if self.pbc.any():
            cell = np.asarray(state.getPeriodicBoxVectors(asNumpy=True), dtype=np.float64)
            cell = cell * self.pbc[:, None]
        else:
            cell = np.zeros((3, 3))

        outputs = metatomic.execute_model(
            self.model,
            [self._system(np.ascontiguousarray(positions), cell)],
            self.requests,
            check_consistency=self.check_consistency,
        )
        block = outputs[0].block().to(device="cpu").to(arrays="numpy")
        energy = float(np.asarray(block.values).sum())
        if len(outputs) > 1:
            nc = outputs[1].block().to(device="cpu").to(arrays="numpy")
            values = np.asarray(nc.values).reshape(-1, 3)
            atoms = np.asarray(nc.samples.column("atom"))
            # non-conservative forces are not guaranteed to sum to zero
            values = values - values.mean(axis=0, keepdims=True)
        else:
            gradient = block.gradient("positions")
            values = -np.asarray(gradient.values).reshape(-1, 3)
            atoms = np.asarray(gradient.samples.column("atom"))
        local = np.zeros((len(self.types), 3))
        np.add.at(local, atoms, values)
        if self.region is None:
            return energy, local
        forces = np.zeros((self.n_atoms, 3))
        forces[self.region] = local
        return energy, forces
