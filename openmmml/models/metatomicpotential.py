"""
metatomicpotential.py: Implements support for metatomic atomistic models.

This is part of the OpenMM molecular simulation toolkit originating from
Simbios, the NIH National Center for Physics-Based Simulation of
Biological Structures at Stanford, funded under the NIH Roadmap for
Medical Research, grant U54 GM072970. See https://simtk.org.

Portions copyright (c) 2026 Stanford University and the Authors.
Authors: Filippo Bigi
Contributors:

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

import os
from typing import Iterable, Optional

import numpy as np
import openmm
from openmm import unit
from functools import partial

from openmmml.mlpotential import MLPotentialImpl, MLPotentialImplFactory


# Named pre-trained models: maps entry-point name to (hf_repo, file_in_repo).
# These are downloaded as metatrain checkpoints and exported on first use.
KNOWN_MODELS = {
    # PET-MAD v1: trained on MAD 1 (PBEsol, materials and molecules)
    'pet-mad-1-s': ('lab-cosmo/upet', 'models/pet-mad-s-v1.1.0.ckpt'),
    # PET-MAD v1.5: trained on MAD 1.5 (r2SCAN and improved coverage)
    'pet-mad-1.5-xs': ('lab-cosmo/upet', 'models/pet-mad-xs-v1.5.0.ckpt'),
    'pet-mad-1.5-s': ('lab-cosmo/upet', 'models/pet-mad-s-v1.5.0.ckpt'),
    # PET-SPICE: trained on the SPICE dataset (hybrid DFT, organic molecules)
    'pet-spice-s': ('lab-cosmo/upet', 'models/pet-spice-s-v0.2.0.ckpt'),
    'pet-spice-l': ('lab-cosmo/upet', 'models/pet-spice-l-v0.2.0.ckpt'),
}

# Module-level cache so each (model_path, device) is loaded only once.
# The compute closure does NOT hold a direct reference to the TorchScript
# AtomisticModel (which is not picklable), so that OpenMM can deepcopy
# the PythonForce when building interpolated mixed systems.
_model_cache = {}


class MetatomicPotentialImplFactory(MLPotentialImplFactory):
    """This is the factory that creates MetatomicPotentialImpl objects."""

    def createImpl(self, name: str, modelPath: Optional[str] = None, **args) -> MLPotentialImpl:
        return MetatomicPotentialImpl(name, modelPath)


class MetatomicPotentialImpl(MLPotentialImpl):
    """MLPotentialImpl providing support for metatomic atomistic models.

    This implementation supports two modes of use:

    **Pre-trained models**: named models are downloaded and exported automatically on
    first use. The exported model is cached in ``~/.cache/openmm-ml`` for subsequent use.

    Available named models:

    - ``'pet-mad-1-s'``: PET-MAD v1: PET trained on the MAD-1 dataset.
      Accurate on materials and molecules, PBEsol functional.
    - ``'pet-mad-1.5-xs'`` / ``'pet-mad-1.5-s'``: PET-MAD v1.5: PET trained
        on the MAD-1.5 dataset with improved coverage, r2SCAN functional.
    - ``'pet-spice-s'`` / ``'pet-spice-l'``: PET trained on the SPICE dataset,
      optimized for organic molecules and drug-like compounds, hybrid DFT functional.

    >>> potential = MLPotential('pet-mad-1.5-xs')
    >>> system = potential.createSystem(topology)

    **Custom models**: any model exported with ``mtt export`` (or
    ``AtomisticModel.save()``) can be loaded by passing the path:

    >>> potential = MLPotential('metatomic', modelPath='my_model.pt')
    >>> system = potential.createSystem(topology)

    Neighbor lists are computed with vesin (https://github.com/Luthaf/vesin).

    If the model requires extensions (compiled TorchScript ops), pass the directory
    via ``extensionsDirectory``:

    >>> system = potential.createSystem(topology, extensionsDirectory='/path/to/extensions')

    Supports mixed ML/MM systems.
    """

    def __init__(self, name: str, modelPath: Optional[str]) -> None:
        self.name = name
        self.modelPath = modelPath

    def _resolveModelPath(self) -> tuple:
        """Return (pt_path, extensions_dir) for the model, downloading and exporting if needed."""
        if self.name in KNOWN_MODELS:
            hf_repo, ckpt_file = KNOWN_MODELS[self.name]
            cache_dir = os.path.join(self._getCacheDir(), self.name)
            pt_path = os.path.join(cache_dir, 'model.pt')
            ext_dir = os.path.join(cache_dir, 'extensions')
            if not os.path.isfile(pt_path):
                try:
                    from metatrain.cli.export import export_model
                except ImportError as e:
                    raise ImportError(
                        f"metatrain is required to download and export pre-trained models "
                        f"like '{self.name}'.  Install it with 'pip install openmmml[metatomic]'."
                    ) from e
                os.makedirs(cache_dir, exist_ok=True)
                print(
                    f"Downloading and exporting '{self.name}' from "
                    f"'{hf_repo}/{ckpt_file}' — this may take a moment on first use…"
                )
                export_model(
                    path=hf_repo,
                    output=pt_path,
                    path_in_repo=ckpt_file,
                    extensions=ext_dir,
                )
            extensions_directory = ext_dir if os.path.isdir(ext_dir) else None
            return pt_path, extensions_directory

        elif self.name == 'metatomic':
            if self.modelPath is None:
                raise ValueError(
                    "modelPath must be provided when using the generic 'metatomic' potential. "
                    "Use MLPotential('metatomic', modelPath='model.pt')."
                )
            return self.modelPath, None

        else:
            raise ValueError(
                f"Unknown metatomic model '{self.name}'. "
                f"Known named models: {list(KNOWN_MODELS.keys())}. "
                "For a custom model use MLPotential('metatomic', modelPath='model.pt')."
            )

    def addForces(
        self,
        topology: openmm.app.Topology,
        system: openmm.System,
        atoms: Optional[Iterable[int]],
        forceGroup: int,
        extensionsDirectory: Optional[str] = None,
        **args,
    ):
        try:
            import torch
            from metatomic.torch import load_atomistic_model
        except ImportError as e:
            raise ImportError(
                f"Failed to import metatomic-torch: {e}. "
                "Install it with 'pip install openmmml[metatomic]'."
            )

        pt_path, auto_ext_dir = self._resolveModelPath()
        # Explicit argument overrides the auto-detected extensions directory.
        if extensionsDirectory is None:
            extensionsDirectory = auto_ext_dir

        device = self._getTorchDevice(args)

        # Load the model once and stash it in the module-level cache keyed by
        # (path, device).  The compute closure only stores this key (a picklable
        # tuple) so that OpenMM can deepcopy PythonForce for interpolated systems.
        cache_key = (pt_path, str(device))
        if cache_key not in _model_cache:
            model = load_atomistic_model(pt_path, extensionsDirectory)
            model = model.to(device=device)
            _model_cache[cache_key] = model
        model = _model_cache[cache_key]

        import torch
        includedAtoms = list(topology.atoms())
        if atoms is not None:
            includedAtoms = [includedAtoms[i] for i in atoms]
            indices = np.array(list(atoms))
        else:
            indices = None

        atomicTypes = torch.tensor(
            [atom.element.atomic_number for atom in includedAtoms],
            dtype=torch.int32,
            device=device,
        )

        periodic = (
            topology.getPeriodicBoxVectors() is not None
            or system.usesPeriodicBoundaryConditions()
        )

        compute = partial(
            _computeMetatomic,
            modelCacheKey=cache_key,
            atomicTypes=atomicTypes,
            indices=indices,
            periodic=periodic,
        )
        force = openmm.PythonForce(compute)
        force.setForceGroup(forceGroup)
        force.setUsesPeriodicBoundaryConditions(periodic)
        system.addForce(force)


def _computeMetatomic(
    state, modelCacheKey, atomicTypes, indices, periodic
):
    import torch
    import vesin.metatomic
    from metatomic.torch import System, ModelEvaluationOptions, ModelOutput

    model = _model_cache[modelCacheKey]
    device = atomicTypes.device
    dtype = torch.float32 if model.capabilities().dtype == "float32" else torch.float64

    # Positions are kept in OpenMM's native unit (nm).  We tell metatomic the
    # length unit is "nm" via ModelEvaluationOptions.length_unit, so it handles
    # the conversion to the model's native length unit internally.
    positions_nm = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    numAtoms = positions_nm.shape[0]

    if indices is not None:
        positions_nm = positions_nm[indices]

    positions = torch.tensor(positions_nm, dtype=dtype, device=device)
    positions.requires_grad_(True)

    if periodic:
        cell_nm = state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.nanometer)
        cell = torch.tensor(cell_nm, dtype=dtype, device=device)
        pbc = torch.tensor([True, True, True], dtype=torch.bool, device=device)
    else:
        cell = torch.zeros((3, 3), dtype=dtype, device=device)
        pbc = torch.tensor([False, False, False], dtype=torch.bool, device=device)

    system = System(types=atomicTypes, positions=positions, cell=cell, pbc=pbc)

    # Compute all neighbor lists requested by the model, one at a time, using
    # fresh NeighborList objects so there is no state carried between steps.
    for nl_options in model.requested_neighbor_lists():
        calculator = vesin.metatomic.NeighborList(
            nl_options, length_unit="nm", check_consistency=False
        )
        neighbors = calculator.compute(system)
        system.add_neighbor_list(nl_options, neighbors)

    # Request energy in kJ/mol.  AtomisticModel converts from the model's native
    # energy unit to kJ/mol internally.  Length unit "nm" tells it the input
    # positions are in nm.
    options = ModelEvaluationOptions(
        length_unit="nm",
        outputs={"energy": ModelOutput(quantity="energy", unit="kJ/mol", per_atom=False)},
    )

    outputs = model([system], options, check_consistency=False)

    # Energy is in kJ/mol; its gradient w.r.t. positions (in nm) gives forces
    # in kJ/mol/nm directly — no manual unit conversion needed.
    energy = outputs["energy"].block(0).values.squeeze()
    energy.backward()

    forces = -positions.grad.detach().cpu().numpy()
    energy_val = energy.item()

    if indices is not None:
        f = np.zeros((numAtoms, 3), dtype=np.float64)
        f[indices] = forces
        forces = f

    return energy_val, forces
