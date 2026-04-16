"""Tests for the metatomic potential implementation."""

import os
from typing import Dict, List, Optional

import numpy as np
import openmm as mm
import openmm.app as app
import openmm.unit as unit
import pytest
import torch
from metatensor.torch import Labels, TensorBlock, TensorMap

metatomic = pytest.importorskip("metatomic.torch", reason="metatomic-torch is not installed")

from metatomic.torch import (
    AtomisticModel,
    ModelCapabilities,
    ModelMetadata,
    ModelOutput,
    NeighborListOptions,
    System,
)

from openmmml import MLPotential

rtol = 1e-4
platform_ints = range(mm.Platform.getNumPlatforms())
test_data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


# ---------------------------------------------------------------------------
# A minimal model: energy = sum_i z_i * scale (z_i = atomic number).
# Uses a neighbor list so that the interface is exercised end-to-end.
# ---------------------------------------------------------------------------

class _LinearEnergyModel(torch.nn.Module):
    """
    Energy = scale * sum_i( z_i * sum_j(positions[i,j]) )

    This is a non-physical but differentiable energy that lets us verify both
    the energy value and the gradient (forces) analytically.  Forces on atom i
    are -dE/dr_i = -scale * z_i * [1, 1, 1] (in the model's length unit).
    """

    def __init__(self, scale: float = 0.01):
        super().__init__()
        self.scale = torch.tensor(scale, dtype=torch.float64)

    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels] = None,
    ) -> Dict[str, TensorMap]:
        results: Dict[str, TensorMap] = {}
        if "energy" not in outputs:
            return results

        energies = torch.zeros(len(systems), 1, dtype=torch.float64)
        for i, system in enumerate(systems):
            if selected_atoms is None:
                pos = system.positions
                types = system.types
            else:
                mask = selected_atoms.column("system") == i
                atom_ids = selected_atoms.column("atom")[mask]
                pos = system.positions[atom_ids]
                types = system.types[atom_ids]
            # sum_i z_i * (x_i + y_i + z_i)
            energies[i, 0] = self.scale * (
                types.to(torch.float64) * pos.sum(dim=1)
            ).sum()

        systems_idx = torch.arange(len(systems), dtype=torch.int32).reshape(-1, 1)
        block = TensorBlock(
            values=energies,
            samples=Labels(["system"], systems_idx),
            components=torch.jit.annotate(List[Labels], []),
            properties=Labels(["energy"], torch.tensor([[0]], dtype=torch.int32)),
        )
        results["energy"] = TensorMap(
            keys=Labels(["_"], torch.tensor([[0]], dtype=torch.int32)),
            blocks=[block],
        )
        return results

    def requested_neighbor_lists(self) -> List[NeighborListOptions]:
        # Request a small neighbor list so the NL computation path is exercised.
        return [NeighborListOptions(cutoff=3.0, full_list=True, strict=True)]


def _make_model(scale: float = 0.01) -> AtomisticModel:
    module = _LinearEnergyModel(scale=scale).eval()
    capabilities = ModelCapabilities(
        length_unit="angstrom",
        atomic_types=[1, 6, 7, 8],  # H, C, N, O  – covers alanine dipeptide
        interaction_range=3.0,
        outputs={
            "energy": ModelOutput(
                quantity="energy",
                unit="eV",
                per_atom=False,
                explicit_gradients=[],
            ),
        },
        supported_devices=["cpu"],
        dtype="float64",
    )
    metadata = ModelMetadata(name="test-linear-energy")
    return AtomisticModel(module, metadata, capabilities)


def _expected_energy_kj(positions_angstrom, atomic_numbers):
    """Return the expected energy in kJ/mol for the linear model with scale=0.01.

    E (eV) = scale * sum_i( z_i * (x_i + y_i + z_i) )   [positions in Angstrom]
    """
    eV_to_kJmol = 96.48533212331  # CODATA value used by metatomic
    scale = 0.01
    e_eV = scale * sum(
        z * positions_angstrom[i].sum()
        for i, z in enumerate(atomic_numbers)
    )
    return float(e_eV) * eV_to_kJmol


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("platform_int", list(platform_ints))
class TestMetatomicPotential:

    def testCreatePureMLSystem(self, platform_int, tmp_path):
        """A pure ML system produces the expected energy."""
        model_path = str(tmp_path / "model.pt")
        _make_model().save(model_path)

        pdb = app.PDBFile(
            os.path.join(test_data_dir, "alanine-dipeptide", "alanine-dipeptide-explicit.pdb")
        )
        topology = pdb.topology
        # Use only the solute (first chain) to keep the test fast.
        ml_atoms = [a.index for a in next(topology.chains()).atoms()]
        all_atoms = list(topology.atoms())

        # Build a sub-topology containing only the ML atoms.
        sub_topology = app.Topology()
        chain = sub_topology.addChain()
        residue = sub_topology.addResidue("ALA", chain)
        for idx in ml_atoms:
            sub_topology.addAtom(
                all_atoms[idx].name, all_atoms[idx].element, residue
            )

        positions_nm = pdb.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        sub_positions = np.array([positions_nm[i] for i in ml_atoms]) * unit.nanometer

        potential = MLPotential("metatomic", modelPath=model_path)
        system = potential.createSystem(sub_topology)
        platform = mm.Platform.getPlatform(platform_int)
        context = mm.Context(system, mm.VerletIntegrator(0.001), platform)
        context.setPositions(sub_positions)

        energy = context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
            unit.kilojoules_per_mole
        )

        # Compute expected energy from positions in Angstrom.
        pos_ang = np.array([positions_nm[i] for i in ml_atoms]) * 10.0  # nm → Å
        atomic_numbers = [all_atoms[i].element.atomic_number for i in ml_atoms]
        expected = _expected_energy_kj(pos_ang, atomic_numbers)

        assert np.isclose(energy, expected, rtol=rtol), (
            f"Expected {expected:.4f} kJ/mol, got {energy:.4f} kJ/mol"
        )

    def testCreateMixedSystem(self, platform_int, tmp_path):
        """A mixed MM/ML system: ML energy is added to the MM energy."""
        model_path = str(tmp_path / "model.pt")
        _make_model().save(model_path)

        pdb = app.PDBFile(
            os.path.join(test_data_dir, "alanine-dipeptide", "alanine-dipeptide-explicit.pdb")
        )
        topology = pdb.topology
        ff = app.ForceField("amber14-all.xml", "amber14/tip3pfb.xml")
        mm_system = ff.createSystem(topology, nonbondedMethod=app.PME)
        ml_atoms = [a.index for a in next(topology.chains()).atoms()]

        potential = MLPotential("metatomic", modelPath=model_path)
        mixed_system = potential.createMixedSystem(topology, mm_system, ml_atoms, interpolate=False)
        interp_system = potential.createMixedSystem(topology, mm_system, ml_atoms, interpolate=True)

        platform = mm.Platform.getPlatform(platform_int)
        mm_ctx = mm.Context(mm_system, mm.VerletIntegrator(0.001), platform)
        mixed_ctx = mm.Context(mixed_system, mm.VerletIntegrator(0.001), platform)
        interp_ctx = mm.Context(interp_system, mm.VerletIntegrator(0.001), platform)

        mm_ctx.setPositions(pdb.positions)
        mixed_ctx.setPositions(pdb.positions)
        interp_ctx.setPositions(pdb.positions)

        mm_energy = mm_ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
            unit.kilojoules_per_mole
        )
        mixed_energy = mixed_ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
            unit.kilojoules_per_mole
        )
        interp_energy1 = interp_ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
            unit.kilojoules_per_mole
        )
        interp_ctx.setParameter("lambda_interpolate", 0)
        interp_energy0 = interp_ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(
            unit.kilojoules_per_mole
        )

        # At lambda=1 the interpolated system should match the mixed system.
        assert np.isclose(mixed_energy, interp_energy1, rtol=rtol), (
            f"mixed={mixed_energy:.4f}, interp(λ=1)={interp_energy1:.4f}"
        )
        # At lambda=0 the interpolated system should match the pure MM system.
        assert np.isclose(mm_energy, interp_energy0, rtol=rtol), (
            f"mm={mm_energy:.4f}, interp(λ=0)={interp_energy0:.4f}"
        )
