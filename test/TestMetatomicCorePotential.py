from pathlib import Path

import numpy as np
import openmm as mm
import pytest
from openmm import app, unit

from openmmml import MLPotential

metatomic = pytest.importorskip("metatomic")
from metatomic.testing import LennardJonesModel  # noqa: E402

LJ = dict(sigma=3.4, epsilon=0.0104, cutoff=7.0, atomic_type=18, length_unit="angstrom", energy_unit="eV")
LJ_PLUGIN = Path(metatomic.__file__).parent / "libexec/metatomic/metatomic-lj-plugin.so"


def argon(n=3, a=0.526, seed=0):
    """rattled fcc argon, positions and box in nm"""
    basis = np.array([[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]])
    cells = np.array([[i, j, k] for i in range(n) for j in range(n) for k in range(n)])
    positions = a * (cells[:, None] + basis).reshape(-1, 3)
    positions += np.random.default_rng(seed).normal(scale=0.005, size=positions.shape)
    topology = app.Topology()
    residue = topology.addResidue("AR", topology.addChain())
    for _ in positions:
        topology.addAtom("Ar", app.element.argon, residue)
    topology.setPeriodicBoxVectors(np.eye(3) * n * a * unit.nanometer)
    return topology, positions


def evaluate(potential, topology, positions):
    system = potential.createSystem(topology)
    context = mm.Context(system, mm.VerletIntegrator(0.001), mm.Platform.getPlatformByName("Reference"))
    context.setPositions(positions)
    state = context.getState(getEnergy=True, getForces=True)
    return (
        state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole),
        state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer),
    )


def test_numpy_model_matches_finite_differences():
    topology, positions = argon(n=2)
    potential = MLPotential("metatomic-core", model=LennardJonesModel(**LJ))
    energy, forces = evaluate(potential, topology, positions)
    h, atom = 1e-6, 3
    for axis in range(3):
        shift = np.zeros_like(positions)
        shift[atom, axis] = h
        fd = (evaluate(potential, topology, positions - shift)[0] - evaluate(potential, topology, positions + shift)[0]) / (2 * h)
        assert abs(fd - forces[atom, axis]) < 1e-4 * max(1.0, abs(fd))
    assert energy < 0


@pytest.mark.skipif(not LJ_PLUGIN.exists(), reason="metatomic-lj-plugin.so is not installed")
def test_native_plugin_matches_numpy_model():
    topology, positions = argon()
    native = MLPotential(
        "metatomic-core", model="metatomic-lj-model", plugins=[LJ_PLUGIN], modelOptions=LJ
    )
    reference = MLPotential("metatomic-core", model=LennardJonesModel(**LJ))
    (e1, f1), (e2, f2) = evaluate(native, topology, positions), evaluate(reference, topology, positions)
    assert abs(e1 - e2) < 1e-8
    assert np.abs(f1 - f2).max() < 1e-8


def test_mixed_system_subset():
    topology, positions = argon(n=2)
    potential = MLPotential("metatomic-core", model=LennardJonesModel(**LJ))
    system = mm.System()
    for _ in positions:
        system.addParticle(39.948)
    potential._impl.addForces(topology, system, range(4), 0)
    context = mm.Context(system, mm.VerletIntegrator(0.001), mm.Platform.getPlatformByName("Reference"))
    context.setPositions(positions)
    forces = context.getState(getForces=True).getForces(asNumpy=True)._value
    assert np.all(forces[4:] == 0) and np.abs(forces[:4]).max() > 0
