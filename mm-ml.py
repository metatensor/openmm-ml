"""
MM/ML simulation of alanine dipeptide in a 4.0 nm explicit-water box.

The solute (first chain) is described by pet-spice-s. The
solvent and solute-solvent interactions are described by AMBER14.
"""

import sys
import argparse

import numpy as np
import openmm as mm
import openmm.app as app
import openmm.unit as unit

from openmmml import MLPotential


PDB_FILE = "test/data/alanine-dipeptide/alanine-dipeptide-explicit.pdb"
ML_MODEL = "pet-spice-s"
DEVICE = "cuda"
NUM_STEPS = 5
REPORT_INTERVAL = 1
TRAJECTORY_FILE = "mm-ml.pdb"
MINIMIZATION_TOLERANCE = 10.0 * unit.kilojoule_per_mole / unit.nanometer
MINIMIZATION_STEPS = 100
MINIMIZATION_STEP_SIZE = 0.001 * unit.nanometer


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=NUM_STEPS, help="Number of MD steps")
    parser.add_argument("--trajectory", default=TRAJECTORY_FILE, help="Trajectory file to write")
    parser.add_argument("--report-interval", type=int, default=REPORT_INTERVAL, help="Reporter interval in steps")
    parser.add_argument("--device", default=DEVICE, help="Torch device for the ML model")
    parser.add_argument("--minimize", action=argparse.BooleanOptionalAction, default=True, help="Run energy minimization before MD")
    parser.add_argument("--minimize-steps", type=int, default=MINIMIZATION_STEPS, help="Maximum minimization iterations")
    return parser.parse_args()


def load_system(forcefield):
    pdb = app.PDBFile(PDB_FILE)
    return pdb.topology, pdb.positions


def minimize_with_forces(simulation, max_iterations):
    tolerance = MINIMIZATION_TOLERANCE.value_in_unit(unit.kilojoule_per_mole / unit.nanometer)
    step_size = MINIMIZATION_STEP_SIZE.value_in_unit(unit.nanometer)

    state = simulation.context.getState(getEnergy=True, getForces=True, getPositions=True)
    energy = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    positions = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)

    for iteration in range(max_iterations):
        forces = state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer)
        force_norms = np.linalg.norm(forces, axis=1)
        max_force = force_norms.max()
        if max_force < tolerance:
            print(f"Minimization converged at iteration {iteration}, max force {max_force:.3f} kJ/mol/nm")
            return

        accepted = False
        for _ in range(20):
            trial_positions = positions + step_size * forces / max_force
            simulation.context.setPositions(trial_positions * unit.nanometer)
            simulation.context.applyConstraints(1e-6)
            trial_state = simulation.context.getState(getEnergy=True, getForces=True, getPositions=True)
            trial_energy = trial_state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
            if trial_energy < energy:
                state = trial_state
                energy = trial_energy
                positions = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
                step_size *= 1.1
                accepted = True
                break
            step_size *= 0.5

        if not accepted:
            simulation.context.setPositions(positions * unit.nanometer)
            print(f"Minimization stopped at iteration {iteration}; no lower-energy trial step found")
            return

    print(f"Minimization reached {max_iterations} iterations")


def main():
    args = parse_args()
    forcefield = app.ForceField("amber14-all.xml", "amber14/tip3pfb.xml")
    topology, positions = load_system(forcefield)

    mm_system = forcefield.createSystem(
        topology,
        nonbondedMethod=app.PME,
        nonbondedCutoff=0.9 * unit.nanometer,
        constraints=app.HBonds,
    )

    ml_atoms = [atom.index for atom in next(topology.chains()).atoms()]
    print(
        f"ML atoms: {len(ml_atoms)} (solute), "
        f"MM atoms: {topology.getNumAtoms() - len(ml_atoms)} (solvent)"
    )

    potential = MLPotential(ML_MODEL)
    system = potential.createMixedSystem(
        topology,
        mm_system,
        ml_atoms,
        interpolate=False,
        device=args.device,
    )

    integrator = mm.LangevinMiddleIntegrator(
        300 * unit.kelvin,
        1.0 / unit.picosecond,
        2.0 * unit.femtosecond,
    )
    simulation = app.Simulation(topology, system, integrator)
    simulation.context.setPositions(positions)

    if args.minimize:
        print(f"Minimizing energy for up to {args.minimize_steps} iterations...")
        minimize_with_forces(simulation, args.minimize_steps)

    print(f"Writing trajectory to {args.trajectory}")
    simulation.reporters.append(app.PDBReporter(args.trajectory, args.report_interval))

    print(f"Running {args.steps} MM/ML steps...")
    simulation.reporters.append(
        app.StateDataReporter(
            sys.stdout,
            args.report_interval,
            step=True,
            potentialEnergy=True,
            temperature=True,
            speed=True,
        )
    )
    simulation.step(args.steps)
    print("Done.")


if __name__ == "__main__":
    main()
