import logging
from math import sqrt

from ase.optimize import BFGS


class StrainReliefBFGS(BFGS):
    """BFGS optimiser with exit conditions for strain relief.

    This class is a subclass of the BFGS optimiser from the Atomic Simulation Environment (ASE).
    It adds an additional exit condition for strain relief to the BFGS optimiser (1. see below).

    The exit conditions are:
    1. The maximum force on an atom is > fexit (dynamics are exploding).
    2. The number of steps exceeds max_steps.
    3. The minimisation has converged.
    """

    def __init__(self, atoms, **kwargs):
        super().__init__(atoms, **kwargs)

    def run(self, steps, fmax=0.05, fexit=250):
        """Run optimizer.

        Parameters
        ----------
        steps : int
            Number of optimizer steps to be run.
        fmax : float
            Convergence criterion of the forces on atoms.
        fexit : int
            Exit criterion of the forces

        Returns
        -------
        converged : bool
            True if the forces on atoms are converged.
        """
        self.fmax = fmax
        self.n_fmax = 0.0  # maximum force on atom at step n
        self.fexit = fexit
        return self.dynamics_run(steps=steps)

    def dynamics_run(self, steps):
        """Run dynamics algorithm.

        This method will return when the forces on all individual
        atoms are less than *fmax* or when the number of steps exceeds
        *steps*.

        Parameters
        ----------
        steps : int, default=DEFAULT_MAX_STEPS
            Number of dynamics steps to be run.

        Returns
        -------
        converged : bool
            True if the forces on atoms are converged.
        """
        for converged in self.dynamics_irun(steps=steps):
            pass
        return converged

    def dynamics_irun(self, steps):
        """Run dynamics algorithm as generator.

        Parameters
        ----------
        steps : int
            Number of dynamics steps to be run.

        Yields
        ------
        converged : bool
            True if the forces on atoms are converged.
        """
        # update the maximum number of steps
        self.max_steps = self.nsteps + steps

        # compute the initial step
        forces = self.optimizable.get_forces()
        self.n_fmax = sqrt((forces**2).sum(axis=1).max())

        # check exit condition
        is_exit = self.exit()
        yield self.converged(forces=forces)

        while not is_exit:
            # compute the next step
            self.step()
            self.nsteps += 1

            forces = self.optimizable.get_forces()
            self.n_fmax = sqrt((forces**2).sum(axis=1).max())

            # check exit conditions
            is_exit = self.exit()
            yield self.converged(forces=forces)

    def exit(self):
        """Check exit conditions.

        The minimisationwill exit if:
        1. The maximum force on an atom is > fexit (dynamics are exploding).
        2. The number of steps exceeds max_steps.
        3. The minimisation has converged.
        """
        if (self.n_fmax > self.fexit) or (self.nsteps >= self.max_steps) or self.converged():
            self.log()
            return True
        return False

    def log(self):
        e = self.optimizable.get_potential_energy()
        name = self.__class__.__name__

        if self.nsteps >= self.max_steps:
            msg = f"""{name} CONFORMER NOT CONVEREGED: Steps = {self.nsteps},
                fmax = {self.n_fmax}, E = {e} (max steps = {self.max_steps})"""
        elif self.n_fmax > self.fexit:
            msg = f"""{name} CONFORMER NOT CONVEREGED: Steps = {self.nsteps},
                fmax = {self.n_fmax}, E = {e} (fmax > {self.fexit})"""
        elif self.converged():
            msg = (
                f"{name} CONFORMER CONVEREGED: Steps = {self.nsteps}, fmax = {self.n_fmax}, E = {e}"
            )
        logging.debug(msg)
