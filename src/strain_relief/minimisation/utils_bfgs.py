from collections.abc import Iterator
from typing import Any

import numpy as np
from ase import Atoms
from ase.optimize import BFGS
from loguru import logger as logging


class StrainReliefBFGS(BFGS):
    """BFGS optimiser with exit conditions for strain relief.

    Exit conditions:
    1. Maximum force on any atom > fexit (dynamics exploding).
    2. Number of steps exceeds max_steps.
    3. Forces have converged (max force < fmax).
    """

    def __init__(self, atoms: Atoms, **kwargs: Any) -> None:
        super().__init__(atoms, **kwargs)

    def run(self, steps: int, fmax: float = 0.05, fexit: float = 250) -> bool:
        """Run optimizer.

        Parameters
        ----------
        steps : int
            Number of optimizer steps to be run.
        fmax : float
            Convergence criterion (max |F| < fmax).
        fexit : float
            Exit criterion (abort if max |F| > fexit).

        Returns
        -------
        bool
            True if converged.
        """
        self.fmax = fmax
        self.n_fmax = 0.0  # maximum force on atom at step n
        self.fexit = fexit
        return self.dynamics_run(steps=steps)

    def dynamics_run(self, steps: int) -> bool:
        """Run dynamics algorithm until convergence or exit.

        Parameters
        ----------
        steps : int
            Number of dynamics steps to attempt.

        Returns
        -------
        bool
            True if converged.
        """
        for converged in self.dynamics_irun(steps=steps):
            pass
        return converged

    def dynamics_irun(self, steps: int) -> Iterator[bool]:
        """Generator form of dynamics loop.

        Parameters
        ----------
        steps : int
            Number of dynamics steps to attempt.

        Yields
        ------
        bool
            Convergence status after each (attempted) step.
        """
        # update the maximum number of steps
        self.max_steps = self.nsteps + steps

        # initial gradient
        self.gradient = self.optimizable.get_gradient()
        self.n_fmax = float(np.linalg.norm(self.gradient.reshape(-1, 3), axis=1).max())

        is_exit = self.exit()
        yield self.converged(self.gradient)

        while not is_exit:
            self.step()
            self.nsteps += 1

            self.gradient = self.optimizable.get_gradient()
            self.n_fmax = float(np.linalg.norm(self.gradient.reshape(-1, 3), axis=1).max())

            is_exit = self.exit()
            yield self.converged()

    def converged(self, gradient: np.ndarray | None = None) -> bool:
        """Check force convergence.

        Parameters
        ----------
        gradient : np.ndarray | None
            Gradient to test (1D flattened). If None, uses current gradient.

        Returns
        -------
        bool
            True if max force < fmax.
        """
        if gradient is None:
            gradient = self.gradient
        assert gradient.ndim == 1
        return self.optimizable.converged(gradient, self.fmax)

    def exit(self) -> bool:
        """Check exit conditions (explosion, max steps, convergence)."""
        if (self.n_fmax > self.fexit) or (self.nsteps >= self.max_steps) or self.converged():
            self.log()
            return True
        return False

    def log(self) -> None:
        """Log optimisation status for current conformer."""
        e = self.optimizable.get_value()
        name = self.__class__.__name__

        if self.nsteps >= self.max_steps:
            msg = (
                f"{name} CONFORMER NOT CONVERGED: Steps={self.nsteps}, "
                f"fmax={self.n_fmax}, E={e} (max steps={self.max_steps})"
            )
        elif self.n_fmax > self.fexit:
            msg = (
                f"{name} CONFORMER NOT CONVERGED: Steps={self.nsteps}, "
                f"fmax={self.n_fmax}, E={e} (fmax > {self.fexit})"
            )
        elif self.converged():
            msg = f"{name} CONFORMER CONVERGED: Steps={self.nsteps}, " f"fmax={self.n_fmax}, E={e}"
        else:
            msg = f"{name} STATUS: Steps={self.nsteps}, fmax={self.n_fmax}, E={e}"

        logging.debug(msg)
