from abc import ABC
from typing import List, Optional, Tuple, Union

import torch

from ..core import QuadraticPolynomial


class ABCModel(ABC, QuadraticPolynomial):
    """
    Abstract class that serves as a base component to define quadratic
    optimization models that can be solved using the Simulated
    Bifurcation algorithm.

    Attributes
    ----------
    domain : str
        The optimization domain of the problem.

    """

    domain: Union[str, List[str]]

    def optimize(
        self,
        agents: int = 128,
        max_steps: int = 10000,
        best_only: bool = True,
        ballistic: bool = False,
        heated: bool = False,
        minimize: bool = True,
        verbose: bool = True,
        *,
        use_window: bool = True,
        sampling_period: int = 50,
        convergence_threshold: int = 50,
        timeout: Optional[float] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return super().optimize(
            self.domain,
            agents,
            max_steps,
            best_only,
            ballistic,
            heated,
            minimize,
            verbose,
            use_window=use_window,
            sampling_period=sampling_period,
            convergence_threshold=convergence_threshold,
            timeout=timeout,
        )

    def minimize(
        self,
        agents: int = 128,
        max_steps: int = 10000,
        best_only: bool = True,
        ballistic: bool = False,
        heated: bool = False,
        verbose: bool = True,
        *,
        use_window: bool = True,
        sampling_period: int = 50,
        convergence_threshold: int = 50,
        timeout: Optional[float] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.optimize(
            agents,
            max_steps,
            best_only,
            ballistic,
            heated,
            True,
            verbose,
            use_window=use_window,
            sampling_period=sampling_period,
            convergence_threshold=convergence_threshold,
            timeout=timeout,
        )

    def maximize(
        self,
        agents: int = 128,
        max_steps: int = 10000,
        best_only: bool = True,
        ballistic: bool = False,
        heated: bool = False,
        verbose: bool = True,
        *,
        use_window: bool = True,
        sampling_period: int = 50,
        convergence_threshold: int = 50,
        timeout: Optional[float] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.optimize(
            agents,
            max_steps,
            best_only,
            ballistic,
            heated,
            False,
            verbose,
            use_window=use_window,
            sampling_period=sampling_period,
            convergence_threshold=convergence_threshold,
            timeout=timeout,
        )
