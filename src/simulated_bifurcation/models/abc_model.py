from abc import ABC
from typing import List, Literal, Optional, Tuple, Union

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
        *,
        agents: int = 128,
        max_steps: int = 10000,
        best_only: bool = True,
        mode: Literal["ballistic", "discrete"] = "ballistic",
        heated: bool = False,
        minimize: bool = True,
        verbose: bool = True,
        early_stopping: bool = True,
        sampling_period: int = 50,
        convergence_threshold: int = 50,
        timeout: Optional[float] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return super().optimize(
            domain=self.domain,
            agents=agents,
            max_steps=max_steps,
            best_only=best_only,
            mode=mode,
            heated=heated,
            minimize=minimize,
            verbose=verbose,
            early_stopping=early_stopping,
            sampling_period=sampling_period,
            convergence_threshold=convergence_threshold,
            timeout=timeout,
            dtype=dtype,
        )

    def minimize(
        self,
        *,
        agents: int = 128,
        max_steps: int = 10000,
        best_only: bool = True,
        mode: Literal["ballistic", "discrete"] = "ballistic",
        heated: bool = False,
        verbose: bool = True,
        early_stopping: bool = True,
        sampling_period: int = 50,
        convergence_threshold: int = 50,
        timeout: Optional[float] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.optimize(
            agents=agents,
            max_steps=max_steps,
            best_only=best_only,
            mode=mode,
            heated=heated,
            minimize=True,
            verbose=verbose,
            early_stopping=early_stopping,
            sampling_period=sampling_period,
            convergence_threshold=convergence_threshold,
            timeout=timeout,
            dtype=dtype,
        )

    def maximize(
        self,
        *,
        agents: int = 128,
        max_steps: int = 10000,
        best_only: bool = True,
        mode: Literal["ballistic", "discrete"] = "ballistic",
        heated: bool = False,
        verbose: bool = True,
        early_stopping: bool = True,
        sampling_period: int = 50,
        convergence_threshold: int = 50,
        timeout: Optional[float] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.optimize(
            agents=agents,
            max_steps=max_steps,
            best_only=best_only,
            mode=mode,
            heated=heated,
            minimize=False,
            verbose=verbose,
            early_stopping=early_stopping,
            sampling_period=sampling_period,
            convergence_threshold=convergence_threshold,
            timeout=timeout,
            dtype=dtype,
        )
