from abc import ABC, abstractmethod
from typing import Any, Literal, Optional, Union

import torch

from ..core.quadratic_polynomial import QuadraticPolynomial


class ABCQuadraticModel(ABC):
    domain: str
    sense: str

    @abstractmethod
    def _as_quadratic_polynomial(
        self, dtype: Optional[torch.dtype], device: Optional[Union[str, torch.device]]
    ) -> QuadraticPolynomial:
        raise NotImplementedError() # pragma: no cover

    @abstractmethod
    def _from_optimized_tensor(
        self, optimized_tensor: torch.Tensor, optimized_cost: torch.Tensor
    ) -> Any:
        raise NotImplementedError() # pragma: no cover

    def solve(
        self,
        dtype: Optional[torch.dtype] = None,
        device: Optional[Union[str, torch.device]] = None,
        agents: int = 128,
        max_steps: int = 10_000,
        mode: Literal["ballistic", "discrete"] = "ballistic",
        heated: bool = False,
        verbose: bool = True,
        early_stopping: bool = True,
        sampling_period: int = 50,
        convergence_threshold: int = 50,
        timeout: Optional[float] = None,
    ) -> Any:
        quadratic_model = self._as_quadratic_polynomial(dtype=dtype, device=device)
        kwargs = {
            "domain": self.domain,
            "agents": agents,
            "max_steps": max_steps,
            "mode": mode,
            "heated": heated,
            "best_only": True,
            "verbose": verbose,
            "early_stopping": early_stopping,
            "sampling_period": sampling_period,
            "convergence_threshold": convergence_threshold,
            "timeout": timeout,
        }
        if self.sense == "minimize":
            return self._from_optimized_tensor(*quadratic_model.minimize(**kwargs))
        elif self.sense == "maximize":
            return self._from_optimized_tensor(*quadratic_model.maximize(**kwargs))
        else: # pragma: no cover
            raise ValueError(
                f'Unknown optimization sense {self.sense}. Expected "maximize" or "minimize".'
            )
