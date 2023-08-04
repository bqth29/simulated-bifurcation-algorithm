import time

import torch
from optimizer_mode import OptimizerMode
from symplectic_integrator import SymplecticIntegrator


class FasterSymplecticIntegrator(SymplecticIntegrator):
    def momentum_update(self, coefficient: float) -> None:
        torch.add(self.momentum, self.position, alpha=coefficient, out=self.momentum)

    def position_update(self, coefficient: float) -> None:
        torch.add(self.position, self.momentum, alpha=coefficient, out=self.position)

    def quadratic_position_update(
        self, coefficient: float, matrix: torch.Tensor
    ) -> None:
        # do not use out=self.position because of side effects
        self.position = torch.addmm(
            self.position,
            matrix,
            self.activation_function(self.momentum),
            alpha=coefficient,
        )


def sanity_check():
    dimension = 69
    n_agents = 42
    shape = (dimension, n_agents)
    mode = OptimizerMode.BALLISTIC
    dtype = torch.float32
    device = "cpu"
    momentum_coefficient = 0.001
    position_coefficient = 0.001
    quadratic_coefficient = 0.001
    matrix = (
        0.2 * torch.rand(size=(dimension, dimension), device=device, dtype=dtype) - 0.1
    )

    reference_integrator = SymplecticIntegrator(shape, mode, dtype, device)
    faster_integrator = FasterSymplecticIntegrator(shape, mode, dtype, device)
    faster_integrator.momentum = torch.clone(reference_integrator.momentum)
    faster_integrator.position = torch.clone(reference_integrator.position)

    for _ in range(74):
        reference_integrator.step(
            momentum_coefficient, position_coefficient, quadratic_coefficient, matrix
        )
        faster_integrator.step(
            momentum_coefficient, position_coefficient, quadratic_coefficient, matrix
        )
        assert torch.all(
            torch.isclose(faster_integrator.momentum, reference_integrator.momentum)
        )
        assert torch.all(
            torch.isclose(faster_integrator.position, reference_integrator.position)
        )

    print("\n\nSanity check successfully passed\n")


def runtime_comparison(f_1, f_2, n_repeats, *args):
    start = time.time()
    for _ in range(n_repeats):
        f_1(*args)
    end = time.time()
    t_1 = (end - start) / n_repeats
    start = time.time()
    for _ in range(n_repeats):
        f_2(*args)
    end = time.time()
    t_2 = (end - start) / n_repeats
    return t_1, t_2


def main():
    sanity_check()

    mode = OptimizerMode.BALLISTIC
    dtype = torch.float32
    momentum_coefficient = 0.001
    position_coefficient = 0.001
    quadratic_coefficient = 0.001
    n_fast_repeats = 10_000_000
    n_slow_repeats = 100_000_000

    if torch.cuda.is_available():
        devices = ["cpu", "cuda"]
    else:
        devices = ["cpu"]
    dimensions = [100, 300, 800, 2_000, 5_000, 10_000]
    n_agents = [1, 4, 16, 32, 64, 128, 256]

    for device in devices:
        for dim in dimensions:
            for n in n_agents:
                shape = (dim, n)
                reference_integrator = SymplecticIntegrator(shape, mode, dtype, device)
                faster_integrator = FasterSymplecticIntegrator(
                    shape, mode, dtype, device
                )
                matrix = (
                    0.2
                    * dim**0.5
                    * torch.rand(size=(dim, dim), device=device, dtype=dtype)
                    - 0.1 * dim**0.5
                )
                # t_1, t_2 = runtime_comparison(
                #     faster_integrator.momentum_update,
                #     reference_integrator.momentum_update,
                #     n_fast_repeats // dim,
                #     momentum_coefficient,
                # )
                t_1, t_2 = runtime_comparison(
                    faster_integrator.quadratic_position_update,
                    reference_integrator.quadratic_position_update,
                    round(n_slow_repeats / dim**1.5),
                    quadratic_coefficient,
                    matrix,
                )
                print(
                    f"{device: >4} dim {dim:5d} {n:3d} agents    "
                    f"fast {t_1 * 1000:6.2f}ms current {t_2 * 1000:6.2f}ms"
                )


if __name__ == "__main__":
    main()
