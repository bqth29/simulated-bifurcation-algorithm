import pytest
import torch

from src.simulated_bifurcation.optimizer import (
    SimulatedBifurcationEngine,
    SymplecticIntegrator,
)

from ..utils import DEVICES, FLOAT_DTYPES


@pytest.mark.parametrize(
    "dtype, device, engine",
    [
        (dtype, device, engine)
        for dtype in FLOAT_DTYPES
        for device in DEVICES
        for engine in SimulatedBifurcationEngine
    ],
)
def test_init_symplectic_integrator(
    dtype: torch.dtype, device: str, engine: SimulatedBifurcationEngine
):
    symplectic_integrator = SymplecticIntegrator(
        (3, 2), engine.activation_function, dtype, device
    )
    assert torch.all(torch.abs(symplectic_integrator.position) < 1)
    assert torch.all(torch.abs(symplectic_integrator.momentum) < 1)


@pytest.mark.parametrize(
    "dtype, device, engine",
    [
        (dtype, device, engine)
        for dtype in FLOAT_DTYPES
        for device in DEVICES
        for engine in SimulatedBifurcationEngine
    ],
)
def test_sample_spins(
    dtype: torch.dtype, device: str, engine: SimulatedBifurcationEngine
):
    symplectic_integrator = SymplecticIntegrator(
        (3, 2), engine.activation_function, dtype, device
    )
    symplectic_integrator.position = torch.tensor(
        [
            [-0.7894, -0.4610],
            [-0.2343, 0.9186],
            [-0.2191, 0.2018],
        ],
        dtype=dtype,
        device=device,
    )
    assert torch.equal(
        symplectic_integrator.sample_spins(),
        torch.tensor(
            [
                [-1, -1],
                [-1, 1],
                [-1, 1],
            ],
            dtype=dtype,
            device=device,
        ),
    )


@pytest.mark.parametrize(
    "dtype, device, engine",
    [
        (dtype, device, engine)
        for dtype in FLOAT_DTYPES
        for device in DEVICES
        for engine in SimulatedBifurcationEngine
    ],
)
def test_position_update(
    dtype: torch.dtype, device: str, engine: SimulatedBifurcationEngine
):
    symplectic_integrator = SymplecticIntegrator(
        (3, 2), engine.activation_function, dtype, device
    )
    symplectic_integrator.position = torch.tensor(
        [
            [-0.7894, -0.4610],
            [-0.2343, 0.9186],
            [-0.2191, 0.2018],
        ],
        dtype=dtype,
        device=device,
    )
    symplectic_integrator.momentum = torch.tensor(
        [
            [-0.4869, 0.5873],
            [0.8815, -0.7336],
            [0.8692, 0.1872],
        ],
        dtype=dtype,
        device=device,
    )
    symplectic_integrator.position_update(0.2)
    assert torch.all(
        torch.isclose(
            symplectic_integrator.position,
            torch.tensor(
                [
                    [-0.8868, -0.3435],
                    [-0.0580, 0.7719],
                    [-0.0453, 0.2392],
                ],
                dtype=dtype,
                device=device,
            ),
            atol=1e-4,
        )
    )


@pytest.mark.parametrize(
    "dtype, device, engine",
    [
        (dtype, device, engine)
        for dtype in FLOAT_DTYPES
        for device in DEVICES
        for engine in SimulatedBifurcationEngine
    ],
)
def test_momentum_update(
    dtype: torch.dtype, device: str, engine: SimulatedBifurcationEngine
):
    symplectic_integrator = SymplecticIntegrator(
        (3, 2), engine.activation_function, dtype, device
    )
    symplectic_integrator.position = torch.tensor(
        [
            [-0.7894, -0.4610],
            [-0.2343, 0.9186],
            [-0.2191, 0.2018],
        ],
        dtype=dtype,
        device=device,
    )
    symplectic_integrator.momentum = torch.tensor(
        [
            [-0.4869, 0.5873],
            [0.8815, -0.7336],
            [0.8692, 0.1872],
        ],
        dtype=dtype,
        device=device,
    )
    symplectic_integrator.momentum_update(0.2)
    assert torch.all(
        torch.isclose(
            symplectic_integrator.momentum,
            torch.tensor(
                [
                    [-0.6448, 0.4951],
                    [0.8346, -0.5499],
                    [0.8254, 0.2276],
                ],
                dtype=dtype,
                device=device,
            ),
            atol=1e-4,
        )
    )


def test_quadratic_position_update():
    symplectic_integrator = SymplecticIntegrator(
        (3, 2), torch.nn.Identity(), torch.float32, "cpu"
    )
    symplectic_integrator.position = torch.tensor(
        [
            [-0.7894, -0.4610],
            [-0.2343, 0.9186],
            [-0.2191, 0.2018],
        ],
        dtype=torch.float32,
    )
    symplectic_integrator.momentum = torch.tensor(
        [
            [-0.4869, 0.5873],
            [0.8815, -0.7336],
            [0.8692, 0.1872],
        ],
        dtype=torch.float32,
    )
    symplectic_integrator.quadratic_momentum_update(
        0.2,
        torch.tensor(
            [
                [0, 0.2, 0.3],
                [0.2, 0, 0.1],
                [0.3, 0.1, 0],
            ],
            dtype=torch.float32,
        ),
    )
    assert torch.all(
        torch.isclose(
            symplectic_integrator.momentum,
            torch.tensor(
                [
                    [-0.5094, 0.6362],
                    [0.8455, -0.7480],
                    [0.8171, 0.1779],
                ],
                dtype=torch.float32,
            ),
            atol=1e-4,
        )
    )


@pytest.mark.parametrize(
    "dtype, device, engine",
    [
        (dtype, device, engine)
        for dtype in FLOAT_DTYPES
        for device in DEVICES
        for engine in SimulatedBifurcationEngine
    ],
)
def test_inelastic_walls_simulation(
    dtype: torch.dtype, device: str, engine: SimulatedBifurcationEngine
):
    symplectic_integrator = SymplecticIntegrator(
        (3, 2), engine.activation_function, dtype, device
    )
    symplectic_integrator.position = torch.tensor(
        [
            [-2.7894, -0.4610],
            [-1.2343, 1.9186],
            [-0.2191, 0.2018],
        ],
        dtype=dtype,
        device=device,
    )
    symplectic_integrator.momentum = torch.tensor(
        [
            [-0.4869, 0.5873],
            [0.8815, -0.7336],
            [0.8692, 0.1872],
        ],
        dtype=dtype,
        device=device,
    )
    symplectic_integrator.simulate_inelastic_walls()
    assert torch.all(
        torch.isclose(
            symplectic_integrator.position,
            torch.tensor(
                [
                    [-1, -0.4610],
                    [-1, 1],
                    [-0.2191, 0.2018],
                ],
                dtype=dtype,
                device=device,
            ),
            atol=1e-4,
        )
    )
    assert torch.all(
        torch.isclose(
            symplectic_integrator.momentum,
            torch.tensor(
                [
                    [0, 0.5873],
                    [0, 0],
                    [0.8692, 0.1872],
                ],
                dtype=dtype,
                device=device,
            ),
            atol=1e-4,
        )
    )


@pytest.mark.parametrize(
    "dtype, device, engine",
    [
        (dtype, device, engine)
        for dtype in FLOAT_DTYPES
        for device in DEVICES
        for engine in [SimulatedBifurcationEngine.bSB, SimulatedBifurcationEngine.HbSB]
    ],
)
def test_full_step_ballistic(
    dtype: torch.dtype, device: str, engine: SimulatedBifurcationEngine
):
    symplectic_integrator = SymplecticIntegrator(
        (3, 2), engine.activation_function, dtype, device
    )
    symplectic_integrator.position = torch.tensor(
        [
            [-2.7894, -0.4610],
            [-1.2343, 1.9186],
            [-0.2191, 0.2018],
        ],
        dtype=dtype,
        device=device,
    )
    symplectic_integrator.momentum = torch.tensor(
        [
            [-0.4869, 0.5873],
            [0.8815, -0.7336],
            [0.8692, 0.1872],
        ],
        dtype=dtype,
        device=device,
    )
    symplectic_integrator.step(
        0.2,
        0.2,
        0.2,
        torch.tensor(
            [
                [0, 0.2, 0.3],
                [0.2, 0, 0.1],
                [0.3, 0.1, 0],
            ],
            dtype=dtype,
            device=device,
        ),
    )
    assert torch.all(
        torch.isclose(
            symplectic_integrator.position,
            torch.tensor(
                [
                    [-1, -0.3620],
                    [-1, 1],
                    [-0.0540, 0.2473],
                ],
                dtype=dtype,
                device=device,
            ),
            atol=1e-4,
        )
    )
    assert torch.all(
        torch.isclose(
            symplectic_integrator.momentum,
            torch.tensor(
                [
                    [0, 0.5839],
                    [0, 0],
                    [0.6233, 0.2428],
                ],
                dtype=dtype,
                device=device,
            ),
            atol=1e-4,
        )
    )


@pytest.mark.parametrize(
    "dtype, device, engine",
    [
        (dtype, device, engine)
        for dtype in FLOAT_DTYPES
        for device in DEVICES
        for engine in [SimulatedBifurcationEngine.dSB, SimulatedBifurcationEngine.HdSB]
    ],
)
def test_full_step_discrete(
    dtype: torch.dtype, device: str, engine: SimulatedBifurcationEngine
):
    symplectic_integrator = SymplecticIntegrator(
        (3, 2), engine.activation_function, dtype, device
    )
    symplectic_integrator.position = torch.tensor(
        [
            [-2.7894, -0.4610],
            [-1.2343, 1.9186],
            [-0.2191, 0.2018],
        ],
        dtype=dtype,
        device=device,
    )
    symplectic_integrator.momentum = torch.tensor(
        [
            [-0.4869, 0.5873],
            [0.8815, -0.7336],
            [0.8692, 0.1872],
        ],
        dtype=dtype,
        device=device,
    )
    symplectic_integrator.step(
        0.2,
        0.2,
        0.2,
        torch.tensor(
            [
                [0, 0.2, 0.3],
                [0.2, 0, 0.1],
                [0.3, 0.1, 0],
            ],
            dtype=dtype,
            device=device,
        ),
    )
    assert torch.all(
        torch.isclose(
            symplectic_integrator.position,
            torch.tensor(
                [
                    [-1, -0.3620],
                    [-1, 1],
                    [-0.0540, 0.2473],
                ],
                dtype=dtype,
                device=device,
            ),
            atol=1e-4,
        )
    )
    assert torch.all(
        torch.isclose(
            symplectic_integrator.momentum,
            torch.tensor(
                [
                    [0, 0.5951],
                    [0, 0],
                    [0.7454, 0.1876],
                ],
                dtype=dtype,
                device=device,
            ),
            atol=1e-4,
        )
    )
