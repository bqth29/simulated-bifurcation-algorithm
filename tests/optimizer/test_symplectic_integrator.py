import torch

from src.simulated_bifurcation.optimizer import SymplecticIntegrator


def test_init_ballistic_symplectic_integrator():
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
    assert torch.equal(
        symplectic_integrator.sample_spins(),
        torch.tensor(
            [
                [-1, -1],
                [-1, 1],
                [-1, 1],
            ],
            dtype=torch.float32,
        ),
    )


def test_init_discrete_symplectic_integrator():
    symplectic_integrator = SymplecticIntegrator(
        (3, 2), torch.sign, torch.float32, "cpu"
    )
    symplectic_integrator.position = torch.tensor(
        [
            [-0.7894, -0.4610],
            [-0.2343, 0.9186],
            [-0.2191, 0.2018],
        ],
        dtype=torch.float32,
    )
    assert torch.equal(
        symplectic_integrator.sample_spins(),
        torch.tensor(
            [
                [-1, -1],
                [-1, 1],
                [-1, 1],
            ],
            dtype=torch.float32,
        ),
    )


def test_position_update():
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
                dtype=torch.float32,
            ),
            atol=1e-4,
        )
    )


def test_momentum_update():
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
                dtype=torch.float32,
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


def test_inelastic_walls_simulation():
    symplectic_integrator = SymplecticIntegrator(
        (3, 2), torch.nn.Identity(), torch.float32, "cpu"
    )
    symplectic_integrator.position = torch.tensor(
        [
            [-2.7894, -0.4610],
            [-1.2343, 1.9186],
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
                dtype=torch.float32,
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
                dtype=torch.float32,
            ),
            atol=1e-4,
        )
    )


def test_full_step():
    symplectic_integrator = SymplecticIntegrator(
        (3, 2), torch.nn.Identity(), torch.float32, "cpu"
    )
    symplectic_integrator.position = torch.tensor(
        [
            [-2.7894, -0.4610],
            [-1.2343, 1.9186],
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
            dtype=torch.float32,
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
                dtype=torch.float32,
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
                dtype=torch.float32,
            ),
            atol=1e-4,
        )
    )
