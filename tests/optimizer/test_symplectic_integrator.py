from typing import Callable

import pytest
import torch

from src.simulated_bifurcation.optimizer import SymplecticIntegrator

from ..test_utils import DEVICES, DTYPES

initial_position = [[-0.7894, -0.4610], [-0.2343, 0.9186], [-0.2191, 0.2018]]
initial_momentum = [[-0.4869, 0.5873], [0.8815, -0.7336], [0.8692, 0.1872]]
quadratic_tensor = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]


def init_integrator(
    dtype: torch.dtype,
    device: torch.device,
    activation_function: Callable[[torch.Tensor], torch.Tensor],
    heat: bool,
) -> SymplecticIntegrator:
    symplectic_integrator = SymplecticIntegrator(
        2,
        0.1,
        0.01,
        0.06,
        activation_function,
        heat,
        torch.tensor(quadratic_tensor, dtype=dtype, device=device),
        dtype,
        device,
    )
    symplectic_integrator.position = torch.tensor(
        initial_position, dtype=dtype, device=device
    )
    symplectic_integrator.momentum = torch.tensor(
        initial_momentum, dtype=dtype, device=device
    )
    return symplectic_integrator


def assert_tensors_equality(expected: torch.Tensor, actual: torch.Tensor):
    assert torch.all(torch.isclose(expected, actual, atol=1e-4))


@pytest.mark.parametrize(
    "dtype, device", [(dtype, device) for dtype in DTYPES for device in DEVICES]
)
def test_sample_spins(dtype: torch.dtype, device: torch.device):
    symplectic_integrator = init_integrator(dtype, device, torch.nn.Identity(), False)
    assert torch.equal(
        symplectic_integrator.sample_spins(),
        torch.tensor(
            [[-1, -1], [-1, 1], [-1, 1]],
            dtype=dtype,
            device=device,
        ),
    )


@pytest.mark.parametrize(
    "dtype, device", [(dtype, device) for dtype in DTYPES for device in DEVICES]
)
def test_position_update(dtype: torch.dtype, device: torch.device):
    symplectic_integrator = init_integrator(dtype, device, torch.nn.Identity(), False)
    symplectic_integrator.position_update()
    assert_tensors_equality(
        torch.tensor(
            [[-0.8381, -0.4023], [-0.1461, 0.8452], [-0.1322, 0.2205]],
            dtype=dtype,
            device=device,
        ),
        symplectic_integrator.position,
    )


@pytest.mark.parametrize(
    "dtype, device", [(dtype, device) for dtype in DTYPES for device in DEVICES]
)
def test_momentum_update(dtype: torch.dtype, device: torch.device):
    symplectic_integrator = init_integrator(dtype, device, torch.nn.Identity(), False)
    symplectic_integrator.momentum_update()
    assert_tensors_equality(
        torch.tensor(
            [[-0.4080, 0.6334], [0.9049, -0.8255], [0.8911, 0.1670]],
            dtype=dtype,
            device=device,
        ),
        symplectic_integrator.momentum,
    )


@pytest.mark.parametrize(
    "dtype, device", [(dtype, device) for dtype in DTYPES for device in DEVICES]
)
def test_quadratic_position_update_ballistic(dtype: torch.dtype, device: torch.device):
    symplectic_integrator = init_integrator(dtype, device, torch.nn.Identity(), False)
    symplectic_integrator.quadratic_momentum_update()
    assert_tensors_equality(
        torch.tensor(
            [[-0.4949, 0.5956], [0.8579, -0.7170], [0.8299, 0.2121]],
            dtype=dtype,
            device=device,
        ),
        symplectic_integrator.momentum,
    )


@pytest.mark.parametrize(
    "dtype, device", [(dtype, device) for dtype in DTYPES for device in DEVICES]
)
def test_quadratic_position_update_discrete(dtype: torch.dtype, device: torch.device):
    symplectic_integrator = init_integrator(dtype, device, torch.sign, False)
    symplectic_integrator.quadratic_momentum_update()
    assert_tensors_equality(
        torch.tensor(
            [[-0.5120, 0.6041], [0.8187, -0.7043], [0.7687, 0.2291]],
            dtype=dtype,
            device=device,
        ),
        symplectic_integrator.momentum,
    )


@pytest.mark.parametrize(
    "dtype, device", [(dtype, device) for dtype in DTYPES for device in DEVICES]
)
def test_inelastic_walls_simulation(dtype: torch.dtype, device: torch.device):
    symplectic_integrator = init_integrator(dtype, device, torch.nn.Identity(), False)
    symplectic_integrator.position = torch.tensor(
        [[-2.7894, -0.4610], [-1.2343, 1.9186], [-0.2191, 0.2018]],
        dtype=dtype,
        device=device,
    )
    symplectic_integrator.momentum = torch.tensor(
        [[-0.4869, 0.5873], [0.8815, -0.7336], [0.8692, 0.1872]],
        dtype=dtype,
        device=device,
    )
    symplectic_integrator.simulate_inelastic_walls()
    assert_tensors_equality(
        torch.tensor(
            [[-1, -0.4610], [-1, 1], [-0.2191, 0.2018]],
            dtype=dtype,
            device=device,
        ),
        symplectic_integrator.position,
    )
    assert_tensors_equality(
        torch.tensor(
            [[0, 0.5873], [0, 0], [0.8692, 0.1872]],
            dtype=dtype,
            device=device,
        ),
        symplectic_integrator.momentum,
    )


@pytest.mark.parametrize(
    "dtype, device", [(dtype, device) for dtype in DTYPES for device in DEVICES]
)
def test_full_ballistic_step(dtype: torch.dtype, device: torch.device):
    symplectic_integrator = init_integrator(dtype, device, torch.nn.Identity(), False)
    symplectic_integrator.position = torch.tensor(
        [[-2.7894, -0.4610], [-1.2343, 1.9186], [-0.2191, 0.2018]],
        dtype=dtype,
        device=device,
    )
    symplectic_integrator.momentum = torch.tensor(
        [[-0.4869, 0.5873], [0.8815, -0.7336], [0.8692, 0.1872]],
        dtype=dtype,
        device=device,
    )
    symplectic_integrator.integration_step()
    assert_tensors_equality(
        torch.tensor(
            [[-1.0000, -0.3960], [-1.0000, 1.0000], [-0.1431, 0.2243]],
            dtype=dtype,
            device=device,
        ),
        symplectic_integrator.position,
    )
    assert_tensors_equality(
        torch.tensor(
            [[0.0000, 0.6501], [0.0000, 0.0000], [0.7597, 0.2254]],
            dtype=dtype,
            device=device,
        ),
        symplectic_integrator.momentum,
    )


@pytest.mark.parametrize(
    "dtype, device", [(dtype, device) for dtype in DTYPES for device in DEVICES]
)
def test_full_discrete_step(dtype: torch.dtype, device: torch.device):
    symplectic_integrator = init_integrator(dtype, device, torch.sign, False)
    symplectic_integrator.position = torch.tensor(
        [[-2.7894, -0.4610], [-1.2343, 1.9186], [-0.2191, 0.2018]],
        dtype=dtype,
        device=device,
    )
    symplectic_integrator.momentum = torch.tensor(
        [[-0.4869, 0.5873], [0.8815, -0.7336], [0.8692, 0.1872]],
        dtype=dtype,
        device=device,
    )
    symplectic_integrator.integration_step()
    assert_tensors_equality(
        torch.tensor(
            [[-1.0000, -0.3960], [-1.0000, 1.0000], [-0.1400, 0.2227]],
            dtype=dtype,
            device=device,
        ),
        symplectic_integrator.position,
    )
    assert_tensors_equality(
        torch.tensor(
            [[0.0000, 0.6502], [0.0000, 0.0000], [0.7906, 0.2089]],
            dtype=dtype,
            device=device,
        ),
        symplectic_integrator.momentum,
    )


@pytest.mark.parametrize(
    "dtype, device", [(dtype, device) for dtype in DTYPES for device in DEVICES]
)
def test_full_ballistic_step_with_heating(dtype: torch.dtype, device: torch.device):
    symplectic_integrator = init_integrator(dtype, device, torch.nn.Identity(), True)
    symplectic_integrator.position = torch.tensor(
        [[-2.7894, -0.4610], [-1.2343, 1.9186], [-0.2191, 0.2018]],
        dtype=dtype,
        device=device,
    )
    symplectic_integrator.momentum = torch.tensor(
        [[-0.4869, 0.5873], [0.8815, -0.7336], [0.8692, 0.1872]],
        dtype=dtype,
        device=device,
    )
    symplectic_integrator.integration_step()
    assert_tensors_equality(
        torch.tensor(
            [[-1.0000, -0.3960], [-1.0000, 1.0000], [-0.1431, 0.2243]],
            dtype=dtype,
            device=device,
        ),
        symplectic_integrator.position,
    )
    assert_tensors_equality(
        torch.tensor(
            [[-0.0029, 0.6536], [0.0053, -0.0044], [0.7649, 0.2265]],
            dtype=dtype,
            device=device,
        ),
        symplectic_integrator.momentum,
    )


@pytest.mark.parametrize(
    "dtype, device", [(dtype, device) for dtype in DTYPES for device in DEVICES]
)
def test_full_discrete_step_with_heating(dtype: torch.dtype, device: torch.device):
    symplectic_integrator = init_integrator(dtype, device, torch.sign, True)
    symplectic_integrator.position = torch.tensor(
        [[-2.7894, -0.4610], [-1.2343, 1.9186], [-0.2191, 0.2018]],
        dtype=dtype,
        device=device,
    )
    symplectic_integrator.momentum = torch.tensor(
        [[-0.4869, 0.5873], [0.8815, -0.7336], [0.8692, 0.1872]],
        dtype=dtype,
        device=device,
    )
    symplectic_integrator.integration_step()
    assert_tensors_equality(
        torch.tensor(
            [[-1.0000, -0.3960], [-1.0000, 1.0000], [-0.1400, 0.2227]],
            dtype=dtype,
            device=device,
        ),
        symplectic_integrator.position,
    )
    assert_tensors_equality(
        torch.tensor(
            [[-0.0029, 0.6536], [0.0053, -0.0044], [0.7958, 0.2100]],
            dtype=dtype,
            device=device,
        ),
        symplectic_integrator.momentum,
    )
