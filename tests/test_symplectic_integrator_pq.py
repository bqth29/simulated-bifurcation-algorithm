import os
import sys
import unittest

sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "src",
            "simulated_bifurcation",
            "optimizer",
        )
    ),
)

from symplectic_integrator_pq import (
    BitplaneIsingMatrixPq,
    Q16_ONE,
    SymplecticIntegratorPq,
)


class TestSimulatedBifurcationPq(unittest.TestCase):
    def test_bitplane_ising_popcnt_field_matches_exact(self):
        # 8-spin frustrated anti-ferromagnetic ring + cross couplings
        n = 8
        j_mat = [[0.0] * n for _ in range(n)]
        for i in range(n):
            j_mat[i][(i + 1) % n] = -1.0
            j_mat[(i + 1) % n][i] = -1.0
            j_mat[i][(i + 3) % n] = 1.0
            j_mat[(i + 3) % n][i] = 1.0

        ising = BitplaneIsingMatrixPq.from_matrix(j_mat)
        spins = [1, -1, 1, -1, 1, 1, -1, -1]
        pos_q16 = [s * Q16_ONE for s in spins]
        fields_q16 = ising.ising_field_popcnt_q16(pos_q16)

        for i in range(n):
            exact_field = sum(j_mat[i][j] * spins[j] for j in range(n))
            self.assertAlmostEqual(fields_q16[i] / float(Q16_ONE), exact_field, delta=1e-5)

    def test_dsb_symplectic_integrator_pq_finds_ground_state_maxcut(self):
        # 6-vertex bipartite graph K_{3,3}: anti-ferromagnetic couplings J_{ij} = -1
        # between {0,1,2} and {3,4,5}. Ground state has all 9 edges cut -> energy = -9.0!
        n = 6
        j_mat = [[0.0] * n for _ in range(n)]
        for u in range(3):
            for v in range(3, 6):
                j_mat[u][v] = -1.0
                j_mat[v][u] = -1.0

        integrator = SymplecticIntegratorPq(
            j_mat,
            time_step=0.1,
            pressure_slope=0.02,
            heat=True,
            seed=7,
        )
        spins = integrator.run(max_steps=120)
        energy = integrator.ising.ising_energy_popcnt(spins)
        # Ground state energy of K_{3,3} with J=-1 on bipartite edges is -9.0
        self.assertAlmostEqual(energy, -9.0, delta=1e-4)

    def test_zero_coupling_and_single_spin_guard(self):
        # Verify n_spins=1 and all-zero J never divide by zero
        integrator = SymplecticIntegratorPq([[0.0]], time_step=0.1)
        spins = integrator.run(max_steps=5)
        self.assertEqual(len(spins), 1)


if __name__ == "__main__":
    unittest.main()
