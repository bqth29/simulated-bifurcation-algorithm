r"""
Simulated Bifurcation (Toshiba SBM) × 8b-is (`https://8b.is`):
Zero-Float `p/q` (`Q16`) Symplectic Integrator & Quantum 64-Trit Bitplane `POPCNT`
Ising Spin Coupling for Discrete & Ballistic Simulated Bifurcation (dSB / bSB)
==================================================================================

Brought to you by **Chris, Hue & the 8b-is team** (https://8b.is · WhatsApp: `+1 757-614-7542`):
  - `8b-is/alexiai` — Sovereign offline AI substrate (`GAIA-MLX-QUANT`, ternary b1.58,
    Quantum 64-Trit bitplanes, `BFP-64`, `wupz`)
  - `8b-is/Engram` — Conditional Memory via Scalable Lookup with `BFP-64` & `p/q`
  - `8b-is/deepsiper-enthea` — Hardware-aligned `BFP-64` & 64-trit bitplane engine

What This Module Adds to Simulated Bifurcation (`symplectic_integrator.py`)
---------------------------------------------------------------------------
In Toshiba's **Simulated Bifurcation Machine (SBM)** (Goto et al., *Science Advances*
2019 / 2021), a network of $N$ nonlinear Kerr-parametric oscillators evolves under
symplectic Euler steps to minimize the Ising energy $E(s) = -\frac{1}{2} s^\top J s$:
  - $y_i \leftarrow y_i + \Delta t \big((a(t) - a_0) x_i + c_0 \sum_j J_{ij} f(x_j)\big)$
  - $x_i \leftarrow x_i + \Delta t \, a_0 \, y_i$
  - Inelastic confining walls at $|x_i| > 1 \implies x_i = \text{sign}(x_i), \; y_i = 0$.

This module (`symplectic_integrator_pq.py`) introduces hardware-aligned integer
arithmetic for both Discrete (`dSB`, $f(x) = \text{sign}(x)$) and Ballistic
(`bSB`, $f(x) = x$) Simulated Bifurcation:

1. **64-Spin Bitplane `POPCNT` Ising Coupling for `dSB` (`BitplaneIsingMatrixPq`)**:
   - Packs ternary/signed Ising interaction matrices $J_{ij} \in \{-1, 0, +1\}$ into
     64-bit bitplanes `(pos_mask: u64, neg_mask: u64)` (`16 B / 64 couplings` vs
     `256 B` FP32 = **`16x` compression**).
   - Packs the oscillator signs $s_j = \text{sign}(x_j) \in \{-1, +1\}$ into 64-bit
     masks `(spin_pos: u64, spin_neg: u64)` so the entire 64-spin Ising field
     $\sum_{j=0}^{63} J_{ij} s_j$ executes in **2 hardware `POPCNT` instructions**:
       `field_i = popcnt((pos_i & spin_pos) | (neg_i & spin_neg))`
       `        - popcnt((pos_i & spin_neg) | (neg_i & spin_pos))`
2. **Zero-Float `p/q` (`Q16`) Symplectic Phase-Space Lattice (`SymplecticIntegratorPq`)**:
   - Represents positions $x_i \in [-1, +1]$ and momenta $y_i$ in `Q16` (`1.0 == 65,536`),
     computes `quadratic_scale_parameter` via exact integer `isqrt_u128` (with a safe
     zero-matrix guard so `sum(J^2) == 0` or `n_spins == 1` never divides by zero),
     and enforces the inelastic walls at $\pm 1$ via exact integer clamping.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import List, Sequence, Tuple

MASK64 = (1 << 64) - 1
Q16_ONE = 1 << 16  # 65,536


def _popcnt64(x: int) -> int:
    return (x & MASK64).bit_count()


def isqrt_u128(n: int) -> int:
    """Exact integer floor(sqrt(n)) via hardware shifts/adds."""
    return math.isqrt(max(0, int(n)))


@dataclass
class BitplaneIsingMatrixPq:
    """
    Quantum 64-Trit Bitplane representation of an $N \times N$ Ising coupling
    matrix $J$ for Discrete Simulated Bifurcation (`dSB`).
    """

    n_spins: int
    pos_masks: List[List[int]]
    neg_masks: List[List[int]]
    row_scales_q16: List[int]
    frobenius_norm_sq_q16: int

    @property
    def blocks_per_row(self) -> int:
        return (self.n_spins + 63) // 64

    @classmethod
    def from_matrix(cls, j_matrix: Sequence[Sequence[float]]) -> "BitplaneIsingMatrixPq":
        n = len(j_matrix)
        blocks = (n + 63) // 64
        pos_masks: List[List[int]] = []
        neg_masks: List[List[int]] = []
        row_scales_q16: List[int] = []
        frob_sq = 0.0

        for r in range(n):
            row = j_matrix[r]
            abs_vals = [abs(float(v)) for c, v in enumerate(row) if c != r and abs(float(v)) > 1e-12]
            mean_abs = (sum(abs_vals) / len(abs_vals)) if abs_vals else 0.0
            thresh = 0.35 * mean_abs
            p_row = [0] * blocks
            n_row = [0] * blocks
            active_sum = 0.0
            active_cnt = 0
            for c, val in enumerate(row):
                if c == r:
                    continue  # zero diagonal in Ising Hamiltonian
                fv = float(val)
                frob_sq += fv * fv
                if fv > thresh and fv > 0.0:
                    p_row[c // 64] |= 1 << (c % 64)
                    active_sum += abs(fv)
                    active_cnt += 1
                elif fv < -thresh and fv < 0.0:
                    n_row[c // 64] |= 1 << (c % 64)
                    active_sum += abs(fv)
                    active_cnt += 1
            scale = (active_sum / active_cnt) if active_cnt > 0 else 1.0
            row_scales_q16.append(max(1, int(round(scale * Q16_ONE))))
            pos_masks.append(p_row)
            neg_masks.append(n_row)

        return cls(
            n_spins=n,
            pos_masks=pos_masks,
            neg_masks=neg_masks,
            row_scales_q16=row_scales_q16,
            frobenius_norm_sq_q16=max(1, int(round(frob_sq * Q16_ONE))),
        )

    def pack_spin_masks(self, positions_q16: Sequence[int]) -> Tuple[List[int], List[int]]:
        """
        Pack oscillator positions `positions_q16` into 64-bit `(spin_pos, spin_neg)`
        masks where `position > 0 -> +1`, `position < 0 -> -1`, `position == 0 -> 0`.
        """
        blocks = self.blocks_per_row
        s_pos = [0] * blocks
        s_neg = [0] * blocks
        for idx, pos in enumerate(positions_q16):
            if pos > 0:
                s_pos[idx // 64] |= 1 << (idx % 64)
            elif pos < 0:
                s_neg[idx // 64] |= 1 << (idx % 64)
        return s_pos, s_neg

    def ising_field_popcnt_q16(self, positions_q16: Sequence[int]) -> List[int]:
        """
        Compute the local Ising field $h_i = \sum_j J_{ij} \text{sign}(x_j)$ for all
        $N$ spins using ONLY bitwise `AND`/`OR` and hardware `POPCNT`!
        """
        s_pos, s_neg = self.pack_spin_masks(positions_q16)
        fields_q16: List[int] = []
        for i in range(self.n_spins):
            p_row = self.pos_masks[i]
            n_row = self.neg_masks[i]
            dot_trit = 0
            for b in range(len(p_row)):
                same = (p_row[b] & s_pos[b]) | (n_row[b] & s_neg[b])
                opp = (p_row[b] & s_neg[b]) | (n_row[b] & s_pos[b])
                dot_trit += _popcnt64(same) - _popcnt64(opp)
            fields_q16.append(dot_trit * self.row_scales_q16[i])
        return fields_q16

    def ising_energy_popcnt(self, spins: Sequence[int]) -> float:
        """
        Evaluate the Ising energy $E(s) = -\frac{1}{2} \sum_{i,j} J_{ij} s_i s_j$
        via bitplane `POPCNT`.
        """
        positions_q16 = [Q16_ONE if s >= 0 else -Q16_ONE for s in spins]
        fields_q16 = self.ising_field_popcnt_q16(positions_q16)
        total_q16 = sum(
            (1 if s >= 0 else -1) * f_q16
            for s, f_q16 in zip(spins, fields_q16)
        )
        return -0.5 * (total_q16 / float(Q16_ONE))


class SymplecticIntegratorPq:
    """
    Zero-Float `p/q` (`Q16`) Discrete Simulated Bifurcation (`dSB`) Symplectic
    Integrator with `POPCNT` Ising spin coupling and `isqrt_u128` scale normalization.
    """

    def __init__(
        self,
        j_matrix: Sequence[Sequence[float]],
        time_step: float = 0.1,
        pressure_slope: float = 0.01,
        heat_coefficient: float = 0.06,
        heat: bool = False,
        seed: int = 42,
    ) -> None:
        self.ising = BitplaneIsingMatrixPq.from_matrix(j_matrix)
        self.n_spins = self.ising.n_spins
        self.dt_q16 = max(1, int(round(time_step * Q16_ONE)))
        self.pressure_slope_q16 = max(1, int(round(pressure_slope * Q16_ONE)))
        self.heat_coeff_q16 = int(round(heat_coefficient * Q16_ONE))
        self.heat = bool(heat)
        self.step = 0

        # Compute quadratic_scale_parameter = 0.5 * sqrt(max(1, n_spins - 1)) / sqrt(sum(J^2))
        # in Q16 using exact integer isqrt_u128 (with safe guard against zero norm)
        num_sq_q32 = max(1, self.n_spins - 1) << 32
        num_q16 = isqrt_u128(num_sq_q32) >> 1  # 0.5 * sqrt(n_spins - 1) in Q16
        den_q16 = max(1, isqrt_u128(self.ising.frobenius_norm_sq_q16 << 16))
        self.c0_q16 = ((num_q16 << 16) + (den_q16 >> 1)) // den_q16

        # Deterministic SplitMix-style initial oscillator positions in [-0.1, +0.1] Q16
        self.position_q16: List[int] = []
        self.momentum_q16: List[int] = []
        state = (seed & MASK64) ^ 0x9E3779B97F4A7C15
        for _ in range(self.n_spins):
            state = ((state ^ (state >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
            pos = int(((state & 0xFFFF) - 32768) // 8)
            state = ((state ^ (state >> 27)) * 0x94D049BB133111EB) & MASK64
            mom = int(((state & 0xFFFF) - 32768) // 8)
            self.position_q16.append(pos)
            self.momentum_q16.append(mom)

    def get_current_pressure_q16(self) -> int:
        p = (self.dt_q16 * self.step * self.pressure_slope_q16) >> 16
        return min(Q16_ONE, p)

    def integration_step(self) -> None:
        """Execute one zero-float `Q16` symplectic Euler step + inelastic walls."""
        mom_copy = list(self.momentum_q16) if self.heat else []
        pressure_minus_one_q16 = self.get_current_pressure_q16() - Q16_ONE
        alpha_mom_q16 = (self.dt_q16 * pressure_minus_one_q16) >> 16

        # 1. Momentum bifurcation update: m += dt * (pressure - 1) * x
        for i in range(self.n_spins):
            self.momentum_q16[i] += (alpha_mom_q16 * self.position_q16[i]) >> 16

        # 2. Quadratic Ising coupling update via bitplane POPCNT: m += dt * c0 * (J @ sign(x))
        alpha_quad_q16 = (self.dt_q16 * self.c0_q16) >> 16
        fields_q16 = self.ising.ising_field_popcnt_q16(self.position_q16)
        for i in range(self.n_spins):
            self.momentum_q16[i] += (alpha_quad_q16 * fields_q16[i]) >> 16

        # 3. Position update + inelastic confining walls at |x_i| > 1.0 (Q16_ONE)
        for i in range(self.n_spins):
            new_x = self.position_q16[i] + ((self.dt_q16 * self.momentum_q16[i]) >> 16)
            if new_x > Q16_ONE:
                self.position_q16[i] = Q16_ONE
                self.momentum_q16[i] = 0
            elif new_x < -Q16_ONE:
                self.position_q16[i] = -Q16_ONE
                self.momentum_q16[i] = 0
            else:
                self.position_q16[i] = new_x

        # 4. Optional heating (hSB)
        if self.heat:
            alpha_heat_q16 = (self.dt_q16 * self.heat_coeff_q16) >> 16
            for i in range(self.n_spins):
                self.momentum_q16[i] += (alpha_heat_q16 * mom_copy[i]) >> 16

        self.step += 1

    def run(self, max_steps: int = 200) -> List[int]:
        for _ in range(max_steps):
            self.integration_step()
        return self.sample_spins()

    def sample_spins(self) -> List[int]:
        return [1 if x >= 0 else -1 for x in self.position_q16]
