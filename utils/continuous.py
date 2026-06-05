from __future__ import annotations

import numpy as np
from scipy.linalg import expm

from utils.discrete import is_absorbing_state


def build_generator_matrix_from_transition_matrix(
    P: np.ndarray,
    states: list[tuple[int, int]],
    grid: np.ndarray,
    transition_rate: float,
) -> np.ndarray:
    """
    Build a CTMC generator matrix Q from a DTMC transition matrix P.

    For non-absorbing states:

        q_ij = lambda P_ij, i != j

    and:

        q_ii = -sum_{j != i} q_ij

    Absorbing states have zero rows.
    """
    Q = np.zeros_like(P, dtype=float)

    for i, state in enumerate(states):
        if is_absorbing_state(grid, state):
            Q[i, i] = 0.0
            continue

        for j in range(len(states)):
            if i != j:
                Q[i, j] = transition_rate * P[i, j]

        Q[i, i] = -np.sum(Q[i, :])

    return Q


def transition_matrix_over_time(
    Q: np.ndarray,
    time_t: float,
) -> np.ndarray:
    """
    Compute P(t) = exp(Q t).
    """
    return expm(Q * time_t)