from __future__ import annotations

import random

import numpy as np


ACTIONS: dict[str, tuple[int, int]] = {
    "UP": (-1, 0),
    "DOWN": (1, 0),
    "LEFT": (0, -1),
    "RIGHT": (0, 1),
}


def is_inside_grid(grid: np.ndarray, pos: tuple[int, int]) -> bool:
    r, c = pos
    n_rows, n_cols = grid.shape
    return 0 <= r < n_rows and 0 <= c < n_cols


def is_absorbing_state(grid: np.ndarray, pos: tuple[int, int]) -> bool:
    r, c = pos
    return grid[r, c] in {"H", "V"}


def move(grid: np.ndarray, pos: tuple[int, int], action: str) -> tuple[int, int]:
    """
    If the robot tries to move outside the grid, it remains in the same cell.
    """
    dr, dc = ACTIONS[action]
    new_pos = (pos[0] + dr, pos[1] + dc)

    if is_inside_grid(grid, new_pos):
        return new_pos

    return pos


def get_valid_actions(grid: np.ndarray, pos: tuple[int, int]) -> list[str]:
    """
    Valid intended actions are actions that do not leave the grid.

    Absorbing states have no actions.
    """
    if is_absorbing_state(grid, pos):
        return []

    valid_actions: list[str] = []

    for action in ACTIONS:
        next_pos = move(grid, pos, action)

        if next_pos != pos:
            valid_actions.append(action)

    return valid_actions


def choose_policy_action(
    grid: np.ndarray,
    pos: tuple[int, int],
    policy_type: str,
    victim_pos: tuple[int, int],
) -> str | None:
    valid_actions = get_valid_actions(grid, pos)

    if len(valid_actions) == 0:
        return None

    if policy_type == "Random":
        return random.choice(valid_actions)

    r, c = pos
    tr, tc = victim_pos

    preferred_actions: list[str] = []

    if abs(tr - r) >= abs(tc - c):
        if tr > r:
            preferred_actions.append("DOWN")
        elif tr < r:
            preferred_actions.append("UP")

        if tc > c:
            preferred_actions.append("RIGHT")
        elif tc < c:
            preferred_actions.append("LEFT")
    else:
        if tc > c:
            preferred_actions.append("RIGHT")
        elif tc < c:
            preferred_actions.append("LEFT")

        if tr > r:
            preferred_actions.append("DOWN")
        elif tr < r:
            preferred_actions.append("UP")

    for action in preferred_actions:
        if action in valid_actions:
            return action

    return random.choice(valid_actions)


def get_policy_action_probabilities(
    grid: np.ndarray,
    pos: tuple[int, int],
    policy_type: str,
    victim_pos: tuple[int, int],
) -> dict[str, float]:
    """
    Returns π(a | s).
    """
    valid_actions = get_valid_actions(grid, pos)

    if len(valid_actions) == 0:
        return {}

    if policy_type == "Random":
        prob = 1.0 / len(valid_actions)
        return {action: prob for action in valid_actions}

    greedy_action = choose_policy_action(
        grid=grid,
        pos=pos,
        policy_type=policy_type,
        victim_pos=victim_pos,
    )

    if greedy_action is None:
        return {}

    return {greedy_action: 1.0}


def sample_actual_action(intended_action: str, success_prob: float) -> str:
    """
    With probability success_prob, the robot performs the intended action.

    Otherwise, it slips into one of the other directions uniformly.
    """
    other_actions = [a for a in ACTIONS if a != intended_action]

    possible_actions = [intended_action] + other_actions

    probabilities = [success_prob] + [
        (1 - success_prob) / len(other_actions)
        for _ in other_actions
    ]

    return random.choices(possible_actions, probabilities)[0]


def get_environment_action_probabilities(
    intended_action: str,
    success_prob: float,
) -> dict[str, float]:
    """
    Returns P(actual_action | intended_action).
    """
    other_actions = [a for a in ACTIONS if a != intended_action]

    action_probs = {
        intended_action: success_prob,
    }

    slip_prob = (1 - success_prob) / len(other_actions)

    for action in other_actions:
        action_probs[action] = slip_prob

    return action_probs