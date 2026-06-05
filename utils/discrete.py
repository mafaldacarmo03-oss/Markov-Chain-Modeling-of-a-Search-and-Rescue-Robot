from __future__ import annotations

import random
from collections import deque

import numpy as np
import pandas as pd


from math import gcd

from utils.policies import (
    ACTIONS,
    choose_policy_action,
    get_environment_action_probabilities,
    get_policy_action_probabilities,
    get_valid_actions,
    move,
    sample_actual_action,
)


def neighbors(
    pos: tuple[int, int],
    n_rows: int,
    n_cols: int,
):
    r, c = pos

    for dr, dc in ACTIONS.values():
        nr, nc = r + dr, c + dc

        if 0 <= nr < n_rows and 0 <= nc < n_cols:
            yield nr, nc


def path_exists(
    start: tuple[int, int],
    victim: tuple[int, int],
    hazards: set[tuple[int, int]],
    n_rows: int,
    n_cols: int,
) -> bool:
    """
    Checks that there is at least one safe path from R to V.
    Hazards are terminal failure states, so the safe path avoids H.
    """
    blocked = set(hazards)

    queue = deque([start])
    visited = {start}

    while queue:
        pos = queue.popleft()

        if pos == victim:
            return True

        for nxt in neighbors(pos, n_rows, n_cols):
            if nxt in visited:
                continue

            if nxt in blocked:
                continue

            visited.add(nxt)
            queue.append(nxt)

    return False


def generate_random_grid(
    size: int,
    n_hazards: int,
    seed: int | None = None,
) -> np.ndarray:
    """
    Generates a random Frozen-Lake-style rescue grid.

    R = robot start
    V = victim / success absorbing state
    H = hole / failure absorbing state
    . = safe frozen cell
    """
    rng = random.Random(seed)

    n_rows = size
    n_cols = size

    all_cells = [(r, c) for r in range(n_rows) for c in range(n_cols)]

    max_attempts = 1000

    for _ in range(max_attempts):
        start = rng.choice(all_cells)

        remaining = [cell for cell in all_cells if cell != start]
        victim = rng.choice(remaining)

        remaining = [cell for cell in remaining if cell != victim]
        rng.shuffle(remaining)

        hazards = set(remaining[:n_hazards])

        if path_exists(start, victim, hazards, n_rows, n_cols):
            grid = np.full((n_rows, n_cols), ".", dtype=object)

            for r, c in hazards:
                grid[r, c] = "H"

            grid[start] = "R"
            grid[victim] = "V"

            return grid

    raise RuntimeError("Could not generate a valid grid. Try fewer holes.")


def get_grid_positions(grid: np.ndarray) -> tuple[tuple[int, int], tuple[int, int]]:
    start_pos = tuple(np.argwhere(grid == "R")[0])
    victim_pos = tuple(np.argwhere(grid == "V")[0])

    return start_pos, victim_pos


def is_hazard(grid: np.ndarray, pos: tuple[int, int]) -> bool:
    r, c = pos
    return grid[r, c] == "H"


def is_victim(grid: np.ndarray, pos: tuple[int, int]) -> bool:
    r, c = pos
    return grid[r, c] == "V"


def is_absorbing_state(grid: np.ndarray, pos: tuple[int, int]) -> bool:
    return is_hazard(grid, pos) or is_victim(grid, pos)


def get_markov_states(grid: np.ndarray) -> list[tuple[int, int]]:
    """
    Every grid cell is a Markov state.

    Therefore:
        4x4 -> 16 states
        8x8 -> 64 states
    """
    n_rows, n_cols = grid.shape

    return [
        (r, c)
        for r in range(n_rows)
        for c in range(n_cols)
    ]


def get_transition_distribution(
    grid: np.ndarray,
    pos: tuple[int, int],
    policy_type: str,
    success_prob: float,
    victim_pos: tuple[int, int],
) -> dict[tuple[int, int], float]:
    """
    Computes P_pi(s' | s).

    The state is only the robot position.
    """
    if is_absorbing_state(grid, pos):
        return {pos: 1.0}

    policy_probs = get_policy_action_probabilities(
        grid=grid,
        pos=pos,
        policy_type=policy_type,
        victim_pos=victim_pos,
    )

    transition_probs: dict[tuple[int, int], float] = {}

    for intended_action, policy_prob in policy_probs.items():
        env_probs = get_environment_action_probabilities(
            intended_action=intended_action,
            success_prob=success_prob,
        )

        for actual_action, env_prob in env_probs.items():
            next_pos = move(grid, pos, actual_action)
            prob = policy_prob * env_prob
            transition_probs[next_pos] = transition_probs.get(next_pos, 0.0) + prob

    return transition_probs


def build_transition_matrix(
    grid: np.ndarray,
    policy_type: str,
    success_prob: float,
    victim_pos: tuple[int, int],
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    states = get_markov_states(grid)
    state_to_idx = {state: i for i, state in enumerate(states)}

    P = np.zeros((len(states), len(states)))

    for state in states:
        i = state_to_idx[state]

        distribution = get_transition_distribution(
            grid=grid,
            pos=state,
            policy_type=policy_type,
            success_prob=success_prob,
            victim_pos=victim_pos,
        )

        for next_state, prob in distribution.items():
            j = state_to_idx[next_state]
            P[i, j] += prob

    return P, states


def reset_simulation_state(
    session_state,
    start_pos: tuple[int, int],
) -> None:
    session_state.robot_pos = start_pos
    session_state.path = [start_pos]
    session_state.terminated = False
    session_state.status = "Running"
    session_state.last_transition = None


def step_robot(
    session_state,
    grid: np.ndarray,
    policy_type: str,
    success_prob: float,
    victim_pos: tuple[int, int],
) -> None:
    if session_state.terminated:
        return

    current_pos = session_state.robot_pos

    if is_absorbing_state(grid, current_pos):
        session_state.terminated = True
        return

    intended_action = choose_policy_action(
        grid=grid,
        pos=current_pos,
        policy_type=policy_type,
        victim_pos=victim_pos,
    )

    if intended_action is None:
        session_state.terminated = True
        session_state.status = "Terminated: no valid actions available."
        return

    actual_action = sample_actual_action(
        intended_action=intended_action,
        success_prob=success_prob,
    )

    next_pos = move(grid, current_pos, actual_action)

    session_state.robot_pos = next_pos
    session_state.path.append(next_pos)

    if is_hazard(grid, next_pos):
        session_state.terminated = True
        session_state.status = "Failure: robot fell into a hole."

    elif is_victim(grid, next_pos):
        session_state.terminated = True
        session_state.status = "Success: victim reached."

    else:
        session_state.status = "Running"

    session_state.last_transition = {
        "from_state": current_pos,
        "to_state": next_pos,
        "valid_actions": get_valid_actions(grid, current_pos),
        "intended_action": intended_action,
        "actual_action": actual_action,
    }


def state_table(
    grid: np.ndarray,
    states: list[tuple[int, int]],
) -> pd.DataFrame:
    rows = []

    for i, state in enumerate(states):
        cell_type = grid[state]

        if cell_type == "R":
            description = "Start state"
        elif cell_type == "V":
            description = "Success absorbing state"
        elif cell_type == "H":
            description = "Failure absorbing state"
        elif cell_type == ".":
            description = "Safe frozen cell"
        else:
            description = "Other"

        rows.append(
            {
                "State ID": f"s{i}",
                "Position": str(state),
                "Cell type": cell_type,
                "Description": description,
                "Absorbing": is_absorbing_state(grid, state),
            }
        )

    return pd.DataFrame(rows)


def matrix_to_dataframe(
    matrix: np.ndarray,
    states: list[tuple[int, int]],
) -> pd.DataFrame:
    labels = [f"s{i}" for i in range(len(states))]
    return pd.DataFrame(matrix, index=labels, columns=labels)


def initial_state_distribution(
    states: list[tuple[int, int]],
    initial_pos: tuple[int, int],
) -> np.ndarray:
    """
    Build the initial probability distribution pi_0.

    If the robot starts in state initial_pos, then:
        pi_0[initial_state] = 1
        pi_0[other_states] = 0
    """
    pi0 = np.zeros(len(states), dtype=float)

    initial_index = states.index(initial_pos)
    pi0[initial_index] = 1.0

    return pi0


def state_distribution_after_n_steps(
    P: np.ndarray,
    states: list[tuple[int, int]],
    initial_pos: tuple[int, int],
    n: int,
) -> np.ndarray:
    """
    Compute pi_n = pi_0 P^n.

    Here:
        pi_0 is the initial state distribution.
        P is the one-step transition matrix.
        n is the number of transition steps.
    """
    if n < 0:
        raise ValueError("n must be nonnegative.")

    pi0 = initial_state_distribution(states, initial_pos)

    return pi0 @ np.linalg.matrix_power(P, n)


def evolve_state_distribution(
    pi: np.ndarray,
    P: np.ndarray,
) -> np.ndarray:
    """
    Compute one probability update:

        pi_next = pi P
    """
    return pi @ P


def distribution_to_dataframe(
    distribution: np.ndarray,
    states: list[tuple[int, int]],
) -> pd.DataFrame:
    """
    Convert a probability distribution over states into a dataframe.
    """
    return pd.DataFrame(
        {
            "State": [f"s{i}" for i in range(len(states))],
            "Position": [str(state) for state in states],
            "Probability": distribution,
        }
    )




def _positive_edges(P: np.ndarray, tol: float = 1e-12) -> dict[int, list[int]]:
    """
    Directed graph induced by transition matrix P.
    Edge i -> j exists if P[i,j] > tol.
    """
    n = P.shape[0]
    graph = {i: [] for i in range(n)}

    for i in range(n):
        for j in range(n):
            if P[i, j] > tol:
                graph[i].append(j)

    return graph


def _tarjan_scc(graph: dict[int, list[int]]) -> list[list[int]]:
    """
    Tarjan algorithm for strongly connected components.
    Each strongly connected component is one communication class.
    """
    index = 0
    stack = []
    on_stack = set()
    indices = {}
    lowlink = {}
    components = []

    def strongconnect(v):
        nonlocal index

        indices[v] = index
        lowlink[v] = index
        index += 1

        stack.append(v)
        on_stack.add(v)

        for w in graph[v]:
            if w not in indices:
                strongconnect(w)
                lowlink[v] = min(lowlink[v], lowlink[w])
            elif w in on_stack:
                lowlink[v] = min(lowlink[v], indices[w])

        if lowlink[v] == indices[v]:
            component = []

            while True:
                w = stack.pop()
                on_stack.remove(w)
                component.append(w)

                if w == v:
                    break

            components.append(sorted(component))

    for v in graph:
        if v not in indices:
            strongconnect(v)

    return components


def _is_closed_class(
    component: list[int],
    graph: dict[int, list[int]],
) -> bool:
    """
    A class is closed if there are no positive-probability transitions
    from states inside the class to states outside the class.
    """
    component_set = set(component)

    for i in component:
        for j in graph[i]:
            if j not in component_set:
                return False

    return True


def _class_period(
    component: list[int],
    graph: dict[int, list[int]],
) -> int | None:
    """
    Compute the period of a communication class.

    The period is the gcd of all possible return times.

    For a singleton class with a self-loop, period = 1.
    For a singleton class without a self-loop, there is no return cycle,
    so the period is returned as None.
    """
    component_set = set(component)

    if len(component) == 1:
        state = component[0]

        if state in graph[state]:
            return 1

        return None

    start = component[0]

    distances = {start: 0}
    queue = deque([start])

    while queue:
        u = queue.popleft()

        for v in graph[u]:
            if v not in component_set:
                continue

            if v not in distances:
                distances[v] = distances[u] + 1
                queue.append(v)

    period = 0

    for u in component:
        for v in graph[u]:
            if v not in component_set:
                continue

            cycle_difference = distances[u] + 1 - distances[v]
            period = gcd(period, abs(cycle_difference))

    if period == 0:
        return None

    return period


def classify_markov_chain(
    P: np.ndarray,
    states: list[tuple[int, int]],
    grid: np.ndarray,
    tol: float = 1e-12,
) -> pd.DataFrame:
    """
    Classify the Markov chain into communication classes.

    For finite-state Markov chains:
      - closed classes are recurrent;
      - non-closed classes are transient.

    Period:
      - period = 1 means aperiodic;
      - period > 1 means periodic.
    """
    graph = _positive_edges(P, tol=tol)
    components = _tarjan_scc(graph)

    rows = []

    for class_id, component in enumerate(components):
        closed = _is_closed_class(component, graph)
        recurrence = "recurrent" if closed else "transient"

        period = _class_period(component, graph)

        if period is None:
            periodicity = "undefined / no return cycle"
        elif period == 1:
            periodicity = "aperiodic"
        else:
            periodicity = f"periodic, period {period}"

        positions = [states[i] for i in component]
        cell_types = [grid[pos] for pos in positions]

        absorbing = (
            len(component) == 1
            and P[component[0], component[0]] > 1 - tol
        )

        rows.append(
            {
                "Class": f"C{class_id}",
                "State IDs": ", ".join(f"s{i}" for i in component),
                "Positions": ", ".join(str(pos) for pos in positions),
                "Cell types": ", ".join(str(cell) for cell in cell_types),
                "Closed": closed,
                "Recurrent / transient": recurrence,
                "Period": period if period is not None else "-",
                "Periodicity": periodicity,
                "Absorbing": absorbing,
            }
        )

    return pd.DataFrame(rows)


def compute_success_before_hazard_probabilities(P, states, grid, victim_pos):
    """
    Compute h_i = P(reach victim before hazard | X_0 = s_i).

    Boundary conditions:
        h_i = 1 for victim state
        h_i = 0 for hazard states

    For all other states:
        h_i = sum_j P_ij h_j
    """
    n = len(states)

    victim_idx = states.index(victim_pos)

    hazard_indices = [
        i for i, state in enumerate(states)
        if is_hazard(grid, state)
    ]

    absorbing_indices = set(hazard_indices + [victim_idx])

    transient_indices = [
        i for i in range(n)
        if i not in absorbing_indices
    ]

    h = np.zeros(n, dtype=float)
    h[victim_idx] = 1.0

    if len(hazard_indices) == 0:
        # No hazard exists. Success before hazard is trivially 1
        # for all states that can eventually reach the victim.
        # For this app, we use 1 everywhere as the meaningful interpretation.
        h[:] = 1.0
        return h

    if len(transient_indices) > 0:
        Q = P[np.ix_(transient_indices, transient_indices)]
        R_to_success = P[np.ix_(transient_indices, [victim_idx])].reshape(-1)

        I = np.eye(len(transient_indices))

        h_transient = np.linalg.solve(I - Q, R_to_success)

        for local_idx, global_idx in enumerate(transient_indices):
            h[global_idx] = h_transient[local_idx]

    return h


def build_absorbing_reordered_matrix(P, states, grid, victim_pos):
    victim_idx = states.index(victim_pos)

    hazard_indices = [
        i for i, state in enumerate(states)
        if is_hazard(grid, state)
    ]

    absorbing_indices = hazard_indices + [victim_idx]

    # Remove duplicates, preserving order
    absorbing_indices = list(dict.fromkeys(absorbing_indices))

    transient_indices = [
        i for i in range(len(states))
        if i not in absorbing_indices
    ]

    reordered_indices = transient_indices + absorbing_indices

    P_reordered = P[np.ix_(reordered_indices, reordered_indices)]

    reordered_states = [states[i] for i in reordered_indices]
    reordered_labels = [f"s{i} {states[i]}" for i in reordered_indices]

    n_transient = len(transient_indices)
    n_absorbing = len(absorbing_indices)

    Q = P_reordered[:n_transient, :n_transient]
    R = P_reordered[:n_transient, n_transient:]
    zero_block = P_reordered[n_transient:, :n_transient]
    I_block = P_reordered[n_transient:, n_transient:]

    return {
        "P_reordered": P_reordered,
        "reordered_indices": reordered_indices,
        "reordered_states": reordered_states,
        "reordered_labels": reordered_labels,
        "transient_indices": transient_indices,
        "absorbing_indices": absorbing_indices,
        "n_transient": n_transient,
        "n_absorbing": n_absorbing,
        "Q": Q,
        "R": R,
        "zero_block": zero_block,
        "I_block": I_block,
    }